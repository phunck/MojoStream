# src/linalg/kernels.mojo
# B-Pack Q4 Matmul Kernel – extrahiert aus dem Benchmark-Stand.
# Einziger Kernel der exportiert wird: matmul_q4_bpack (128 GFLOPS auf AVX2).
from std.algorithm.functional import parallelize
from std.memory import UnsafePointer, memset_zero
from std.sys.info import simd_width_of, num_logical_cores
from std.random import rand
from std.math import sqrt, cos, sin, exp, log

# ---------------------------------------------------------------------------
# Compile-Zeit Konstanten
# ---------------------------------------------------------------------------
alias DT     = DType.float32
alias SIMD_W = simd_width_of[DT]()  # 8 (AVX2) oder 16 (AVX-512)
alias HALF_W = SIMD_W // 2          # Bytes pro SIMD-Vektor in gepacktem uint8

alias MR       = 4    # Mikro-Kernel Zeilen
alias NR_SIMD  = 2    # SIMD-Vektoren pro Spalten-Tile
alias NR       = NR_SIMD * SIMD_W  # skalare Spalten pro Tile (= 16)
alias BM       = 64   # M-Block für BPack-Kernel
alias BK       = 128  # K-Tile (B-Puffer = BK×NR×4 = 8 KB → L1d)

# L2-Cache-Blocking für große N (A-Panel-Packing)
# i7-7500U: 256 KB L2 pro physischem Kern, 2 HT-Threads teilen den L2.
# Pro Thread: MC×BK×4 = 128×128×4 = 64 KB + B-Puffer 8 KB = 72 KB ≪ 128 KB (halbes L2 ✓)
alias MC = 128  # MC-Block-Größe für A-Packing

alias PtrT  = UnsafePointer[Scalar[DT], MutAnyOrigin]
alias U8Ptr = UnsafePointer[UInt8,      MutAnyOrigin]

# ---------------------------------------------------------------------------
# Matrix – row-major, float32, heap-alloziert
# ---------------------------------------------------------------------------
struct Matrix(Movable):
    var storage: List[Scalar[DT]]
    var rows: Int
    var cols: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.storage = List[Scalar[DT]]()
        self.storage.resize(rows * cols, 0)

    @always_inline
    fn data(self) -> PtrT:
        return rebind[PtrT](self.storage.unsafe_ptr())

    @always_inline
    fn load[w: Int](self, r: Int, c: Int) -> SIMD[DT, w]:
        return self.data().load[width=w](r * self.cols + c)

    @always_inline
    fn store[w: Int](self, r: Int, c: Int, v: SIMD[DT, w]):
        self.data().store(r * self.cols + c, v)

    fn fill_random(mut self):
        rand[DT](self.data(), self.rows * self.cols)

    fn zero(mut self):
        memset_zero(self.data(), self.rows * self.cols)


# ---------------------------------------------------------------------------
# Q4Matrix – 4-bit gepackte Gewichte, symmetrische globale Skala
# Layout: storage[k * packed_cols + j] enthält zwei Nibbles:
#   low  = byte & 0x0F  → logische Spalte 2j
#   high = byte >> 4    → logische Spalte 2j + 1
# Dequant: w_fp32 = (nibble - 8) * scale
# ---------------------------------------------------------------------------
struct Q4Matrix(Movable):
    var storage: List[UInt8]
    var rows: Int
    var packed_cols: Int  # = logical_cols // 2
    var scale: Float32

    fn __init__(out self, rows: Int, logical_cols: Int, scale: Float32):
        self.rows        = rows
        self.packed_cols = logical_cols // 2
        self.scale       = scale
        self.storage     = List[UInt8]()
        self.storage.resize(rows * self.packed_cols, 0)

    @always_inline
    fn data(self) -> U8Ptr:
        return rebind[U8Ptr](self.storage.unsafe_ptr())

    fn logical_cols(self) -> Int:
        return self.packed_cols * 2

    fn fill_random(mut self):
        rand[DType.uint8](self.data(), self.rows * self.packed_cols)


# ---------------------------------------------------------------------------
# dequant_vec – inline Dequantisierung
# Lädt HALF_W gepackte Bytes, entpackt via >> 4 und & 0x0F,
# wendet Offset -8 und Skala an. Ergebnis lebt nur im SIMD-Register.
# ---------------------------------------------------------------------------
@always_inline
fn dequant_vec(ptr: U8Ptr, byte_off: Int, sv: SIMD[DT, HALF_W]) -> SIMD[DT, SIMD_W]:
    var p    = ptr.load[width=HALF_W](byte_off)
    var lo   = (p & SIMD[DType.uint8, HALF_W](0x0F)).cast[DType.int32]()
    var hi   = ((p >> SIMD[DType.uint8, HALF_W](4)) & SIMD[DType.uint8, HALF_W](0x0F)).cast[DType.int32]()
    var bias = SIMD[DType.int32, HALF_W](8)
    return rebind[SIMD[DT, SIMD_W]](
        ((lo - bias).cast[DT]() * sv).interleave((hi - bias).cast[DT]() * sv)
    )


# ---------------------------------------------------------------------------
# matmul_q4_bpack – B-Panel-Pack + MR=4 Register-Blocked (128 GFLOPS auf AVX2)
#
# Für jedes (kt, nt)-Tile: B[kt:kt+BK, nt:nt+NR] wird einmal dequantisiert
# und als fp32 in einen thread-lokalen 8-KB-L1-Puffer geschrieben.
# Alle BM/MR = 16 m-Tiles desselben Tiles lesen den Puffer aus L1 → kein
# redundanter Dequant im heißen FMA-Loop.
#
# workers=0: alle logischen Kerne (Standard)
# workers=1: single-thread (seriell, für Profiling)
# ---------------------------------------------------------------------------
fn matmul_q4_bpack(C: Matrix, A: Matrix, Bq: Q4Matrix, workers: Int = 0):
    var N_log = Bq.packed_cols * 2
    var pcols = Bq.packed_cols
    var Acols = A.cols
    var sv    = SIMD[DT, HALF_W](Bq.scale)
    var num_m_blocks = (C.rows + BM - 1) // BM

    @parameter
    fn process_m_block(mb: Int):
        var m0   = mb * BM
        var m1   = min(m0 + BM, C.rows)
        var bptr = Bq.data()
        var aptr = A.data()

        # Thread-lokaler B-Puffer: BK × NR float32 = 8 KB → L1d
        var b_buf = List[Scalar[DT]]()
        b_buf.resize(BK * NR, 0)
        var bp = rebind[PtrT](b_buf.unsafe_ptr())

        for kt in range(0, Acols, BK):
            var k1 = min(kt + BK, Acols)
            var bk = k1 - kt

            var nt = 0
            while nt < N_log:
                var byte0 = nt // 2
                var byte1 = byte0 + HALF_W

                # Pre-Dequant: bk × 2 dequant_vec → 8 KB fp32, amortisiert über BM/MR Tiles
                for k_local in range(bk):
                    var row   = (kt + k_local) * pcols
                    var b_off = k_local * NR
                    bp.store[width=SIMD_W](b_off,          dequant_vec(bptr, row + byte0, sv))
                    bp.store[width=SIMD_W](b_off + SIMD_W, dequant_vec(bptr, row + byte1, sv))

                # Reines FMA-Kernel (8 Akkumulator-Register, B sequenziell aus L1)
                var m = m0
                while m < m1:
                    var acc00 = SIMD[DT, SIMD_W](0); var acc01 = SIMD[DT, SIMD_W](0)
                    var acc10 = SIMD[DT, SIMD_W](0); var acc11 = SIMD[DT, SIMD_W](0)
                    var acc20 = SIMD[DT, SIMD_W](0); var acc21 = SIMD[DT, SIMD_W](0)
                    var acc30 = SIMD[DT, SIMD_W](0); var acc31 = SIMD[DT, SIMD_W](0)

                    for k_local in range(bk):
                        var b_off = k_local * NR
                        var B0 = bp.load[width=SIMD_W](b_off)
                        var B1 = bp.load[width=SIMD_W](b_off + SIMD_W)
                        var base = m * Acols + kt + k_local
                        var a0 = aptr.load(base)
                        var a1 = aptr.load(base + Acols)
                        var a2 = aptr.load(base + 2 * Acols)
                        var a3 = aptr.load(base + 3 * Acols)
                        acc00 = acc00 + a0 * B0;  acc01 = acc01 + a0 * B1
                        acc10 = acc10 + a1 * B0;  acc11 = acc11 + a1 * B1
                        acc20 = acc20 + a2 * B0;  acc21 = acc21 + a2 * B1
                        acc30 = acc30 + a3 * B0;  acc31 = acc31 + a3 * B1

                    var n1 = nt + SIMD_W
                    C.store[SIMD_W](m,   nt, C.load[SIMD_W](m,   nt) + acc00)
                    C.store[SIMD_W](m,   n1, C.load[SIMD_W](m,   n1) + acc01)
                    C.store[SIMD_W](m+1, nt, C.load[SIMD_W](m+1, nt) + acc10)
                    C.store[SIMD_W](m+1, n1, C.load[SIMD_W](m+1, n1) + acc11)
                    C.store[SIMD_W](m+2, nt, C.load[SIMD_W](m+2, nt) + acc20)
                    C.store[SIMD_W](m+2, n1, C.load[SIMD_W](m+2, n1) + acc21)
                    C.store[SIMD_W](m+3, nt, C.load[SIMD_W](m+3, nt) + acc30)
                    C.store[SIMD_W](m+3, n1, C.load[SIMD_W](m+3, n1) + acc31)
                    m += MR

                nt += NR

    var w = workers if workers > 0 else num_logical_cores()
    parallelize[process_m_block](num_m_blocks, w)


# ---------------------------------------------------------------------------
# matmul_q4_bpack_raw – gleicher Kernel, nimmt rohe Pointer statt Q4Matrix.
# Für Streaming-Kontexte, wo zwei Buffer abwechselnd genutzt werden und
# Q4Matrix-Ownership zwischen parallelen Tasks problematisch wäre.
# ---------------------------------------------------------------------------
fn matmul_q4_bpack_raw(
    C: Matrix,
    A: Matrix,
    bq_ptr:      U8Ptr,   # gepackte uint8 Gewichte
    packed_cols: Int,      # = N // 2
    scale:       Float32,
    workers:     Int = 0,
):
    var N_log = packed_cols * 2
    var Acols = A.cols
    var sv    = SIMD[DT, HALF_W](scale)
    var num_m_blocks = (C.rows + BM - 1) // BM
    var bptr  = bq_ptr

    @parameter
    fn process_m_block(mb: Int):
        var m0 = mb * BM
        var m1 = min(m0 + BM, C.rows)
        var aptr = A.data()

        var b_buf = List[Scalar[DT]]()
        b_buf.resize(BK * NR, 0)
        var bp = rebind[PtrT](b_buf.unsafe_ptr())

        for kt in range(0, Acols, BK):
            var k1 = min(kt + BK, Acols)
            var bk = k1 - kt
            var nt = 0
            while nt < N_log:
                var byte0 = nt // 2
                var byte1 = byte0 + HALF_W
                for k_local in range(bk):
                    var row   = (kt + k_local) * packed_cols
                    var b_off = k_local * NR
                    bp.store[width=SIMD_W](b_off,          dequant_vec(bptr, row + byte0, sv))
                    bp.store[width=SIMD_W](b_off + SIMD_W, dequant_vec(bptr, row + byte1, sv))
                var m = m0
                while m < m1:
                    var acc00 = SIMD[DT, SIMD_W](0); var acc01 = SIMD[DT, SIMD_W](0)
                    var acc10 = SIMD[DT, SIMD_W](0); var acc11 = SIMD[DT, SIMD_W](0)
                    var acc20 = SIMD[DT, SIMD_W](0); var acc21 = SIMD[DT, SIMD_W](0)
                    var acc30 = SIMD[DT, SIMD_W](0); var acc31 = SIMD[DT, SIMD_W](0)
                    for k_local in range(bk):
                        var b_off = k_local * NR
                        var B0 = bp.load[width=SIMD_W](b_off)
                        var B1 = bp.load[width=SIMD_W](b_off + SIMD_W)
                        var base = m * Acols + kt + k_local
                        var a0 = aptr.load(base);              var a1 = aptr.load(base + Acols)
                        var a2 = aptr.load(base + 2 * Acols);  var a3 = aptr.load(base + 3 * Acols)
                        acc00 = acc00 + a0 * B0;  acc01 = acc01 + a0 * B1
                        acc10 = acc10 + a1 * B0;  acc11 = acc11 + a1 * B1
                        acc20 = acc20 + a2 * B0;  acc21 = acc21 + a2 * B1
                        acc30 = acc30 + a3 * B0;  acc31 = acc31 + a3 * B1
                    var n1 = nt + SIMD_W
                    C.store[SIMD_W](m,   nt, C.load[SIMD_W](m,   nt) + acc00)
                    C.store[SIMD_W](m,   n1, C.load[SIMD_W](m,   n1) + acc01)
                    C.store[SIMD_W](m+1, nt, C.load[SIMD_W](m+1, nt) + acc10)
                    C.store[SIMD_W](m+1, n1, C.load[SIMD_W](m+1, n1) + acc11)
                    C.store[SIMD_W](m+2, nt, C.load[SIMD_W](m+2, nt) + acc20)
                    C.store[SIMD_W](m+2, n1, C.load[SIMD_W](m+2, n1) + acc21)
                    C.store[SIMD_W](m+3, nt, C.load[SIMD_W](m+3, nt) + acc30)
                    C.store[SIMD_W](m+3, n1, C.load[SIMD_W](m+3, n1) + acc31)
                    m += MR
                nt += NR

    var w = workers if workers > 0 else num_logical_cores()
    parallelize[process_m_block](num_m_blocks, w)


# ---------------------------------------------------------------------------
# matmul_q4_bpack_l2 – L2-gecachter Kernel mit A-Panel-Packing
#
# Skalierungs-invariant: ~100+ GFLOPS für beliebiges N (1024, 4096, 8192).
#
# Zusätzliche Tiling-Ebene: M in MC=128-Blöcke.
# Für jedes (mc, kc)-Tile → A-Panel in MR-block-major Layout umkopieren.
#   panel[(m_l/MR)*bk*MR + k*MR + (m_l%MR)] = A[m0+m_l, kt+k]
#   Mikro-Kernel-A-Zugriffe: vollsequenziell aus L2 (kein Stride).
#
# Cache-Footprint pro Thread:
#   A-Panel: 128×128×4 = 64 KB  → L2  (256 KB, 2 HT-Threads teilen = 128 KB/Thread)
#   B-Puffer: 128×16×4 = 8 KB   → L1d
# ---------------------------------------------------------------------------

fn matmul_q4_bpack_l2(C: Matrix, A: Matrix, Bq: Q4Matrix, workers: Int = 0):
    var M     = C.rows
    var K     = A.cols
    var N_log = Bq.packed_cols * 2
    var pcols = Bq.packed_cols
    var sv    = SIMD[DT, HALF_W](Bq.scale)
    var n_mc  = (M + MC - 1) // MC

    @parameter
    fn process_mc(mc_idx: Int):
        var m0    = mc_idx * MC
        var m1    = min(m0 + MC, M)
        var mc    = m1 - m0
        var aptr  = A.data()
        var bptr  = Bq.data()
        var Acols = A.cols

        # A-Panel: MC×BK float32, MR-block-major (64 KB → L2)
        var a_panel = List[Scalar[DT]]()
        a_panel.resize(MC * BK, 0)
        var ap = rebind[PtrT](a_panel.unsafe_ptr())

        # B-Puffer: BK×NR float32, pre-dequant (8 KB → L1d)
        var b_buf = List[Scalar[DT]]()
        b_buf.resize(BK * NR, 0)
        var bp = rebind[PtrT](b_buf.unsafe_ptr())

        for kt in range(0, K, BK):
            var k1 = min(kt + BK, K)
            var bk = k1 - kt

            # ── A-Panel-Packing: stride-Acols → sequenziell ───────────────
            for m_l in range(mc):
                var mrb     = m_l // MR
                var mro     = m_l % MR
                var src_row = (m0 + m_l) * Acols + kt
                for kl in range(bk):
                    ap.store(mrb * bk * MR + kl * MR + mro,
                             aptr.load(src_row + kl))

            # ── N-Schleife ─────────────────────────────────────────────────
            var nt = 0
            while nt < N_log:
                var byte0 = nt // 2
                var byte1 = byte0 + HALF_W

                # B-Tile pre-dequant in L1-Puffer
                for kl in range(bk):
                    var row  = (kt + kl) * pcols
                    var boff = kl * NR
                    bp.store[width=SIMD_W](boff,        dequant_vec(bptr, row + byte0, sv))
                    bp.store[width=SIMD_W](boff+SIMD_W, dequant_vec(bptr, row + byte1, sv))

                # Mikro-Kernel: A aus L2-Panel, B aus L1-Puffer
                var m   = m0
                var mrb = 0
                while m < m1:
                    var acc00 = SIMD[DT, SIMD_W](0); var acc01 = SIMD[DT, SIMD_W](0)
                    var acc10 = SIMD[DT, SIMD_W](0); var acc11 = SIMD[DT, SIMD_W](0)
                    var acc20 = SIMD[DT, SIMD_W](0); var acc21 = SIMD[DT, SIMD_W](0)
                    var acc30 = SIMD[DT, SIMD_W](0); var acc31 = SIMD[DT, SIMD_W](0)

                    for kl in range(bk):
                        var boff = kl * NR
                        var B0   = bp.load[width=SIMD_W](boff)
                        var B1   = bp.load[width=SIMD_W](boff + SIMD_W)
                        var poff = mrb * bk * MR + kl * MR
                        var a0 = ap.load(poff + 0)
                        var a1 = ap.load(poff + 1)
                        var a2 = ap.load(poff + 2)
                        var a3 = ap.load(poff + 3)
                        acc00 = acc00 + a0 * B0;  acc01 = acc01 + a0 * B1
                        acc10 = acc10 + a1 * B0;  acc11 = acc11 + a1 * B1
                        acc20 = acc20 + a2 * B0;  acc21 = acc21 + a2 * B1
                        acc30 = acc30 + a3 * B0;  acc31 = acc31 + a3 * B1

                    var n1 = nt + SIMD_W
                    C.store[SIMD_W](m,   nt, C.load[SIMD_W](m,   nt) + acc00)
                    C.store[SIMD_W](m,   n1, C.load[SIMD_W](m,   n1) + acc01)
                    C.store[SIMD_W](m+1, nt, C.load[SIMD_W](m+1, nt) + acc10)
                    C.store[SIMD_W](m+1, n1, C.load[SIMD_W](m+1, n1) + acc11)
                    C.store[SIMD_W](m+2, nt, C.load[SIMD_W](m+2, nt) + acc20)
                    C.store[SIMD_W](m+2, n1, C.load[SIMD_W](m+2, n1) + acc21)
                    C.store[SIMD_W](m+3, nt, C.load[SIMD_W](m+3, nt) + acc30)
                    C.store[SIMD_W](m+3, n1, C.load[SIMD_W](m+3, n1) + acc31)
                    m += MR
                    mrb += 1

                nt += NR

    var w = workers if workers > 0 else num_logical_cores()
    parallelize[process_mc](n_mc, w)


# ---------------------------------------------------------------------------
# matmul_q4_prepacked – kein Stride-Zugriff auf B
#
# Erwartet B im Pre-Tiled Layout (erzeugt von create_fake_model.py --pre-packed):
#   tile_base = (kt_idx * n_nt + nt_idx) * BK * (NR//2)
#   kl_bytes  = tile_base + k_local * (NR//2)
# Alle Lade-Ops sind sequenziell → Hardware-Prefetcher maximal effizient.
#
# KEIN on-the-fly Stride-Packing mehr (kein `k * pcols + byte_off`).
# ---------------------------------------------------------------------------

alias TILE_BYTES = NR // 2  # 8 Bytes pro (k_local, nt_tile) → 2 × HALF_W Reads


fn matmul_q4_prepacked(
    C:       Matrix,
    A:       Matrix,
    bq_ptr:  U8Ptr,    # pre-tiled uint8 Gewichte
    scale:   Float32,
    K:       Int,      # Gewichts-Dimensionen
    N_out:   Int,
    workers: Int = 0,
):
    var M      = C.rows
    var sv     = SIMD[DT, HALF_W](scale)
    var n_kt   = K    // BK
    var n_nt   = N_out // NR
    var t_size = BK * TILE_BYTES  # 1024 Bytes pro (kt, nt)-Tile
    var num_mb = (M + BM - 1) // BM

    @parameter
    fn process_mb(mb: Int):
        var m0    = mb * BM
        var m1    = min(m0 + BM, M)
        var aptr  = A.data()
        var Acols = A.cols

        var b_buf = List[Scalar[DT]]()
        b_buf.resize(BK * NR, 0)
        var bp = rebind[PtrT](b_buf.unsafe_ptr())

        for kt_idx in range(n_kt):
            var kt = kt_idx * BK

            for nt_idx in range(n_nt):
                var nt        = nt_idx * NR
                var tile_base = (kt_idx * n_nt + nt_idx) * t_size

                # Pre-Dequant: sequenziell aus Tile (kein Stride!)
                for kl in range(BK):
                    var kl_off = tile_base + kl * TILE_BYTES
                    var boff   = kl * NR
                    bp.store[width=SIMD_W](boff,        dequant_vec(bq_ptr, kl_off,         sv))
                    bp.store[width=SIMD_W](boff+SIMD_W, dequant_vec(bq_ptr, kl_off+HALF_W,  sv))

                # MR=4 Mikro-Kernel (FMA aus L1-Puffer, identisch zu matmul_q4_bpack)
                var m = m0
                while m < m1:
                    var acc00 = SIMD[DT, SIMD_W](0); var acc01 = SIMD[DT, SIMD_W](0)
                    var acc10 = SIMD[DT, SIMD_W](0); var acc11 = SIMD[DT, SIMD_W](0)
                    var acc20 = SIMD[DT, SIMD_W](0); var acc21 = SIMD[DT, SIMD_W](0)
                    var acc30 = SIMD[DT, SIMD_W](0); var acc31 = SIMD[DT, SIMD_W](0)

                    for kl in range(BK):
                        var boff = kl * NR
                        var B0   = bp.load[width=SIMD_W](boff)
                        var B1   = bp.load[width=SIMD_W](boff + SIMD_W)
                        var base = m * Acols + kt + kl
                        var a0 = aptr.load(base);             var a1 = aptr.load(base + Acols)
                        var a2 = aptr.load(base + 2 * Acols); var a3 = aptr.load(base + 3 * Acols)
                        acc00 = acc00 + a0 * B0;  acc01 = acc01 + a0 * B1
                        acc10 = acc10 + a1 * B0;  acc11 = acc11 + a1 * B1
                        acc20 = acc20 + a2 * B0;  acc21 = acc21 + a2 * B1
                        acc30 = acc30 + a3 * B0;  acc31 = acc31 + a3 * B1

                    var n1 = nt + SIMD_W
                    C.store[SIMD_W](m,   nt, C.load[SIMD_W](m,   nt) + acc00)
                    C.store[SIMD_W](m,   n1, C.load[SIMD_W](m,   n1) + acc01)
                    C.store[SIMD_W](m+1, nt, C.load[SIMD_W](m+1, nt) + acc10)
                    C.store[SIMD_W](m+1, n1, C.load[SIMD_W](m+1, n1) + acc11)
                    C.store[SIMD_W](m+2, nt, C.load[SIMD_W](m+2, nt) + acc20)
                    C.store[SIMD_W](m+2, n1, C.load[SIMD_W](m+2, n1) + acc21)
                    C.store[SIMD_W](m+3, nt, C.load[SIMD_W](m+3, nt) + acc30)
                    C.store[SIMD_W](m+3, n1, C.load[SIMD_W](m+3, n1) + acc31)
                    m += MR

    var w = workers if workers > 0 else num_logical_cores()
    parallelize[process_mb](num_mb, w)


# ---------------------------------------------------------------------------
# rmsnorm_inplace – SIMD-optimierte Row-wise RMSNorm (in-place)
# Jede Zeile x wird normalisiert: x /= rms(x)
# ---------------------------------------------------------------------------

fn rmsnorm_inplace(x: PtrT, n_rows: Int, n_cols: Int):
    for row in range(n_rows):
        var rp = x + row * n_cols
        # Summe der Quadrate via SIMD-Reduktion
        var sq = SIMD[DT, SIMD_W](0)
        var i  = 0
        while i < n_cols:
            var v = rp.load[width=SIMD_W](i)
            sq += v * v
            i  += SIMD_W
        var rms_inv = Float32(1.0) / sqrt(sq.reduce_add() / Float32(n_cols) + Float32(1e-6))
        var scale_v = SIMD[DT, SIMD_W](rms_inv)
        i = 0
        while i < n_cols:
            rp.store[width=SIMD_W](i, rp.load[width=SIMD_W](i) * scale_v)
            i += SIMD_W


# ===========================================================================
# GEMMA 4 EXTENSION LAYER  –  Per-Layer Embeddings (PLE)
#
# Gemma 4 (April 2026) stabilisiert den Dynamikbereich der Aktivierungen
# durch einen layer-spezifischen Skalar, der auf den Residual-Stream
# angewendet wird. Für 4-bit-Quantisierung erhöht das die nutzbare Präzision,
# da der Quantizer die volle INT4-Range nutzen kann.
#
# Ablauf im Forward-Pass:
#   x_residual = ple_scale_inplace(x, ple_scales[layer])
#   y = rmsnorm(x @ W_Q4)
#   x = x + y  (residual connection)
# ===========================================================================

fn ple_scale_inplace(x: PtrT, n_rows: Int, n_cols: Int, scale: Float32):
    """Gemma 4 Per-Layer Embedding (PLE) Skalierung.
    Multipliziert alle Aktivierungen mit dem layer-spezifischen Skalar.
    SIMD-vektorisiert, in-place. O(n_rows × n_cols) Operationen.
    TODO: Erweitern auf per-head Skalierung (Gemma 4 Multi-Head PLE)."""
    var sv    = SIMD[DT, SIMD_W](scale)
    var total = n_rows * n_cols
    var i     = 0
    while i < total:
        x.store[width=SIMD_W](i, x.load[width=SIMD_W](i) * sv)
        i += SIMD_W


fn swiglu_inplace(gate: PtrT, up: PtrT, n: Int):
    """SwiGLU: up[i] = up[i] * silu(gate[i]),  silu(x) = x * sigmoid(x)."""
    var i = 0
    while i + SIMD_W <= n:
        var g   = gate.load[width=SIMD_W](i)
        var u   = up.load[width=SIMD_W](i)
        var sig = SIMD[DT, SIMD_W](1.0) / (SIMD[DT, SIMD_W](1.0) + exp(-g))
        up.store[width=SIMD_W](i, u * g * sig)
        i += SIMD_W
    while i < n:
        var g = gate.load(i)
        var u = up.load(i)
        up.store(i, u * g / (Float32(1.0) + exp(-g)))
        i += 1


# ===========================================================================
# TASK 1 – RoPE (Rotary Positional Embeddings)
#
# Gemma 4 wendet RoPE auf jeden Query- und Key-Kopf an, bevor das
# Attention-Dot-Product berechnet wird. Das codiert relative Positionen
# direkt in die Vektoren, ohne separate Positional-Embedding-Tabelle.
#
# Rotation der Paare (x[i], x[i+half]) mit Frequenz θ_i:
#   x_rot[i]      = x[i] * cos(θ_i) – x[i+half] * sin(θ_i)
#   x_rot[i+half] = x[i] * sin(θ_i) + x[i+half] * cos(θ_i)
#
# θ_i = pos / base^(2i/head_dim)
#      = pos * exp(–2i/head_dim * ln(base))
#
# Optimierung: cos/sin-Tabelle einmalig für die aktuelle Position berechnen
# (O(head_dim/2) skalare Aufrufe), dann SIMD-Rotation über alle n_heads.
# ===========================================================================

fn apply_rope_inplace(
    ptr:      PtrT,
    n_heads:  Int,
    head_dim: Int,
    pos:      Int,
    base:     Float32 = 10000.0,
):
    """RoPE in-place. ptr zeigt auf flat buffer (n_heads × head_dim).
    Paare: (x[i], x[i+half]) für i in 0..half."""
    var half    = head_dim // 2
    var ln_base = log(base)   # ln(10000) ≈ 9.2103

    var cos_buf = List[Scalar[DT]]()
    var sin_buf = List[Scalar[DT]]()
    cos_buf.resize(half, 0)
    sin_buf.resize(half, 0)
    var cp = rebind[PtrT](cos_buf.unsafe_ptr())
    var sp = rebind[PtrT](sin_buf.unsafe_ptr())

    var pos_f = Float32(pos)
    for i in range(half):
        var inv_freq = exp(-Float32(2 * i) / Float32(head_dim) * ln_base)
        var theta    = pos_f * inv_freq
        cp.store(i, cos(theta))
        sp.store(i, sin(theta))

    for h in range(n_heads):
        var hp = ptr + h * head_dim
        var i  = 0
        while i + SIMD_W <= half:
            var xi  = hp.load[width=SIMD_W](i)
            var xih = hp.load[width=SIMD_W](i + half)
            var cv  = cp.load[width=SIMD_W](i)
            var sv  = sp.load[width=SIMD_W](i)
            hp.store[width=SIMD_W](i,        xi * cv - xih * sv)
            hp.store[width=SIMD_W](i + half, xi * sv + xih * cv)
            i += SIMD_W
        while i < half:
            var xi  = hp.load(i);  var xih = hp.load(i + half)
            var cv  = cp.load(i);  var sv  = sp.load(i)
            hp.store(i,        xi * cv - xih * sv)
            hp.store(i + half, xi * sv + xih * cv)
            i += 1


fn silu_inplace(x: PtrT, n: Int):
    """SiLU in-place: x[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(–x[i]))."""
    var i = 0
    while i + SIMD_W <= n:
        var v   = x.load[width=SIMD_W](i)
        var sig = SIMD[DT, SIMD_W](1.0) / (SIMD[DT, SIMD_W](1.0) + exp(-v))
        x.store[width=SIMD_W](i, v * sig)
        i += SIMD_W
    while i < n:
        var v = x.load(i)
        x.store(i, v / (Float32(1.0) + exp(-v)))
        i += 1
