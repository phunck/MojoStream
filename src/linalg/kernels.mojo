# src/linalg/kernels.mojo
# B-Pack Q4 Matmul Kernel – extrahiert aus dem Benchmark-Stand.
# Einziger Kernel der exportiert wird: matmul_q4_bpack (128 GFLOPS auf AVX2).
from std.algorithm.functional import parallelize
from std.memory import UnsafePointer, memset_zero
from std.sys.info import simd_width_of, num_logical_cores
from std.random import rand

# ---------------------------------------------------------------------------
# Compile-Zeit Konstanten
# ---------------------------------------------------------------------------
alias DT     = DType.float32
alias SIMD_W = simd_width_of[DT]()  # 8 (AVX2) oder 16 (AVX-512)
alias HALF_W = SIMD_W // 2          # Bytes pro SIMD-Vektor in gepacktem uint8

alias MR       = 4    # Mikro-Kernel Zeilen
alias NR_SIMD  = 2    # SIMD-Vektoren pro Spalten-Tile
alias NR       = NR_SIMD * SIMD_W  # skalare Spalten pro Tile (= 16)
alias BM       = 64   # M-Block für parallelize
alias BK       = 128  # K-Tile (bestimmt B-Puffer-Größe und Cache-Affinität)

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
