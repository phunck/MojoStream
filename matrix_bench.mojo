# matrix_bench.mojo – CPU-optimierte 1024x1024 Matmul (float32) – Mojo 0.26
from std.algorithm.functional import parallelize
from std.memory import UnsafePointer, memset_zero
from std.sys.info import simd_width_of, num_logical_cores, num_physical_cores
from std.time import perf_counter_ns
from std.random import rand

alias DT = DType.float32
alias N = 1024
alias SIMD_W = simd_width_of[DT]()

alias BM = 64
alias BN = 128
alias BK = 128

alias PtrT = UnsafePointer[Scalar[DT], MutAnyOrigin]


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


# ----------------------------------------------------------------------
# Q4Matrix: 4-bit gepackte Gewichte, eine globale Float32-Skala.
# Layout: storage[k*packed_cols + j] enthaelt zwei Nibbles fuer logische
# Spalten (2j) und (2j+1). Dequant: w = (nibble - 8) * scale.
# ----------------------------------------------------------------------

alias U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]


struct Q4Matrix(Movable):
    var storage: List[UInt8]
    var rows: Int
    var packed_cols: Int   # = logical_cols // 2
    var scale: Float32

    fn __init__(out self, rows: Int, logical_cols: Int, scale: Float32):
        self.rows = rows
        self.packed_cols = logical_cols // 2
        self.scale = scale
        self.storage = List[UInt8]()
        self.storage.resize(rows * self.packed_cols, 0)

    @always_inline
    fn data(self) -> U8Ptr:
        return rebind[U8Ptr](self.storage.unsafe_ptr())

    fn logical_cols(self) -> Int:
        return self.packed_cols * 2


# ----------------------------------------------------------------------
# Loader: liest A.bin (float32), B_q4.bin (uint8), meta.txt (N, scale, refs)
# ----------------------------------------------------------------------

struct Meta(Copyable, Movable):
    var n: Int
    var scale: Float32
    var c00: Float32
    var clast: Float32

    fn __init__(out self, n: Int, scale: Float32, c00: Float32, clast: Float32):
        self.n = n
        self.scale = scale
        self.c00 = c00
        self.clast = clast


fn load_meta(path: String) raises -> Meta:
    """Liest N, scale, C_ref[0,0], C_ref[N-1,N-1] aus meta.txt."""
    with open(path, "r") as f:
        var content = f.read()
        var lines = content.split("\n")
        return Meta(
            Int(lines[0]),
            Float32(atof(lines[1])),
            Float32(atof(lines[2])),
            Float32(atof(lines[3])),
        )


fn load_a_matrix(path: String, n: Int) raises -> Matrix:
    var m = Matrix(n, n)
    with open(path, "r") as f:
        var data = f.read_bytes()
        var src = data.unsafe_ptr().bitcast[Float32]()
        var dst = m.data()
        for i in range(n * n):
            dst.store(i, src.load(i))
    return m^


fn load_q4_matrix(path: String, rows: Int, logical_cols: Int, scale: Float32) raises -> Q4Matrix:
    var bq = Q4Matrix(rows, logical_cols, scale)
    var packed = rows * (logical_cols // 2)
    with open(path, "r") as f:
        var data = f.read_bytes()
        var src = data.unsafe_ptr()
        var dst = bq.data()
        for i in range(packed):
            dst.store(i, src.load(i))
    return bq^


# ----------------------------------------------------------------------
# Fused-Dequant Tiled Matmul: 4-bit Gewichte werden in der innersten
# SIMD-Schleife per >> 4 und & 0x0F entpackt, sofort in float32 konvertiert,
# mit der Skala multipliziert und in C akkumuliert.
# Die fp32-Form der Gewichte existiert nur in SIMD-Registern, nie im RAM.
# ----------------------------------------------------------------------

alias HALF_W = SIMD_W // 2   # so viele Bytes laden wir pro SIMD-Iteration
alias MR = 4                  # Mikro-Kernel: Zeilen (4×16 Kernel)
alias MR6 = 6                 # erweiterter Mikro-Kernel: Zeilen (6×16 Kernel + A-Packing)
alias NR_SIMD = 2             # SIMD-Vektoren pro Tile-Spalte (beide Kernels)


@always_inline
fn dequant_vec(ptr: U8Ptr, byte_off: Int, sv: SIMD[DT, HALF_W]) -> SIMD[DT, SIMD_W]:
    """Laedt HALF_W gepackte Bytes, entpackt Nibbles via >> 4 / & 0x0F,
    wendet Offset -8 und Skala an. Ergebnis: SIMD[float32, SIMD_W] nur im Register."""
    var p  = ptr.load[width=HALF_W](byte_off)
    var lo = (p & SIMD[DType.uint8, HALF_W](0x0F)).cast[DType.int32]()
    var hi = ((p >> SIMD[DType.uint8, HALF_W](4)) & SIMD[DType.uint8, HALF_W](0x0F)).cast[DType.int32]()
    var bias = SIMD[DType.int32, HALF_W](8)
    return rebind[SIMD[DT, SIMD_W]](
        ((lo - bias).cast[DT]() * sv).interleave((hi - bias).cast[DT]() * sv)
    )


fn matmul_q4_fused_tiled(C: Matrix, A: Matrix, Bq: Q4Matrix):
    var N_log = Bq.packed_cols * 2
    var num_m_blocks = (C.rows + BM - 1) // BM
    var scale = Bq.scale
    var pcols = Bq.packed_cols

    @parameter
    fn process_m_block(mb: Int):
        var m0 = mb * BM
        var m1 = min(m0 + BM, C.rows)

        for kt in range(0, A.cols, BK):
            var k1 = min(kt + BK, A.cols)
            for nt in range(0, N_log, BN):
                var n1 = min(nt + BN, N_log)
                for m in range(m0, m1):
                    for k in range(kt, k1):
                        var a = A.data().load(m * A.cols + k)
                        var n = nt
                        while n < n1:
                            # 1) HALF_W gepackte Bytes laden -> SIMD_W Gewichte
                            var packed = Bq.data().load[width=HALF_W](
                                k * pcols + (n // 2)
                            )
                            # 2) Bit-Manipulation in Registern
                            var lo = packed & SIMD[DType.uint8, HALF_W](0x0F)
                            var hi = (packed >> SIMD[DType.uint8, HALF_W](4)) \
                                     & SIMD[DType.uint8, HALF_W](0x0F)
                            # 3) Vorzeichenrichtiger Offset in int32
                            var lo_i = lo.cast[DType.int32]() - SIMD[DType.int32, HALF_W](8)
                            var hi_i = hi.cast[DType.int32]() - SIMD[DType.int32, HALF_W](8)
                            # 4) Float-Konvertierung + globale Skala
                            var lo_f = lo_i.cast[DT]() * SIMD[DT, HALF_W](scale)
                            var hi_f = hi_i.cast[DT]() * SIMD[DT, HALF_W](scale)
                            # 5) Interleave -> [lo0,hi0,lo1,hi1,...] = logische Reihenfolge
                            #    Typ ist SIMD[DT, 2*HALF_W] - explizit als SIMD_W rebinden
                            var weights = rebind[SIMD[DT, SIMD_W]](lo_f.interleave(hi_f))
                            # 6) FMA in C
                            var c = C.load[SIMD_W](m, n)
                            C.store[SIMD_W](m, n, c + a * weights)
                            n += SIMD_W

    parallelize[process_m_block](num_m_blocks, num_m_blocks)


# ----------------------------------------------------------------------
# Register-Blocked Q4 Matmul (4×16 Mikro-Kernel)
#
# Statt C pro k-Schritt zu lesen/schreiben, halten 8 SIMD-Akkumulatoren
# (MR=4 Zeilen × NR_SIMD=2 Vektoren) die Partial-Sums in Registern für
# den gesamten K-Tile. Write-Back nach C erfolgt einmal am Ende des Tiles.
#
# Arith. Intensität des Mikro-Kernels:
#   Lesen : MR × BK floats (A-Spalten) + 2 × BK × HALF_W bytes (B gepackt)
#   FMAs  : MR × NR_SIMD × BK = 4 × 2 × 128 = 1024 pro Micro-Kernel-Aufruf
#   Schreiben: 2×MR SIMD-Stores (= 8, vs. 128×MR×NR_SIMD in der naiven Version)
# ----------------------------------------------------------------------

fn matmul_q4_regblocked(C: Matrix, A: Matrix, Bq: Q4Matrix):
    var N_log  = Bq.packed_cols * 2
    var pcols  = Bq.packed_cols
    var Acols  = A.cols
    var sv     = SIMD[DT, HALF_W](Bq.scale)   # Skala-Broadcast vorab
    var num_m_blocks = (C.rows + BM - 1) // BM

    @parameter
    fn process_m_block(mb: Int):
        var m0   = mb * BM
        var m1   = min(m0 + BM, C.rows)
        var bptr = Bq.data()
        var aptr = A.data()

        for kt in range(0, Acols, BK):
            var k1 = min(kt + BK, Acols)

            # Aeussere N-Schleife in NR_SIMD*SIMD_W=16-Schritten
            var nt = 0
            while nt < N_log:
                var byte0 = nt // 2           # Byte-Offset fuer Cols nt..nt+7
                var byte1 = byte0 + HALF_W    # Byte-Offset fuer Cols nt+8..nt+15

                # Aeussere M-Schleife in MR=4-Schritten
                var m = m0
                while m < m1:

                    # ---- 8 Akkumulator-Register initialisieren ----
                    var acc00 = SIMD[DT, SIMD_W](0)
                    var acc01 = SIMD[DT, SIMD_W](0)
                    var acc10 = SIMD[DT, SIMD_W](0)
                    var acc11 = SIMD[DT, SIMD_W](0)
                    var acc20 = SIMD[DT, SIMD_W](0)
                    var acc21 = SIMD[DT, SIMD_W](0)
                    var acc30 = SIMD[DT, SIMD_W](0)
                    var acc31 = SIMD[DT, SIMD_W](0)

                    # ---- K-Inner-Loop: kein C-Zugriff ----
                    for k in range(kt, k1):
                        var row = k * pcols

                        # Dequant direkt im Register: 2×4 Bytes → 2×8 float32
                        var B0 = dequant_vec(bptr, row + byte0, sv)
                        var B1 = dequant_vec(bptr, row + byte1, sv)

                        # MR=4 A-Skalare (Outer-Product Strategie)
                        var base0 = m * Acols + k
                        var a0 = aptr.load(base0)
                        var a1 = aptr.load(base0 + Acols)
                        var a2 = aptr.load(base0 + 2 * Acols)
                        var a3 = aptr.load(base0 + 3 * Acols)

                        # FMA: jeder A-Skalar aktualisiert NR_SIMD=2 Akkumulatoren
                        acc00 = acc00 + a0 * B0;  acc01 = acc01 + a0 * B1
                        acc10 = acc10 + a1 * B0;  acc11 = acc11 + a1 * B1
                        acc20 = acc20 + a2 * B0;  acc21 = acc21 + a2 * B1
                        acc30 = acc30 + a3 * B0;  acc31 = acc31 + a3 * B1

                    # ---- Write-back: addiere Partial-Sum zu C (8 Stores) ----
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
                nt += NR_SIMD * SIMD_W

    parallelize[process_m_block](num_m_blocks, num_m_blocks)


# ----------------------------------------------------------------------
# 6×16 Register-Blocked Q4 Matmul mit A-Panel Packing
#
# A-Packing: Jeder MR6×BK A-Block wird in einen kontinuierlichen Puffer
# (column-major: panel[k*MR6+i]) umkopiert. Damit werden die 4-KB-Stride-
# Zugriffe auf A durch sequentielle L1-Zugriffe ersetzt.
#
# Mikro-Kernel: 12 Akkumulator-Register (MR6=6 × NR_SIMD=2), kein C-Zugriff
# im K-Loop. B-Dequant (>> 4, & 0x0F) direkt vor jeder FMA-Operation.
# @parameter for in der Pack-Schleife für compile-time Loop-Unrolling.
#
# Tail-Handling: verbleibende < MR6 Zeilen pro BM-Block (~6% der Arbeit)
# werden mit einem single-row regblocked-Loop abgearbeitet.
# ----------------------------------------------------------------------

fn matmul_q4_packed_6x16(C: Matrix, A: Matrix, Bq: Q4Matrix):
    var N_log  = Bq.packed_cols * 2
    var pcols  = Bq.packed_cols
    var Acols  = A.cols
    var sv     = SIMD[DT, HALF_W](Bq.scale)
    var num_m_blocks = (C.rows + BM - 1) // BM

    @parameter
    fn process_m_block(mb: Int):
        var m0   = mb * BM
        var m1   = min(m0 + BM, C.rows)
        var bptr = Bq.data()
        var aptr = A.data()

        # A-Panel: MR6 × BK Floats, einmal pro Thread allokiert, pro (m,kt) neu befüllt.
        # Layout column-major: panel[k_local * MR6 + i] = A[m+i, kt+k_local]
        # Groesse 6×128 = 768 Floats = 3 KB → passt komplett in L1d.
        var a_panel = List[Scalar[DT]]()
        a_panel.resize(MR6 * BK, 0)
        var ap = rebind[PtrT](a_panel.unsafe_ptr())

        for kt in range(0, Acols, BK):
            var k1  = min(kt + BK, Acols)
            var bk  = k1 - kt          # effektive K-Tile-Breite

            # ── Volle 6-Zeilen Mikro-Kernel-Blocks ──
            var m = m0
            while m + MR6 <= m1:

                # A-Packing: MR6 Zeilen × bk Spalten in kontinuierlichen Puffer.
                # @parameter for i in range(MR6) wird zur Compile-Zeit unrollt.
                for k_local in range(bk):
                    var k = kt + k_local
                    var poff = k_local * MR6
                    @parameter
                    for i in range(MR6):
                        ap.store(poff + i, aptr.load((m + i) * Acols + k))

                # ── N-Dimension in 16-Spalten-Tiles ──
                var nt = 0
                while nt < N_log:
                    var byte0 = nt // 2
                    var byte1 = byte0 + HALF_W

                    # 12 Akkumulator-Register (MR6 × NR_SIMD)
                    var acc00 = SIMD[DT, SIMD_W](0); var acc01 = SIMD[DT, SIMD_W](0)
                    var acc10 = SIMD[DT, SIMD_W](0); var acc11 = SIMD[DT, SIMD_W](0)
                    var acc20 = SIMD[DT, SIMD_W](0); var acc21 = SIMD[DT, SIMD_W](0)
                    var acc30 = SIMD[DT, SIMD_W](0); var acc31 = SIMD[DT, SIMD_W](0)
                    var acc40 = SIMD[DT, SIMD_W](0); var acc41 = SIMD[DT, SIMD_W](0)
                    var acc50 = SIMD[DT, SIMD_W](0); var acc51 = SIMD[DT, SIMD_W](0)

                    # K-Inner-Loop: B-Dequant direkt vor FMA, A aus gepacktem Panel
                    for k_local in range(bk):
                        var row  = (kt + k_local) * pcols
                        # Dequant: >> 4 und & 0x0F liefern fp32 SIMD direkt in Register
                        var B0 = dequant_vec(bptr, row + byte0, sv)
                        var B1 = dequant_vec(bptr, row + byte1, sv)
                        # Sequentieller Zugriff auf A-Panel (kein Stride mehr)
                        var poff = k_local * MR6
                        var a0 = ap.load(poff + 0)
                        var a1 = ap.load(poff + 1)
                        var a2 = ap.load(poff + 2)
                        var a3 = ap.load(poff + 3)
                        var a4 = ap.load(poff + 4)
                        var a5 = ap.load(poff + 5)
                        # 12 FMAs (Outer Product: 6 A-Skalare × 2 B-Vektoren)
                        acc00 = acc00 + a0 * B0;  acc01 = acc01 + a0 * B1
                        acc10 = acc10 + a1 * B0;  acc11 = acc11 + a1 * B1
                        acc20 = acc20 + a2 * B0;  acc21 = acc21 + a2 * B1
                        acc30 = acc30 + a3 * B0;  acc31 = acc31 + a3 * B1
                        acc40 = acc40 + a4 * B0;  acc41 = acc41 + a4 * B1
                        acc50 = acc50 + a5 * B0;  acc51 = acc51 + a5 * B1

                    # Write-back: 12 Stores (vs. bk×12 in naiver Version)
                    var n1 = nt + SIMD_W
                    C.store[SIMD_W](m,   nt, C.load[SIMD_W](m,   nt) + acc00)
                    C.store[SIMD_W](m,   n1, C.load[SIMD_W](m,   n1) + acc01)
                    C.store[SIMD_W](m+1, nt, C.load[SIMD_W](m+1, nt) + acc10)
                    C.store[SIMD_W](m+1, n1, C.load[SIMD_W](m+1, n1) + acc11)
                    C.store[SIMD_W](m+2, nt, C.load[SIMD_W](m+2, nt) + acc20)
                    C.store[SIMD_W](m+2, n1, C.load[SIMD_W](m+2, n1) + acc21)
                    C.store[SIMD_W](m+3, nt, C.load[SIMD_W](m+3, nt) + acc30)
                    C.store[SIMD_W](m+3, n1, C.load[SIMD_W](m+3, n1) + acc31)
                    C.store[SIMD_W](m+4, nt, C.load[SIMD_W](m+4, nt) + acc40)
                    C.store[SIMD_W](m+4, n1, C.load[SIMD_W](m+4, n1) + acc41)
                    C.store[SIMD_W](m+5, nt, C.load[SIMD_W](m+5, nt) + acc50)
                    C.store[SIMD_W](m+5, n1, C.load[SIMD_W](m+5, n1) + acc51)

                    nt += NR_SIMD * SIMD_W
                m += MR6

            # ── Tail: verbleibende 1-5 Zeilen – single-row regblocked ──
            # BM=64 mod MR6=6 = 4 Zeilen pro Block → ~6% der Gesamtarbeit
            while m < m1:
                var nt = 0
                while nt < N_log:
                    var byte0 = nt // 2
                    var acc0 = SIMD[DT, SIMD_W](0)
                    var acc1 = SIMD[DT, SIMD_W](0)
                    for k in range(kt, k1):
                        var a  = aptr.load(m * Acols + k)
                        acc0 = acc0 + a * dequant_vec(bptr, k * pcols + byte0,         sv)
                        acc1 = acc1 + a * dequant_vec(bptr, k * pcols + byte0 + HALF_W, sv)
                    var n1 = nt + SIMD_W
                    C.store[SIMD_W](m, nt, C.load[SIMD_W](m, nt) + acc0)
                    C.store[SIMD_W](m, n1, C.load[SIMD_W](m, n1) + acc1)
                    nt += NR_SIMD * SIMD_W
                m += 1

    parallelize[process_m_block](num_m_blocks, num_m_blocks)


# ----------------------------------------------------------------------
# Q4 B-Panel-Pack + MR=4 Register-Blocked (schnellste Variante)
#
# Strategie: für jedes (kt, nt)-Tile wird B[kt:kt+BK, nt:nt+NR] einmal
# komplett dequantisiert und als fp32 in einen lokalen 8-KB-Puffer geschrieben
# (BK=128 × NR=16 × 4 Bytes = 8 KB → passt in L1d).
#
# Alle BM/MR = 16 m-Micro-Blöcke desselben (kt, nt)-Tiles nutzen denselben
# Puffer → Dequant-Overhead sinkt von 1× pro k-Schritt auf 1/16 (amortisiert).
#
# Inner Loop: pure FMA (kein Bit-Shifting, keine Casts), B sequenziell aus L1.
# Arithmetische Intensität: 8 FMAs pro k_local vs. vorher 8 FMAs + 2 dequant_vec.
# ----------------------------------------------------------------------

alias NR = NR_SIMD * SIMD_W   # = 16 Spalten pro Mikro-Kernel Tile


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

        # Lokaler B-Panel-Buffer: BK × NR float32 = 8 KB — einmal allokiert pro Thread.
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

                # ── Pre-Dequant: BK × 16 fp32 in lokalen L1-Puffer schreiben ──
                # Kosten: bk × 2 dequant_vec — amortisiert über BM/MR = 16 m-Tiles
                for k_local in range(bk):
                    var row = (kt + k_local) * pcols
                    var b_off = k_local * NR
                    bp.store[width=SIMD_W](b_off,        dequant_vec(bptr, row + byte0, sv))
                    bp.store[width=SIMD_W](b_off + SIMD_W, dequant_vec(bptr, row + byte1, sv))

                # ── Reines FMA-Kernel über B-Buffer (kein Dequant im Hot-Path) ──
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

    # workers=0 → alle logischen Kerne; sonst explizit gesetzt
    var w = workers if workers > 0 else num_logical_cores()
    parallelize[process_m_block](num_m_blocks, w)


fn matmul_simd_parallel(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            var a = A.data().load(m * A.cols + k)
            # Manuelle SIMD-Schleife: B.cols ist Vielfaches von SIMD_W (1024 / 8 oder 16)
            var n = 0
            while n < B.cols:
                var b = B.load[SIMD_W](k, n)
                var c = C.load[SIMD_W](m, n)
                C.store[SIMD_W](m, n, c + a * b)
                n += SIMD_W

    parallelize[calc_row](C.rows, C.rows)


fn matmul_tiled(C: Matrix, A: Matrix, B: Matrix):
    var num_m_blocks = (C.rows + BM - 1) // BM

    @parameter
    fn process_m_block(mb: Int):
        var m0 = mb * BM
        var m1 = min(m0 + BM, C.rows)

        for kt in range(0, A.cols, BK):
            var k1 = min(kt + BK, A.cols)
            for nt in range(0, B.cols, BN):
                var n1 = min(nt + BN, B.cols)

                for m in range(m0, m1):
                    for k in range(kt, k1):
                        var a = A.data().load(m * A.cols + k)
                        var n = nt
                        while n < n1:
                            var b = B.load[SIMD_W](k, n)
                            var c = C.load[SIMD_W](m, n)
                            C.store[SIMD_W](m, n, c + a * b)
                            n += SIMD_W

    parallelize[process_m_block](num_m_blocks, num_m_blocks)


fn main() raises:
    print("=== Mojo-Benchmark:", N, "x", N, "float32 matmul ===")
    print("SIMD width =", SIMD_W, " BM/BN/BK =", BM, BN, BK)

    var A = Matrix(N, N)
    var B = Matrix(N, N)
    var C = Matrix(N, N)
    A.fill_random()
    B.fill_random()

    var flops = 2.0 * Float64(N) * Float64(N) * Float64(N)

    # ---------- Variante 1: SIMD + parallelize ----------
    matmul_simd_parallel(C, A, B)  # warm-up
    var best1: UInt = UInt.MAX
    for _ in range(3):
        C.zero()
        var t0 = perf_counter_ns()
        matmul_simd_parallel(C, A, B)
        var dt = perf_counter_ns() - t0
        if dt < best1:
            best1 = dt
    var s1 = Float64(Int(best1)) / 1.0e9
    var g1 = flops / s1 / 1.0e9
    print("Mojo SIMD+par   :", s1 * 1000.0, "ms  ", g1, "GFLOPS")

    # ---------- Variante 2: Tiled + SIMD ----------
    C.zero()
    matmul_tiled(C, A, B)  # warm-up
    var best2: UInt = UInt.MAX
    for _ in range(3):
        C.zero()
        var t0 = perf_counter_ns()
        matmul_tiled(C, A, B)
        var dt = perf_counter_ns() - t0
        if dt < best2:
            best2 = dt
    var s2 = Float64(Int(best2)) / 1.0e9
    var g2 = flops / s2 / 1.0e9
    print("Mojo Tiled+SIMD :", s2 * 1000.0, "ms  ", g2, "GFLOPS")

    print("RESULT_MOJO_MS=",       s1 * 1000.0)
    print("RESULT_MOJO_GFLOPS=",   g1)
    print("RESULT_MOJO_TILED_MS=", s2 * 1000.0)
    print("RESULT_MOJO_TILED_GFLOPS=", g2)

    # ---------- Variante 3: Q4 Fused Dequant ----------
    print()
    print("=== Mojo Q4 Fused Dequant ===")
    var meta_path = "meta.txt"
    var a_path    = "A.bin"
    var bq_path   = "B_q4.bin"
    var meta = load_meta(meta_path)
    var Nq        = meta.n
    var scale_q   = meta.scale
    var ref_c00   = meta.c00
    var ref_clast = meta.clast
    print("Q4 N =", Nq, " scale =", scale_q)

    var Aq = load_a_matrix(a_path, Nq)
    var Bq = load_q4_matrix(bq_path, Nq, Nq, scale_q)
    var Cq = Matrix(Nq, Nq)

    matmul_q4_fused_tiled(Cq, Aq, Bq)  # warm-up + Korrektheitscheck
    var c00_mojo = Cq.data().load(0)
    var clast_mojo = Cq.data().load(Nq * Nq - 1)
    print("Korrektheit (NumPy vs Mojo): C[0,0] np=", ref_c00, " mojo=", c00_mojo,
          "  | C[N-1,N-1] np=", ref_clast, " mojo=", clast_mojo)

    var best_q: UInt = UInt.MAX
    for _ in range(3):
        Cq.zero()
        var t0 = perf_counter_ns()
        matmul_q4_fused_tiled(Cq, Aq, Bq)
        var dt = perf_counter_ns() - t0
        if dt < best_q:
            best_q = dt
    var sq = Float64(Int(best_q)) / 1.0e9
    var gq = (2.0 * Float64(Nq) * Float64(Nq) * Float64(Nq)) / sq / 1.0e9
    print("Mojo Q4 Fused   :", sq * 1000.0, "ms  ", gq, "GFLOPS")
    print("RESULT_Q4_MOJO_MS=",     sq * 1000.0)
    print("RESULT_Q4_MOJO_GFLOPS=", gq)

    # ---------- Variante 4: Q4 Register-Blocked (4×16 Mikro-Kernel) ----------
    print()
    print("=== Mojo Q4 Register-Blocked (MR=", MR, " NR=", NR_SIMD * SIMD_W, ") ===")
    var Crb = Matrix(Nq, Nq)

    # Warm-up + Korrektheitscheck
    matmul_q4_regblocked(Crb, Aq, Bq)
    var rb_c00   = Crb.data().load(0)
    var rb_clast = Crb.data().load(Nq * Nq - 1)
    print("Korrektheit: C[0,0] np=", ref_c00, " regblocked=", rb_c00,
          "  | C[N-1,N-1] np=", ref_clast, " regblocked=", rb_clast)

    var best_rb: UInt = UInt.MAX
    for _ in range(3):
        Crb.zero()
        var t0 = perf_counter_ns()
        matmul_q4_regblocked(Crb, Aq, Bq)
        var dt = perf_counter_ns() - t0
        if dt < best_rb:
            best_rb = dt
    var srb = Float64(Int(best_rb)) / 1.0e9
    var grb = (2.0 * Float64(Nq) * Float64(Nq) * Float64(Nq)) / srb / 1.0e9
    print("Mojo Q4 RegBlocked:", srb * 1000.0, "ms  ", grb, "GFLOPS")
    print("RESULT_Q4_RB_MS=",     srb * 1000.0)
    print("RESULT_Q4_RB_GFLOPS=", grb)

    # ---------- Variante 5: Q4 6×16 mit A-Panel Packing ----------
    print()
    print("=== Mojo Q4 6×16 + A-Packing (MR6=", MR6, " NR=", NR_SIMD * SIMD_W, ") ===")
    var Cpk = Matrix(Nq, Nq)
    matmul_q4_packed_6x16(Cpk, Aq, Bq)  # warm-up
    var pk_c00   = Cpk.data().load(0)
    var pk_clast = Cpk.data().load(Nq * Nq - 1)
    print("Korrektheit: C[0,0] np=", ref_c00, " packed=", pk_c00,
          "  | C[N-1,N-1] np=", ref_clast, " packed=", pk_clast)

    var best_pk: UInt = UInt.MAX
    for _ in range(3):
        Cpk.zero()
        var t0 = perf_counter_ns()
        matmul_q4_packed_6x16(Cpk, Aq, Bq)
        var dt = perf_counter_ns() - t0
        if dt < best_pk:
            best_pk = dt
    var spk = Float64(Int(best_pk)) / 1.0e9
    var gpk = (2.0 * Float64(Nq) * Float64(Nq) * Float64(Nq)) / spk / 1.0e9
    print("Mojo Q4 6x16+Pack :", spk * 1000.0, "ms  ", gpk, "GFLOPS")
    print("RESULT_Q4_PK_MS=",     spk * 1000.0)
    print("RESULT_Q4_PK_GFLOPS=", gpk)

    # ---------- Variante 6: Q4 B-Panel-Pack + MR=4 (Zielstrategie) ----------
    print()
    print("=== Mojo Q4 B-Panel-Pack + RegBlocked ===")
    var Cbp = Matrix(Nq, Nq)
    matmul_q4_bpack(Cbp, Aq, Bq)   # warm-up
    var bp_c00   = Cbp.data().load(0)
    var bp_clast = Cbp.data().load(Nq * Nq - 1)
    print("Korrektheit: C[0,0] np=", ref_c00, " bpack=", bp_c00,
          "  | C[N-1,N-1] np=", ref_clast, " bpack=", bp_clast)

    var best_bp: UInt = UInt.MAX
    for _ in range(3):
        Cbp.zero()
        var t0 = perf_counter_ns()
        matmul_q4_bpack(Cbp, Aq, Bq)
        var dt = perf_counter_ns() - t0
        if dt < best_bp:
            best_bp = dt
    var sbp = Float64(Int(best_bp)) / 1.0e9
    var gbp = (2.0 * Float64(Nq) * Float64(Nq) * Float64(Nq)) / sbp / 1.0e9
    print("Mojo Q4 B-Pack (MT) :", sbp * 1000.0, "ms  ", gbp, "GFLOPS")
    print("RESULT_Q4_BP_MS=",     sbp * 1000.0)
    print("RESULT_Q4_BP_GFLOPS=", gbp)

    # ---------- Variante 7: Scaling-Analyse 1-Thread vs. N-Thread ----------
    var n_logical = num_logical_cores()
    var n_physical = num_physical_cores()
    print()
    print("=== Mojo Scaling-Analyse ===")
    print("Hardware: logical=", n_logical, " physical=", n_physical)

    # 1-Thread (seriell, kein Parallelismus)
    var Cst = Matrix(Nq, Nq)
    matmul_q4_bpack(Cst, Aq, Bq, 1)   # warm-up
    var best_st: UInt = UInt.MAX
    for _ in range(3):
        Cst.zero()
        var t0 = perf_counter_ns()
        matmul_q4_bpack(Cst, Aq, Bq, 1)
        var dt = perf_counter_ns() - t0
        if dt < best_st:
            best_st = dt
    var sst = Float64(Int(best_st)) / 1.0e9
    var gst = (2.0 * Float64(Nq) * Float64(Nq) * Float64(Nq)) / sst / 1.0e9
    print("1-Thread            :", sst * 1000.0, "ms  ", gst, "GFLOPS")

    # N-Thread (alle logischen Kerne)
    var best_mt: UInt = UInt.MAX
    var Cmt = Matrix(Nq, Nq)
    matmul_q4_bpack(Cmt, Aq, Bq, n_logical)   # warm-up
    for _ in range(3):
        Cmt.zero()
        var t0 = perf_counter_ns()
        matmul_q4_bpack(Cmt, Aq, Bq, n_logical)
        var dt = perf_counter_ns() - t0
        if dt < best_mt:
            best_mt = dt
    var smt = Float64(Int(best_mt)) / 1.0e9
    var gmt = (2.0 * Float64(Nq) * Float64(Nq) * Float64(Nq)) / smt / 1.0e9
    var speedup  = sst / smt
    var eff_pct  = speedup / Float64(n_logical) * 100.0
    print(n_logical, "-Thread           :", smt * 1000.0, "ms  ", gmt, "GFLOPS")
    print("Speedup:", speedup, "x  |  Efficiency:", eff_pct, "%")

    print("RESULT_CORES=",          n_logical)
    print("RESULT_Q4_BP_1T_MS=",    sst * 1000.0)
    print("RESULT_Q4_BP_1T_GFLOPS=", gst)
    print("RESULT_Q4_BP_MT_MS=",    smt * 1000.0)
    print("RESULT_Q4_BP_MT_GFLOPS=", gmt)
