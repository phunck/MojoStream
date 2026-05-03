# src/bench/benchmark_q4.mojo
#
# GFLOPS-Vergleich: B-Pack (klassisch) vs. Fused-AVX2 (neuer Kernel)
#
# Metrik: effektive GFLOPS = 2 × M × K × N / Zeit
# (gleiche FLOP-Zählung wie FP32-Matmul, da wir identische Arithmetik ausführen)
#
# Erwartung für memory-bound Workloads (Streaming):
#   Fused-AVX2 ≥ B-Pack, da 8× weniger Lade-Instruktionen + kein L1-Buffer-Write
#
# Erwartung für compute-bound Workloads (alles in Cache):
#   B-Pack kann leicht führen (L1-Buffer amortisiert Dequant über MR=4 Zeilen)
#
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores
from std.random import rand as rnd

from src.linalg.kernels import (
    Matrix, Q4Matrix, DT, SIMD_W, MR, NR_FW, NR_FC,
    matmul_q4_bpack, matmul_q4_bpack_l2,
    matmul_q4_fused_avx2,
    unpack_nibbles, Q4Nibbles,
)

alias BENCH_REPS : Int = 5   # Wiederholungen für stabilen Best-Case


fn _best_ns_bpack(mut C: Matrix, A: Matrix, Bq: Q4Matrix, w: Int) -> UInt:
    C.zero(); matmul_q4_bpack(C, A, Bq, w)   # warm-up
    var best: UInt = UInt.MAX
    for _ in range(BENCH_REPS):
        C.zero()
        var t0 = perf_counter_ns()
        matmul_q4_bpack(C, A, Bq, w)
        var dt = perf_counter_ns() - t0
        if dt < best: best = dt
    return best


fn _best_ns_l2(mut C: Matrix, A: Matrix, Bq: Q4Matrix, w: Int) -> UInt:
    C.zero(); matmul_q4_bpack_l2(C, A, Bq, w)
    var best: UInt = UInt.MAX
    for _ in range(BENCH_REPS):
        C.zero()
        var t0 = perf_counter_ns()
        matmul_q4_bpack_l2(C, A, Bq, w)
        var dt = perf_counter_ns() - t0
        if dt < best: best = dt
    return best


fn _best_ns_fused(mut C: Matrix, A: Matrix, Bq: Q4Matrix, w: Int) -> UInt:
    C.zero(); matmul_q4_fused_avx2(C, A, Bq, w)
    var best: UInt = UInt.MAX
    for _ in range(BENCH_REPS):
        C.zero()
        var t0 = perf_counter_ns()
        matmul_q4_fused_avx2(C, A, Bq, w)
        var dt = perf_counter_ns() - t0
        if dt < best: best = dt
    return best


fn bench_kernels(M: Int, K: Int, N: Int, workers: Int) raises:
    """Vergleicht alle Q4-Kernel-Varianten für eine gegebene Matrix-Größe."""
    var flops = 2.0 * Float64(M) * Float64(K) * Float64(N)
    var Ab    = Matrix(M, K);  Ab.fill_random()
    var Cb    = Matrix(M, N)
    var Bq    = Q4Matrix(K, N, Float32(0.1));  Bq.fill_random()

    print("M=", M, " K=", K, " N=", N)

    var ns_bp    = _best_ns_bpack(Cb, Ab, Bq, workers)
    var ns_l2    = _best_ns_l2(Cb, Ab, Bq, workers)
    var ns_fused = _best_ns_fused(Cb, Ab, Bq, workers)

    var ms_bp    = Float64(Int(ns_bp))    / 1e6
    var ms_l2    = Float64(Int(ns_l2))    / 1e6
    var ms_fused = Float64(Int(ns_fused)) / 1e6
    var gf_bp    = flops / (Float64(Int(ns_bp))    / 1e9) / 1e9
    var gf_l2    = flops / (Float64(Int(ns_l2))    / 1e9) / 1e9
    var gf_fused = flops / (Float64(Int(ns_fused)) / 1e9) / 1e9

    print("  B-Pack (L1):     ", ms_bp,    "ms /", gf_bp,    "GFLOPS")
    print("  B-Pack-L2:       ", ms_l2,    "ms /", gf_l2,    "GFLOPS")
    print("  Fused-AVX2 (32B):", ms_fused, "ms /", gf_fused, "GFLOPS")
    print("  Speedup vs B-Pack:   ", gf_fused / (gf_bp + Float64(1e-9)), "×")
    print("  Speedup vs B-Pack-L2:", gf_fused / (gf_l2 + Float64(1e-9)), "×")
    print()


fn bench_unpack_nibbles() raises:
    """Micro-Benchmark: unpack_nibbles Durchsatz (Dequant-only, ohne FMA)."""
    print("── unpack_nibbles Micro-Benchmark ──────────────────────────")
    var buf   = List[UInt8]()
    buf.resize(32 * 1024, 0)  # 32 KB Test-Daten
    rnd[DType.uint8](buf.unsafe_ptr(), 32 * 1024)
    var ptr   = buf.unsafe_ptr()
    var scale = Float32(0.1)

    # 100k Blöcke à 32 Bytes → 3.2 MB Daten
    var N_BLOCKS = 100_000
    var t0    = perf_counter_ns()
    var dummy = SIMD[DT, 32](0)
    for i in range(N_BLOCKS):
        var bytes = ptr.load[width=32]((i % 1024) * 32)
        var q4    = unpack_nibbles(bytes, scale)
        dummy = dummy + q4.lo   # verhindert Dead-Code-Eliminierung
    var dt_ns = perf_counter_ns() - t0

    var bytes_processed = Float64(N_BLOCKS) * 32.0
    var gb_per_s = bytes_processed / (Float64(Int(dt_ns)) / 1e9) / 1e9
    var floats_per_s = bytes_processed * 2.0 / (Float64(Int(dt_ns)) / 1e9) / 1e9  # 64 floats per 32 bytes
    print("  Blöcke:        ", N_BLOCKS, " × 32 Bytes")
    print("  Durchsatz:     ", gb_per_s, " GB/s  (Q4 Input)")
    print("  Float32-Rate:  ", floats_per_s, " GFloat/s  (dequantisierte Werte)")
    print("  dummy (kein DCE):", dummy[0])
    print()


fn run_q4_benchmark(workers: Int) raises:
    print("══════════════════════════════════════════════════════════════")
    print("  Q4 SIMD Kernel Benchmark")
    print("  Threads:", workers, "  SIMD_W:", SIMD_W, "  NR_FC:", NR_FC)
    print("══════════════════════════════════════════════════════════════")
    print()

    # Micro-Benchmark: reiner Dequant-Durchsatz
    bench_unpack_nibbles()

    # Kernel-Vergleich: verschiedene Matmul-Größen
    print("── Kernel-Vergleich (M×K×N) ─────────────────────────────────")
    print()

    # Kleine Matrizen (Streaming-ähnlich, M=4=MR)
    bench_kernels(4, 1024, 1024, workers)

    # Mittlere Matrizen
    bench_kernels(64, 1024, 1024, workers)

    # Große Matrizen (Gemma-4 27B Demo-ähnlich)
    bench_kernels(4, 4096, 4096, workers)

    # Square (klassischer Benchmark-Stil)
    bench_kernels(1024, 1024, 1024, workers)

    print("══════════════════════════════════════════════════════════════")
