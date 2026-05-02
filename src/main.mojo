# src/main.mojo – Q4 Inference Benchmark (Tokens/Sekunde)
#
# Simuliert einen vereinfachten LLM-Forward-Pass:
#   for layer in 0..N_LAYERS:
#     x = rmsnorm(x @ W_Q4)   # linear projection + normalisierung
#
# Zwei Benchmarks:
#   (A) COMPUTE-only: Gewichte pre-loaded in RAM (eliminiert I/O)
#   (B) STREAMING:    Gewichte Layer-für-Layer von SSD geladen
#
# Batch-Größe = MR = 4 (prozessiert 4 Token gleichzeitig; MR-aligned → kein Tail)
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores
from std.memory import UnsafePointer

from src.linalg.kernels import (
    Matrix, U8Ptr, PtrT, DT, SIMD_W,
    matmul_q4_prepacked, matmul_q4_bpack,
    rmsnorm_inplace, num_logical_cores,
)

# ── Konfiguration ────────────────────────────────────────────────────────────
alias D          = 4096   # Modell-Dimension (hidden size)
alias N_LAYERS   = 40     # Anzahl Layer
alias BATCH      = 4      # Token-Batch (MR=4, kein Tail-Handling nötig)
alias N_STEPS    = 3      # Benchmark-Wiederholungen für stabile Messung
alias PACKED_DIR = "model_weights_packed"
alias ROW_DIR    = "model_weights"
# ─────────────────────────────────────────────────────────────────────────────

alias PACKED_SIZE = D * (D // 2)   # Bytes pro Layer (= 8.388.608 für D=4096)


# ── Gewicht-Loader: liest eine Layer-Datei (beide Formate) ───────────────────
fn load_weight(dst: U8Ptr, packed_size: Int, path: String) raises -> Float32:
    var sc: Float32 = 0.0
    with open(path, "r") as f:
        var raw  = f.read_bytes()
        var rptr = raw.unsafe_ptr()
        sc = rptr.bitcast[Float32]().load(0)
        var src = rptr + 4
        for i in range(packed_size):
            dst.store(i, src.load(i))
    return sc


# ── Kernstück: ein Forward-Schritt durch alle Layer ──────────────────────────
fn forward_step(
    mut x:     Matrix,              # (BATCH, D) Aktivierungen, in-place-Update
    mut tmp:   Matrix,              # (BATCH, D) Arbeits-Buffer
    weights:   List[List[UInt8]],   # pre-loaded: weights[layer][raw bytes]
    scales:    List[Float32],
    workers:   Int,
):
    for layer in range(N_LAYERS):
        var src = rebind[U8Ptr](weights[layer].unsafe_ptr())
        tmp.zero()
        matmul_q4_prepacked(tmp, x, src, scales[layer], D, D, workers)
        rmsnorm_inplace(tmp.data(), BATCH, D)

        # Aktivierungen für nächsten Layer übernehmen (pointer-freier Swap via Copy)
        var xd = x.data()
        var td = tmp.data()
        for i in range(BATCH * D):
            xd.store(i, td.load(i))


# ── Compute-Only Benchmark: alles in RAM ──────────────────────────────────────
fn bench_compute_only(workers: Int) raises:
    print("═══════════════════════════════════════════════════════")
    print("  (A) COMPUTE-ONLY BENCHMARK (Gewichte in RAM)")
    print("═══════════════════════════════════════════════════════")
    print("Lade", N_LAYERS, "Layer à", PACKED_SIZE / 1e6, "MB in RAM ...")

    var weights = List[List[UInt8]]()
    var scales  = List[Float32]()

    for i in range(N_LAYERS):
        var buf = List[UInt8]()
        buf.resize(PACKED_SIZE, 0)
        var path  = PACKED_DIR + "/layer_" + String(i) + ".bin"
        var sc    = load_weight(rebind[U8Ptr](buf.unsafe_ptr()), PACKED_SIZE, path)
        weights.append(buf^)
        scales.append(sc)

    print("RAM belegt: ca.", Float64(N_LAYERS * PACKED_SIZE) / 1e6, "MB")
    print("Starte Benchmark (", N_STEPS, "Schritte, batch=", BATCH, ") ...")

    var x   = Matrix(BATCH, D);  x.fill_random()
    var tmp = Matrix(BATCH, D)

    # Warm-up
    forward_step(x, tmp, weights, scales, workers)

    var best: UInt = UInt.MAX
    for _ in range(N_STEPS):
        x.fill_random()
        var t0 = perf_counter_ns()
        forward_step(x, tmp, weights, scales, workers)
        var dt = perf_counter_ns() - t0
        if dt < best: best = dt

    var ms_step   = Float64(Int(best)) / 1e6
    var tps_batch = Float64(BATCH) / (Float64(Int(best)) / 1e9)
    var tps_single = 1.0 / (Float64(Int(best)) / 1e9)
    print()
    print("Compute-Zeit pro Step:  ", ms_step, "ms")
    print("Tokens/Sek (batch=4):   ", tps_batch, "t/s")
    print("Tokens/Sek (batch=1 eff):", tps_single, "t/s")


# ── Streaming Benchmark: Layer-für-Layer von SSD ─────────────────────────────
fn bench_streaming(workers: Int) raises:
    print()
    print("═══════════════════════════════════════════════════════")
    print("  (B) STREAMING BENCHMARK (Layer-für-Layer von SSD)")
    print("═══════════════════════════════════════════════════════")

    var x   = Matrix(BATCH, D);  x.fill_random()
    var tmp = Matrix(BATCH, D)

    var layer_buf = List[UInt8]()
    layer_buf.resize(PACKED_SIZE, 0)
    var lptr = rebind[U8Ptr](layer_buf.unsafe_ptr())

    var best: UInt = UInt.MAX

    # Warm-up: 1 vollständiger Durchlauf
    for layer in range(N_LAYERS):
        var path  = PACKED_DIR + "/layer_" + String(layer) + ".bin"
        var scale = load_weight(lptr, PACKED_SIZE, path)
        tmp.zero()
        matmul_q4_prepacked(tmp, x, lptr, scale, D, D, workers)
        rmsnorm_inplace(tmp.data(), BATCH, D)
        var xd = x.data(); var td = tmp.data()
        for i in range(BATCH * D): xd.store(i, td.load(i))

    for _ in range(N_STEPS):
        x.fill_random()
        var t0 = perf_counter_ns()
        for layer in range(N_LAYERS):
            var path  = PACKED_DIR + "/layer_" + String(layer) + ".bin"
            var scale = load_weight(lptr, PACKED_SIZE, path)
            tmp.zero()
            matmul_q4_prepacked(tmp, x, lptr, scale, D, D, workers)
            rmsnorm_inplace(tmp.data(), BATCH, D)
            var xd = x.data(); var td = tmp.data()
            for i in range(BATCH * D): xd.store(i, td.load(i))
        var dt = perf_counter_ns() - t0
        if dt < best: best = dt

    var ms_step = Float64(Int(best)) / 1e6
    var io_mb   = Float64(N_LAYERS * PACKED_SIZE) / 1e6
    var io_gbs  = io_mb / ms_step

    print("Streaming-Zeit pro Step:    ", ms_step, "ms")
    print("  davon I/O (", io_mb, "MB):  ca.", io_mb / io_gbs, "ms (~", io_gbs, "GB/s)")
    print("Tokens/Sek (batch=4):        ", Float64(BATCH) / (Float64(Int(best)) / 1e9), "t/s")
    print("Tokens/Sek (batch=1 eff):    ", 1.0 / (Float64(Int(best)) / 1e9), "t/s")


# ── Kernel-Scaling-Vergleich ─────────────────────────────────────────────────
fn bench_scaling(workers: Int) raises:
    from src.linalg.kernels import Q4Matrix
    from std.random import rand as rand_u8
    print()
    print("═══════════════════════════════════════════════════════")
    print("  KERNEL SCALING  (BPack vs. Pre-Packed, N=1024/4096)")
    print("═══════════════════════════════════════════════════════")

    for pass_n in range(2):
        var nb      = 1024 if pass_n == 0 else 4096
        var flops_b = 2.0 * Float64(nb) * Float64(nb) * Float64(nb)
        var Ab  = Matrix(nb, nb);  Ab.fill_random()
        var Cb  = Matrix(nb, nb)
        var Q4  = Q4Matrix(nb, nb, Float32(0.1));  Q4.fill_random()

        matmul_q4_bpack(Cb, Ab, Q4, workers)   # warm-up
        var best1: UInt = UInt.MAX
        for _ in range(3):
            Cb.zero(); var t0 = perf_counter_ns()
            matmul_q4_bpack(Cb, Ab, Q4, workers)
            var dt = perf_counter_ns() - t0
            if dt < best1: best1 = dt

        var prepack_buf = List[UInt8](); prepack_buf.resize(nb * (nb // 2), 0)
        var pp_ptr = rebind[U8Ptr](prepack_buf.unsafe_ptr())
        rand_u8[DType.uint8](pp_ptr, nb * (nb // 2))

        matmul_q4_prepacked(Cb, Ab, pp_ptr, Float32(0.1), nb, nb, workers)
        var best2: UInt = UInt.MAX
        for _ in range(3):
            Cb.zero(); var t0 = perf_counter_ns()
            matmul_q4_prepacked(Cb, Ab, pp_ptr, Float32(0.1), nb, nb, workers)
            var dt = perf_counter_ns() - t0
            if dt < best2: best2 = dt

        var ms1 = Float64(Int(best1)) / 1e6; var gf1 = flops_b / (Float64(Int(best1)) / 1e9) / 1e9
        var ms2 = Float64(Int(best2)) / 1e6; var gf2 = flops_b / (Float64(Int(best2)) / 1e9) / 1e9
        print("N=", nb, "  BPack:", ms1, "ms /", gf1,
              "GFLOPS  |  Pre-Packed:", ms2, "ms /", gf2, "GFLOPS")


fn main() raises:
    var ncores = num_logical_cores()
    print("══════════════════════════════════════════════════════════")
    print("  LLM Inference Benchmark  (D=", D, " Layers=", N_LAYERS, " Threads=", ncores, ")")
    print("══════════════════════════════════════════════════════════")

    bench_scaling(ncores)
    bench_compute_only(ncores)
    bench_streaming(ncores)
