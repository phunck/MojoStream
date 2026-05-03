# gemma4_e4b_infer.mojo
#
# Gemma-4 E4B – Tactical Sparse Pinning + Token-Streaming
#
# Ablauf:
#   1. SparsePinEngine initialisieren: 24 Layer in RAM pinnen (~1.13 GB),
#      18 Layer via pread on-demand von SSD; Doppel-Buffer für I/O-Pipeline.
#   2. Memory-Map ausgeben: LIVE / STREAMED Übersicht.
#   3. TTFT: BOS-Token (ID=2), 42-Layer Forward-Pass, Zeit messen.
#   4. Streaming: 7 weitere Token generieren, jedes sofort ausgeben.
#
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores
from std.random import rand as rnd_fill

from src.linalg.kernels import Matrix, DT
from src.streaming.sparse_pin import SparsePinEngine
from src.inference.gemma4_e4b import (
    HybridKVCache, e4b_forward_layer,
    E4B_D, E4B_N_LAYERS, E4B_BATCH,
)

comptime MODEL_PATH  : String = "models/gemma4_e4b_q4.mojostream"
comptime MAX_FULL_SEQ: Int    = 256
comptime N_STREAM    : Int    = 7


# ── Embedding-Approximation (BOS ID=2) ───────────────────────────────────────

fn make_bos_embedding(mut x: Matrix):
    rnd_fill[DT](x.data(), E4B_D)
    var ptr   = x.data()
    var scale = Float32(0.02)
    for j in range(E4B_D): ptr.store(j, ptr.load(j) * scale)


fn proxy_argmax(x: Matrix) -> Int:
    var ptr    = x.data()
    var best   = 0
    var best_v = Float32(0.0)
    for j in range(512):
        var v = ptr.load(j)
        if v < Float32(0): v = -v
        if v > best_v: best_v = v; best = j
    return best


fn renorm_for_next_step(mut x: Matrix):
    var ptr   = x.data()
    var max_v = Float32(1e-6)
    for j in range(E4B_D):
        var v = ptr.load(j)
        if v < Float32(0): v = -v
        if v > max_v: max_v = v
    var scale = Float32(0.02) / max_v
    for j in range(E4B_D): ptr.store(j, ptr.load(j) * scale)
    for j in range(E4B_D, E4B_BATCH * E4B_D): ptr.store(j, Float32(0.0))


# ── Haupt-Inferenzschleife ────────────────────────────────────────────────────

fn run(workers: Int) raises:
    print("══════════════════════════════════════════════════════════════")
    print("  Gemma-4 E4B  –  Sparse Pinning + Token Streaming")
    print("══════════════════════════════════════════════════════════════")
    print("  Modell:  ", MODEL_PATH)
    print("  Threads: ", workers, "  |  Tokens: 1 TTFT +", N_STREAM, "Stream")
    print()

    # ── [1] SparsePinEngine initialisieren ────────────────────────────────
    print("[1/5] Initialisiere SparsePinEngine ...")
    print("      (Pinne 24 Layer; verbleibende 18 bleiben auf SSD)")
    var t_init = perf_counter_ns()
    var engine = SparsePinEngine(MODEL_PATH)
    var init_ms = Float64(Int(perf_counter_ns() - t_init)) / 1e6
    print("  Init:", init_ms, "ms  (inkl. pread 24 gepinnter Layer)")

    # ── [2] Memory-Map ausgeben ────────────────────────────────────────────
    print()
    print("[2/5] Memory-Map (LIVE = RAM, STREAMED = SSD on-demand):")
    engine.print_memory_map()

    # ── [3] KV-Cache + BOS-Embedding ──────────────────────────────────────
    print("[3/5] Alloziere KV-Cache + BOS-Embedding ...")
    var kv = HybridKVCache(MAX_FULL_SEQ)
    var x  = Matrix(E4B_BATCH, E4B_D)
    make_bos_embedding(x)
    print("  KV-Cache:", kv.memory_mb(), "MB  |  Embedding: D=", E4B_D)

    # ── [4] TTFT: Step 0 (alle 42 Layer) ──────────────────────────────────
    print()
    print("[4/5] TTFT (BOS Token, base_pos=0) ...")
    engine.reset_stats()
    var t_ttft = perf_counter_ns()

    for layer in range(E4B_N_LAYERS):
        engine.advance(layer)
        var w = engine.get_layer_ref(layer)
        e4b_forward_layer(x, layer, w, kv, 0, workers)

        if (layer + 1) % 7 == 0 or layer == E4B_N_LAYERS - 1:
            var elapsed = Float64(Int(perf_counter_ns() - t_ttft)) / 1e6
            print("  Layer", layer + 1, "/ 42   (", elapsed, "ms)")

    var ttft_ms = Float64(Int(perf_counter_ns() - t_ttft)) / 1e6
    print("  TTFT:", ttft_ms, "ms")
    engine.print_io_stats()

    # ── [5] Streaming: N_STREAM weitere Token ─────────────────────────────
    print()
    print("[5/5] Echtzeit-Streaming (", N_STREAM, "Token nach TTFT) ...")
    print()
    print("MojoStream > ", end="")

    var tok0 = proxy_argmax(x)
    print(chr(32 + tok0 % 95), end="")

    var t_stream    = perf_counter_ns()
    var sum_step_ms = Float64(0.0)

    for step in range(1, N_STREAM + 1):
        renorm_for_next_step(x)
        engine.reset_stats()

        var t_step = perf_counter_ns()
        for layer in range(E4B_N_LAYERS):
            engine.advance(layer)
            var w = engine.get_layer_ref(layer)
            e4b_forward_layer(x, layer, w, kv, step, workers)
        var step_ms = Float64(Int(perf_counter_ns() - t_step)) / 1e6
        sum_step_ms += step_ms

        var tok = proxy_argmax(x)
        print(chr(32 + tok % 95), end="")

    var stream_ms = Float64(Int(perf_counter_ns() - t_stream)) / 1e6
    print()
    print()
    engine.print_io_stats()

    # ── Ergebnis ──────────────────────────────────────────────────────────
    var avg_ms = sum_step_ms / Float64(N_STREAM)

    print()
    print("══════════════════════════════════════════════════════════════")
    print("  ERGEBNIS")
    print("══════════════════════════════════════════════════════════════")
    print("  Init (24 Layer pinnen):   ", init_ms, "ms")
    print("  TTFT (1. Token):          ", ttft_ms, "ms")
    print("  Streaming (", N_STREAM, "Token):      ", stream_ms, "ms")
    print("  Ø pro Token (nach TTFT):  ", avg_ms, "ms/Token")
    print("  Effektiv t/s:             ", Float64(1000) / avg_ms, "t/s")
    print("  KV-Cache Footprint:       ", kv.memory_mb(), "MB")
    print("  Pinned RAM:               ~1130 MB")
    print("  Stream-Buffer (×2):       ~104 MB")
    print("  RAM gesamt (Schätzung):   ~1234 MB  (vs 1973 MB full load)")
    print("══════════════════════════════════════════════════════════════")


fn main() raises:
    run(num_logical_cores())
