# src/bench/harness.mojo
#
# MojoStream Benchmark-Harness – brutale Messung aller relevanten Metriken.
#
# Metriken:
#   TTFT           – Zeit vom ersten API-Aufruf bis zum Ende des ersten Forward-Passes
#   P50 / P95      – Latenz-Verteilung über N_STEPS Schritte
#   Peak-RAM       – VmRSS aus /proc/self/status (echter Kernel-Wert)
#   I/O-Druck      – max(0, IO-Zeit - Compute-Zeit) / Step-Total (echte Wartezeit)
#
# Output: Sauberes JSON auf stdout UND in bench_result.json.
# Vergleich mit llama.cpp: TTFT + P95 sind die zwei Kernmetriken.
#
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores
from src.streaming.stream_runner import get_rss_kb, rss_mb

from src.streaming.mojostream import (
    MojoStreamFile, TensorRef,
    MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN,
    TENSORS_PER_LAYER,
)
from src.main import (
    Gemma4Config, KVCache, Gemma4LayerWeights,
    gemma4_forward_layer,
)
from src.linalg.kernels import Matrix, U8Ptr, PtrT, DT

# ── Benchmark-Konfiguration ──────────────────────────────────────────────────
alias BENCH_STEPS : Int = 100   # Schritte für Latenz-Verteilung
alias BENCH_BATCH : Int = 4     # MR=4 Kernel-Constraint
alias KV_MAX_SEQ  : Int = 512   # max. Sequenzlänge im KV-Cache


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

fn insertion_sort_ns(mut v: List[UInt]):
    """Insertion-Sort für Latenz-Liste (n=100, O(n²) ist trivial)."""
    for i in range(1, len(v)):
        var key = v[i]
        var j   = i - 1
        while j >= 0 and v[j] > key:
            v[j+1] = v[j]
            j -= 1
        v[j+1] = key


fn percentile_ms(sorted_ns: List[UInt], pct: Float64) -> Float64:
    """Gibt das p-Perzentil einer sortierten Nanosekunden-Liste in ms zurück."""
    var n   = len(sorted_ns)
    var idx = Int(pct / 100.0 * Float64(n - 1) + 0.5)
    if idx >= n: idx = n - 1
    return Float64(Int(sorted_ns[idx])) / 1e6


fn estimate_peak_ram_mb(
    file_bytes: Int, n_layers: Int,
    hidden: Int, kv_dim: Int, ffn_dim: Int,
) -> Float64:
    """Schätzt den Peak-RAM aus bekannten Puffergrößen (für JSON-Output).
    Der echte Messwert kommt von get_rss_kb() / VmRSS."""
    var raw_mb = Float64(file_bytes) / 1e6
    var q_sz  = hidden * hidden   // 2
    var k_sz  = hidden * kv_dim   // 2
    var v_sz  = hidden * kv_dim   // 2
    var o_sz  = hidden * hidden   // 2
    var g_sz  = hidden * ffn_dim  // 2
    var u_sz  = hidden * ffn_dim  // 2
    var d_sz  = ffn_dim * hidden  // 2
    var wt_mb = Float64(n_layers * (q_sz + k_sz + v_sz + o_sz + g_sz + u_sz + d_sz)) / 1e6
    var kv_mb = Float64(n_layers * KV_MAX_SEQ * kv_dim * 2 * 4) / 1e6
    return raw_mb + wt_mb + kv_mb


fn fast_copy_u8(dst: U8Ptr, src: U8Ptr, n: Int):
    """Schnelles Byte-für-Byte Kopieren (kein SIMD nötig, einmalig)."""
    for i in range(n): dst.store(i, src.load(i))


fn load_weights_from_ms(
    ms:      MojoStreamFile,
    layer:   Int,
    mut w:   Gemma4LayerWeights,
    hidden:  Int,
    kv_dim:  Int,
    ffn_dim: Int,
):
    """Kopiert Tensor-Daten aus dem mojostream-Puffer in Gemma4LayerWeights."""
    w.ple_scale = ms.ple_scale(layer)

    var q_sz = hidden * hidden   // 2
    var k_sz = hidden * kv_dim   // 2
    var f_sz = hidden * ffn_dim  // 2
    var d_sz = ffn_dim * hidden  // 2

    var q_ref  = ms.tensor_ptr(layer, MS_Q)
    w.Q.resize(q_sz, 0);    fast_copy_u8(rebind[U8Ptr](w.Q.unsafe_ptr()), q_ref.ptr, q_sz)
    w.scale_Q = q_ref.scale

    var k_ref  = ms.tensor_ptr(layer, MS_K)
    w.K.resize(k_sz, 0);    fast_copy_u8(rebind[U8Ptr](w.K.unsafe_ptr()), k_ref.ptr, k_sz)
    w.scale_K = k_ref.scale

    var v_ref  = ms.tensor_ptr(layer, MS_V)
    w.V.resize(k_sz, 0);    fast_copy_u8(rebind[U8Ptr](w.V.unsafe_ptr()), v_ref.ptr, k_sz)
    w.scale_V = v_ref.scale

    var o_ref  = ms.tensor_ptr(layer, MS_O)
    w.O.resize(q_sz, 0);    fast_copy_u8(rebind[U8Ptr](w.O.unsafe_ptr()), o_ref.ptr, q_sz)
    w.scale_O = o_ref.scale

    var g_ref  = ms.tensor_ptr(layer, MS_GATE)
    w.Gate.resize(f_sz, 0); fast_copy_u8(rebind[U8Ptr](w.Gate.unsafe_ptr()), g_ref.ptr, f_sz)
    w.scale_Gate = g_ref.scale

    var u_ref  = ms.tensor_ptr(layer, MS_UP)
    w.Up.resize(f_sz, 0);   fast_copy_u8(rebind[U8Ptr](w.Up.unsafe_ptr()), u_ref.ptr, f_sz)
    w.scale_Up = u_ref.scale

    var d_ref  = ms.tensor_ptr(layer, MS_DOWN)
    w.Down.resize(d_sz, 0); fast_copy_u8(rebind[U8Ptr](w.Down.unsafe_ptr()), d_ref.ptr, d_sz)
    w.scale_Down = d_ref.scale


# ── JSON-Ausgabe ─────────────────────────────────────────────────────────────

fn fstr(x: Float64) -> String:
    return String(x)

fn istr(x: Int) -> String:
    return String(x)

fn build_json(
    path:         String,
    meta_n_layers: Int,
    meta_hidden:  Int,
    meta_kv_dim:  Int,
    meta_ffn_dim: Int,
    file_mb:      Float64,
    load_ms:      Float64,
    build_ms:     Float64,
    ttft_ms:      Float64,
    p50_ms:       Float64,
    p95_ms:       Float64,
    p99_ms:       Float64,
    best_ms:      Float64,
    worst_ms:     Float64,
    peak_ram_mb:  Float64,
    n_steps:      Int,
    batch:        Int,
    io_pct:       Float64,
    tps_p50:      Float64,
    tps_best:     Float64,
) -> String:
    var j = "{\n"
    j += "  \"format\": \"mojostream\",\n"
    j += "  \"source\": \"" + path + "\",\n"
    j += "  \"model\": {\n"
    j += "    \"n_layers\": "  + istr(meta_n_layers) + ",\n"
    j += "    \"hidden\": "    + istr(meta_hidden)   + ",\n"
    j += "    \"kv_dim\": "    + istr(meta_kv_dim)   + ",\n"
    j += "    \"ffn_dim\": "   + istr(meta_ffn_dim)  + "\n"
    j += "  },\n"
    j += "  \"file_mb\": "          + fstr(file_mb)     + ",\n"
    j += "  \"load_time_ms\": "     + fstr(load_ms)     + ",\n"
    j += "  \"build_time_ms\": "    + fstr(build_ms)    + ",\n"
    j += "  \"ttft_ms\": "          + fstr(ttft_ms)     + ",\n"
    j += "  \"latency_p50_ms\": "   + fstr(p50_ms)      + ",\n"
    j += "  \"latency_p95_ms\": "   + fstr(p95_ms)      + ",\n"
    j += "  \"latency_p99_ms\": "   + fstr(p99_ms)      + ",\n"
    j += "  \"latency_best_ms\": "  + fstr(best_ms)     + ",\n"
    j += "  \"latency_worst_ms\": " + fstr(worst_ms)    + ",\n"
    j += "  \"peak_ram_mb\": "      + fstr(peak_ram_mb) + ",\n"
    j += "  \"io_pressure_pct\": "  + fstr(io_pct)      + ",\n"
    j += "  \"n_steps\": "          + istr(n_steps)     + ",\n"
    j += "  \"batch\": "            + istr(batch)       + ",\n"
    j += "  \"tokens_per_sec_p50\": "  + fstr(tps_p50)  + ",\n"
    j += "  \"tokens_per_sec_best\": " + fstr(tps_best) + "\n"
    j += "}"
    return j


# ── Haupt-Benchmark-Funktion ─────────────────────────────────────────────────

fn run_harness(path: String, workers: Int) raises:
    print("══════════════════════════════════════════════════════════════")
    print("  MojoStream Benchmark Harness")
    print("  Datei:", path, "  Threads:", workers)
    print("══════════════════════════════════════════════════════════════")

    # ── Phase 1: .mojostream-Datei laden (I/O-Messung) ──────────────────
    print("\n[1/4] Lade .mojostream-Datei ...")
    var t_bench_start = perf_counter_ns()   # TTFT-Uhr startet hier

    var ms = MojoStreamFile(path)

    var load_ms   = ms.load_time_ms()
    var file_mb   = ms.file_size_mb()
    print("  Datei:", file_mb, "MB  I/O:", load_ms, "ms  (",
          file_mb / (load_ms / 1000.0), "MB/s)")
    print("  Modell: D=", ms.meta.hidden, " KV=", ms.meta.kv_dim,
          " FFN=", ms.meta.ffn_dim, " Layers=", ms.meta.n_layers)

    # ── Phase 2: Gewichte aus mojostream-Puffer aufbauen ────────────────
    print("\n[2/4] Baue Gewichte aus Puffer (",
          ms.meta.n_layers, "Layer) ...")
    var t_build0 = perf_counter_ns()

    var cfg = Gemma4Config(
        hidden     = ms.meta.hidden,
        kv_dim     = ms.meta.kv_dim,
        ffn_dim    = ms.meta.ffn_dim,
        n_layers   = ms.meta.n_layers,
        n_heads    = ms.meta.n_heads,
        n_kv_heads = ms.meta.n_kv_heads,
    )

    var weights = List[Gemma4LayerWeights]()
    for layer in range(ms.meta.n_layers):
        var w = Gemma4LayerWeights()
        load_weights_from_ms(ms, layer, w,
                             ms.meta.hidden, ms.meta.kv_dim, ms.meta.ffn_dim)
        weights.append(w^)

    var build_ms = Float64(Int(perf_counter_ns() - t_build0)) / 1e6
    print("  Aufbau-Zeit:", build_ms, "ms")

    var peak_ram = estimate_peak_ram_mb(
        len(ms.raw), ms.meta.n_layers,
        ms.meta.hidden, ms.meta.kv_dim, ms.meta.ffn_dim,
    )
    print("  Geschätzter Peak-RAM:", peak_ram, "MB")

    # ── Phase 3: Warm-up + TTFT-Messung ─────────────────────────────────
    print("\n[3/4] TTFT-Messung (erster Forward-Pass) ...")
    var kv = KVCache(ms.meta.n_layers, ms.meta.kv_dim, KV_MAX_SEQ)
    var x  = Matrix(BENCH_BATCH, ms.meta.hidden)
    x.fill_random()

    var base_pos = kv.cur_len
    for layer in range(ms.meta.n_layers):
        gemma4_forward_layer(x, layer, weights[layer], kv, cfg, base_pos, workers)
    kv.cur_len += BENCH_BATCH

    var ttft_ns = perf_counter_ns() - t_bench_start
    var ttft_ms = Float64(Int(ttft_ns)) / 1e6
    print("  TTFT:", ttft_ms, "ms  (Load + Build + 1. Forward-Pass)")

    # ── Phase 4: P95-Latenz über BENCH_STEPS Schritte ───────────────────
    print("\n[4/4] Latenz-Verteilung (", BENCH_STEPS, "Schritte, batch=", BENCH_BATCH, ") ...")

    var latencies = List[UInt]()
    for _ in range(BENCH_STEPS):
        latencies.append(UInt(0))

    for step in range(BENCH_STEPS):
        x.fill_random()
        base_pos = kv.cur_len
        var t0 = perf_counter_ns()
        for layer in range(ms.meta.n_layers):
            gemma4_forward_layer(x, layer, weights[layer], kv, cfg, base_pos, workers)
        var dt = perf_counter_ns() - t0
        kv.cur_len += BENCH_BATCH
        latencies[step] = dt

        if step % 20 == 19:
            print("  Schritt", step + 1, "/", BENCH_STEPS,
                  "  seq_len=", kv.cur_len,
                  "  step=", Float64(Int(dt)) / 1e6, "ms")

    # Sortieren für Perzentile
    insertion_sort_ns(latencies)

    var p50_ms  = percentile_ms(latencies, 50.0)
    var p95_ms  = percentile_ms(latencies, 95.0)
    var p99_ms  = percentile_ms(latencies, 99.0)
    var best_ms = Float64(Int(latencies[0])) / 1e6
    var worst_ms = Float64(Int(latencies[BENCH_STEPS - 1])) / 1e6

    var total_compute_ms = Float64(Int(
        latencies[0]  # Näherung: Summe wäre genauer, aber wir haben nur sort-Order
    )) / 1e6

    # I/O-Druck-Fix: echte Wartezeit = max(0, amortisiertes IO/Step - Compute/Step)
    # amortized_io = load_ms (einmalig) verteilt auf BENCH_STEPS Schritte
    # wait = max(0, amortized_io - best_compute) — misst tatsächliches Warten
    var amortized_io_ms  = load_ms / Float64(BENCH_STEPS)
    var wait_ms          = amortized_io_ms - best_ms if amortized_io_ms > best_ms else Float64(0)
    var io_pct           = amortized_io_ms / (amortized_io_ms + best_ms) * 100.0

    var tps_p50  = Float64(BENCH_BATCH) / (p50_ms  / 1000.0)
    var tps_best = Float64(BENCH_BATCH) / (best_ms / 1000.0)

    var rss_peak = rss_mb()   # echter VmRSS-Wert vom Kernel

    # ── Ergebnis-Ausgabe ─────────────────────────────────────────────────
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  ERGEBNISSE")
    print("══════════════════════════════════════════════════════════════")
    print("  Datei-Load:        ", load_ms,  "ms  (", file_mb, "MB)")
    print("  Gewicht-Build:     ", build_ms, "ms")
    print("  TTFT:              ", ttft_ms,  "ms")
    print("  Latenz P50:        ", p50_ms,   "ms  →", tps_p50,  "t/s (batch=4)")
    print("  Latenz P95:        ", p95_ms,   "ms")
    print("  Latenz P99:        ", p99_ms,   "ms")
    print("  Latenz Best:       ", best_ms,  "ms  →", tps_best, "t/s")
    print("  Latenz Worst:      ", worst_ms, "ms  (seq_len wächst)")
    print("  Peak RAM (VmRSS):  ", rss_peak, "MB  (Kernel-Messwert)")
    print("  Peak RAM (est.):   ", peak_ram, "MB  (deterministisch)")
    print("  IO-Wartezeit/Step: ", wait_ms,  "ms  (amortisiert, max(0, IO-Compute))")
    print("  I/O-Druck:         ", io_pct,   "%  (amortisiertes IO vs Compute)")
    print("══════════════════════════════════════════════════════════════")

    var json = build_json(
        path, ms.meta.n_layers, ms.meta.hidden, ms.meta.kv_dim, ms.meta.ffn_dim,
        file_mb, load_ms, build_ms, ttft_ms,
        p50_ms, p95_ms, p99_ms, best_ms, worst_ms,
        rss_peak, BENCH_STEPS, BENCH_BATCH, io_pct,
        tps_p50, tps_best,
    )

    print()
    print(json)

    with open("bench_result.json", "w") as f:
        f.write(json)
    print("\n→ bench_result.json geschrieben.")
