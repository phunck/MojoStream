# gemma4_e4b_infer.mojo
#
# Gemma-4 E4B – TTFT-Messung + Echtzeit-Token-Streaming
#
# Ablauf:
#   1. 1.97 GB .mojostream laden (SATA: ~5.3 s)
#   2. BOS-Token (ID=2) als approximierten Embedding-Vektor einspeisen
#   3. TTFT: vollständiger 42-Layer Forward-Pass, Zeit messen
#   4. Streaming: 7 weitere Token generieren, jedes sofort ausgeben
#
# Kernel-Constraint: batch=4 (MR=4). Zeile 0 = echter Token,
# Zeilen 1–3 = Null-Padding.
#
# Embedding-Näherung: Da die Embedding-Tabelle nicht im .mojostream liegt,
# wird der BOS-Vektor als N(0, 0.02) approximiert (richtige Größenordnung).
# Anschließend wird der Aktivierungsvektor jedes Steps renormiert als
# Proxy-Embedding für den nächsten Token.
#
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores
from std.random import rand as rnd_fill

from src.linalg.kernels import Matrix, PtrT, DT, U8Ptr
from src.streaming.mojostream import MojoStreamFile
from src.inference.gemma4_e4b import (
    HybridKVCache, E4BLayerRef, load_e4b_layer_ref, e4b_forward_layer,
    E4B_D, E4B_N_LAYERS, E4B_BATCH,
)

comptime MODEL_PATH  : String = "models/gemma4_e4b_q4.mojostream"
comptime MAX_FULL_SEQ: Int    = 256
comptime N_STREAM    : Int    = 7      # Tokens nach TTFT (gesamt = 1+7 = 8)


# ── BOS-Embedding ─────────────────────────────────────────────────────────────

fn make_bos_embedding(mut x: Matrix):
    rnd_fill[DT](x.data(), E4B_D)
    var ptr   = x.data()
    var scale = Float32(0.02)
    for j in range(E4B_D):
        ptr.store(j, ptr.load(j) * scale)
    # Zeilen 1–3: Null-Padding (bereits 0 durch Matrix-Initialisierung)


# ── Proxy-Logit: argmax |x[0, 0:512]| ────────────────────────────────────────

fn proxy_argmax(x: Matrix) -> Int:
    """Gibt argmax der absoluten Aktivierungen zurück als Proxy-Token-ID."""
    var ptr    = x.data()
    var best   = 0
    var best_v = Float32(0.0)
    for j in range(512):
        var v = ptr.load(j)
        if v < Float32(0): v = -v
        if v > best_v:
            best_v = v
            best   = j
    return best


# ── Aktivierung für nächsten Schritt renormieren ──────────────────────────────

fn renorm_for_next_step(mut x: Matrix):
    """Skaliert x[0] auf Magnitude ~0.02 als Proxy-Embedding für nächsten Token."""
    var ptr    = x.data()
    var max_v  = Float32(1e-6)
    for j in range(E4B_D):
        var v = ptr.load(j)
        if v < Float32(0): v = -v
        if v > max_v: max_v = v
    var scale = Float32(0.02) / max_v
    for j in range(E4B_D):
        ptr.store(j, ptr.load(j) * scale)
    # Zeilen 1–3 auf Null setzen (Padding für Kernel-Constraint)
    for j in range(E4B_D, E4B_BATCH * E4B_D):
        ptr.store(j, Float32(0.0))


# ── Streaming-Ausgabe: direkt auf stdout, kein Buffering ─────────────────────

fn stream_write(s: String):
    """Schreibt String sofort auf stdout via write()-Syscall (kein Buffering)."""
    print(s, end="")


fn stream_char(c: Int):
    """Schreibt ein druckbares ASCII-Zeichen sofort auf stdout."""
    print(chr(c), end="")


fn stream_newline():
    print()


# ── Haupt-Inferenzschleife ────────────────────────────────────────────────────

fn run_streaming(workers: Int) raises:
    """
    Lädt das Modell einmalig, misst TTFT für den BOS-Token und streamt
    anschließend N_STREAM weitere Token Zeichen für Zeichen ins Terminal.
    """
    print("══════════════════════════════════════════════════════════════")
    print("  Gemma-4 E4B  –  TTFT + Echtzeit Token-Streaming")
    print("══════════════════════════════════════════════════════════════")
    print("  Modell:  ", MODEL_PATH)
    print("  Threads: ", workers, "  |  Tokens gesamt:", 1 + N_STREAM)
    print()

    # ── [1] Modell laden ──────────────────────────────────────────────
    print("[1/4] Lade .mojostream ...")
    var t_load = perf_counter_ns()
    var ms     = MojoStreamFile(MODEL_PATH)
    var load_ms = Float64(Int(perf_counter_ns() - t_load)) / 1e6
    print("  Geladen:", ms.file_size_mb() / 1e3, "GB  in", load_ms, "ms")

    # ── [2] KV-Cache + BOS-Embedding ─────────────────────────────────
    print("[2/4] Alloziere KV-Cache (", HybridKVCache(MAX_FULL_SEQ).memory_mb(), "MB) ...")
    var kv = HybridKVCache(MAX_FULL_SEQ)
    var x  = Matrix(E4B_BATCH, E4B_D)
    make_bos_embedding(x)

    # ── [3] TTFT: Step 0 (BOS-Token, alle 42 Layer) ───────────────────
    print("[3/4] TTFT-Messung (BOS Token ID=2, base_pos=0) ...")
    var t_ttft = perf_counter_ns()
    for layer in range(E4B_N_LAYERS):
        var w = load_e4b_layer_ref(ms, layer)
        e4b_forward_layer(x, layer, w, kv, 0, workers)
    var ttft_ms = Float64(Int(perf_counter_ns() - t_ttft)) / 1e6
    print("  TTFT:", ttft_ms, "ms")
    print()

    # ── [4] Streaming: N_STREAM weitere Token ─────────────────────────
    print("[4/4] Echtzeit-Streaming  (jedes Zeichen erscheint sofort) ...")
    print()
    stream_write("MojoStream > ")

    # Erstes "Zeichen": Proxy-Token aus dem TTFT-Ergebnis
    var first_tok = proxy_argmax(x)
    stream_char(32 + first_tok % 95)

    var t_stream    = perf_counter_ns()
    var sum_step_ms = Float64(0.0)

    for step in range(1, N_STREAM + 1):
        # x für den nächsten Token vorbereiten
        renorm_for_next_step(x)

        # 42-Layer Forward-Pass für Token an Position `step`
        var t_step = perf_counter_ns()
        for layer in range(E4B_N_LAYERS):
            var w = load_e4b_layer_ref(ms, layer)
            e4b_forward_layer(x, layer, w, kv, step, workers)
        var step_ms = Float64(Int(perf_counter_ns() - t_step)) / 1e6
        sum_step_ms += step_ms

        # Sofort ausgeben – kein Warten auf Sequenzende
        var tok = proxy_argmax(x)
        stream_char(32 + tok % 95)

    var stream_ms = Float64(Int(perf_counter_ns() - t_stream)) / 1e6
    stream_newline()
    stream_newline()

    # ── Zusammenfassung ───────────────────────────────────────────────
    var avg_ms = sum_step_ms / Float64(N_STREAM)

    print("══════════════════════════════════════════════════════════════")
    print("  ERGEBNIS")
    print("══════════════════════════════════════════════════════════════")
    print("  I/O (Modell laden):     ", load_ms, "ms  (", ms.file_size_mb()/1e3, "GB)")
    print("  TTFT (1. Token):        ", ttft_ms, "ms")
    print("  Streaming (", N_STREAM, "Tokens):   ", stream_ms, "ms")
    print("  Ø pro Token (Streaming):", avg_ms, "ms/Token")
    print("  Effektiv t/s:           ",
          Float64(1000) / avg_ms, "t/s  (nach TTFT)")
    print("  KV-Cache Footprint:     ", kv.memory_mb(), "MB")
    print("══════════════════════════════════════════════════════════════")


fn main() raises:
    run_streaming(num_logical_cores())
