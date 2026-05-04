# gemma4_e4b_infer.mojo
#
# Gemma-4 E4B – Vollständige Inferenz-Pipeline
#
# Voraussetzungen (einmalig ausführen):
#   pixi run python scripts/export_lm_head.py
#   → vocab_proto.bin   (69 KB  –  Token-ID → ASCII)
#   → lm_head_proto.bin (10 MB  –  Q4 LM-Head + BOS-Embedding)
#
# Pipeline:
#   1. SparsePinEngine init (24 Layer gepinnt ~1.13 GB, 18 via pread)
#   2. TokenMap + LMHead laden
#   3. Memory-Map ausgeben
#   4. TTFT: BOS-Token (ID=2) mit echtem Embedding, 42-Layer Forward-Pass
#   5. LM-Head Projektion → Temperature-Sampling → echtes Token-ID
#   6. Streaming: N_STREAM weitere Token generieren und sofort ausgeben
#
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores

from src.linalg.kernels import Matrix, DT
from src.streaming.sparse_pin import SparsePinEngine
from src.tokenizer.gemma_proto import TokenMap, load_token_map
from src.inference.lm_head import (
    LMHead, load_lm_head, apply_bos_embedding,
    project_lm_head, temperature_sampling, get_token_embedding,
)
from src.inference.gemma4_e4b import (
    HybridKVCache, e4b_forward_layer,
    E4B_D, E4B_N_LAYERS, E4B_BATCH,
)

comptime MODEL_PATH  : String = "models/gemma4_e4b_q4.mojostream"
comptime VOCAB_PATH  : String = "vocab_proto.bin"
comptime LM_HEAD_PATH: String = "lm_head_proto.bin"
comptime MAX_FULL_SEQ: Int    = 256
comptime N_STREAM    : Int    = 7
comptime TEMPERATURE : Float32 = 0.7   # Sampling-Temperatur (0 = greedy)


fn prepare_next_input(mut x: Matrix, lm: LMHead, token_id: Int):
    """
    Setzt Zeile 0 von x auf das dequantisierte embed_tokens.weight[token_id].
    Nutzt get_token_embedding: extrahiert Spalte token_id aus der Q4-LM-Head-Matrix.
    Das ist echtes Embedding-Lookup — kein Proxy mehr.
    Zeilen 1-3: Null-Padding (Kernel-Constraint MR=4).
    """
    get_token_embedding(token_id, lm, x.data())
    var ptr = x.data()
    for j in range(E4B_D, E4B_BATCH * E4B_D): ptr.store(j, Float32(0.0))


fn run(workers: Int) raises:
    print("══════════════════════════════════════════════════════════════")
    print("  Gemma-4 E4B  –  Tokenizer Bridge + LM-Head + Streaming")
    print("══════════════════════════════════════════════════════════════")
    print("  Modell:    ", MODEL_PATH)
    print("  Vocab:     ", VOCAB_PATH)
    print("  LM-Head:   ", LM_HEAD_PATH)
    print("  Temperatur:", TEMPERATURE, "  |  Tokens: 1 TTFT +", N_STREAM)
    print()

    # ── [1] SparsePinEngine ────────────────────────────────────────────────
    print("[1/6] Initialisiere SparsePinEngine ...")
    var t_init   = perf_counter_ns()
    var engine   = SparsePinEngine(MODEL_PATH)
    var init_ms  = Float64(Int(perf_counter_ns() - t_init)) / 1e6
    print("  Init:", init_ms, "ms  (24 Layer gepinnt)")

    # ── [2] TokenMap + LMHead laden ───────────────────────────────────────
    print("[2/6] Lade TokenMap + LMHead ...")
    var token_map = load_token_map(VOCAB_PATH)
    var lm        = load_lm_head(LM_HEAD_PATH)
    print("  Vocab: ", token_map.n_tokens, "Token  |  LM-Head vocab_n=",
          lm.vocab_n, "  scale=", lm.scale)
    # Stichprobe
    print("  Beispiele: ID 270=", repr(token_map.decode(270)),
          " ID 1003=", repr(token_map.decode(1003)),
          " ID 2000=", repr(token_map.decode(2000)))

    # ── [3] Memory-Map ────────────────────────────────────────────────────
    print()
    print("[3/6] Memory-Map:")
    engine.print_memory_map()

    # ── [4] KV-Cache + echtes BOS-Embedding ──────────────────────────────
    print("[4/6] Alloziere KV-Cache + initialisiere BOS-Embedding ...")
    var kv = HybridKVCache(MAX_FULL_SEQ)
    var x  = Matrix(E4B_BATCH, E4B_D)
    apply_bos_embedding(x, lm)    # Echtes BOS-Embedding (Token-ID 2)
    print("  KV-Cache:", kv.memory_mb(), "MB")
    print("  BOS-Embedding: echtes embed_tokens.weight[2] (nicht Random)")

    # ── [5] TTFT: BOS-Token durch alle 42 Layer ───────────────────────────
    print()
    print("[5/6] TTFT (BOS ID=2, base_pos=0) ...")
    engine.reset_stats()
    var t_ttft = perf_counter_ns()

    for layer in range(E4B_N_LAYERS):
        engine.advance(layer)
        var w = engine.get_layer_ref(layer)
        e4b_forward_layer(x, layer, w, kv, 0, workers)

        if (layer + 1) % 14 == 0 or layer == E4B_N_LAYERS - 1:
            var e = Float64(Int(perf_counter_ns() - t_ttft)) / 1e6
            print("  Layer", layer + 1, "/ 42   (", e, "ms)")

    var ttft_ms = Float64(Int(perf_counter_ns() - t_ttft)) / 1e6

    # LM-Head Projektion → erstes Token
    var logits0 = Matrix(E4B_BATCH, lm.vocab_n)
    project_lm_head(logits0, x, lm, workers)
    var first_token = temperature_sampling(logits0, TEMPERATURE)
    var first_text  = token_map.decode(first_token)
    print("  TTFT:", ttft_ms, "ms  →  Token", first_token,
          "=", repr(first_text))
    engine.print_io_stats()

    # ── [6] Streaming: N_STREAM weitere Token ─────────────────────────────
    print()
    print("[6/6] Echtzeit-Streaming (T=", TEMPERATURE, ", vocab=",
          lm.vocab_n, ") ...")
    print()
    print("Gemma-4 E4B > ", end="")
    print(first_text, end="")

    var t_stream    = perf_counter_ns()
    var sum_step_ms = Float64(0.0)
    var generated   = List[Int]()
    generated.append(first_token)

    for step in range(1, N_STREAM + 1):
        prepare_next_input(x, lm, generated[step - 1])

        var t_step = perf_counter_ns()
        for layer in range(E4B_N_LAYERS):
            engine.advance(layer)
            var w = engine.get_layer_ref(layer)
            e4b_forward_layer(x, layer, w, kv, step, workers)
        var step_ms = Float64(Int(perf_counter_ns() - t_step)) / 1e6
        sum_step_ms += step_ms

        # LM-Head + Sampling
        var logits = Matrix(E4B_BATCH, lm.vocab_n)
        project_lm_head(logits, x, lm, workers)
        var tok  = temperature_sampling(logits, TEMPERATURE)
        var text = token_map.decode(tok)
        generated.append(tok)

        print(text, end="")
        if token_map.is_eos(tok):
            break

    var stream_ms = Float64(Int(perf_counter_ns() - t_stream)) / 1e6
    print()
    print()
    print("  Generierte Token-IDs: ", end="")
    for i in range(len(generated)):
        print(generated[i], "", end="")
    print()

    # ── Zusammenfassung ───────────────────────────────────────────────────
    var avg_ms = sum_step_ms / Float64(N_STREAM)
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  ERGEBNIS")
    print("══════════════════════════════════════════════════════════════")
    print("  Init (24 Layer pinnen):   ", init_ms, "ms")
    print("  TTFT (BOS → 1. Token):   ", ttft_ms, "ms")
    print("  LM-Head vocab_n:          ", lm.vocab_n)
    print("  Temperatur:               ", TEMPERATURE)
    print("  Streaming (", N_STREAM, "Token):      ", stream_ms, "ms")
    print("  Ø pro Token (nach TTFT):  ", avg_ms, "ms/Token")
    print("  Effektiv t/s:             ", Float64(1000) / avg_ms, "t/s")
    print("  RAM: ~1130 MB pinned + 104 MB buffer + 11 MB LM-Head")
    print("══════════════════════════════════════════════════════════════")


fn main() raises:
    run(num_logical_cores())
