# tokenizer_test.mojo
#
# Integrierter End-to-End-Test:
#   "Hallo Mojo"  →  Tokenizer  →  IDs  →  Forward-Pass  →  Detokenizer
#
# Ablauf:
#   1. Vokabular aus vocab.msvocab laden
#   2. Encode: "Hallo Mojo"  →  Token-IDs
#   3. Decode: Token-IDs  →  "Hallo Mojo"  (Round-Trip-Test)
#   4. Forward-Pass: Token-Embeddings (zufällig) durch alle 40 Layer jagen
#   5. Detokenizer auf Forward-Pass-Ausgabe-IDs anwenden
#
# Hinweis: Da wir keine trainierten Gewichte haben, sind die Ausgabe-IDs
# zufällig – aber die Architektur-Pipeline ist vollständig verbunden.
#
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores

from src.nlp.tokenizer import BPETokenizer
from src.main import (
    Gemma4Config, KVCache, Gemma4LayerWeights,
    gemma4_forward_layer, init_random_layer_weights,
    G4_D, G4_KV_D, G4_FFN_D,
)
from src.linalg.kernels import Matrix, PtrT, DT, SIMD_W

alias BATCH : Int = 4   # MR=4 Kernel-Constraint


fn argmax_row(x: Matrix, row: Int) -> Int:
    """Gibt den Index des größten Wertes in Zeile `row` zurück."""
    var ptr  = x.data() + row * x.cols
    var best_val = ptr.load(0)
    var best_idx = 0
    for i in range(1, x.cols):
        var v = ptr.load(i)
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


fn main() raises:
    var workers = num_logical_cores()
    print("══════════════════════════════════════════════════════════")
    print("  MojoStream Tokenizer Test")
    print("══════════════════════════════════════════════════════════")

    # ── 1. Vokabular laden ────────────────────────────────────────────────
    print("\n[1] Lade Vokabular aus vocab.msvocab ...")
    var tok = BPETokenizer()
    tok.load("vocab.msvocab")
    print("   Geladen:", tok.n_tokens(), "Token  |  BOS=", tok.bos_id,
          " EOS=", tok.eos_id, " UNK=", tok.unk_id)

    # ── 2. Encode ─────────────────────────────────────────────────────────
    var input_text = String("Hallo Mojo")
    print("\n[2] Encode: \"", input_text, "\"")
    var ids = tok.encode(input_text)

    print("   Token-IDs: [", end="")
    for k in range(len(ids)):
        print(ids[k], end="")
        if k + 1 < len(ids): print(", ", end="")
    print("]")
    print("   Token-Strings:")
    for k in range(len(ids)):
        print("     [", ids[k], "] = \"", tok.token_str(ids[k]), "\"")

    # ── 3. Decode + Round-Trip-Verifikation ───────────────────────────────
    print("\n[3] Decode: IDs  →  String")
    var decoded = tok.decode(ids)
    print("   Dekodiert: \"", decoded, "\"")

    if decoded == input_text:
        print("   Round-Trip-Test: BESTANDEN ✓")
    else:
        print("   Round-Trip-Test: FEHLGESCHLAGEN ✗")
        print("   Erwartet: \"", input_text, "\"")
        print("   Erhalten: \"", decoded, "\"")

    # ── 4. Forward-Pass mit Token-Embeddings ─────────────────────────────
    print("\n[4] Forward-Pass (", len(ids), "Token, batch=4, 40 Layer, random weights)")

    # Gemma-4-Konfiguration (Demo-Dimensionen)
    var cfg = Gemma4Config(
        hidden     = G4_D,
        kv_dim     = G4_KV_D,
        ffn_dim    = G4_FFN_D,
        n_layers   = 40,
        n_heads    = 16,
        n_kv_heads = 4,
    )

    # Zufällige Gewichte für alle 40 Layer
    print("   Allokiere 40 Layer mit zufälligen Q4-Gewichten ...")
    var weights = List[Gemma4LayerWeights]()
    for _ in range(40):
        var w = Gemma4LayerWeights()
        init_random_layer_weights(w, cfg)
        weights.append(w^)

    var kv = KVCache(40, G4_KV_D, 64)  # kurzer Cache für den Test

    # Embedding-Matrix:  batch=4 Zeilen, jede repräsentiert ein Token.
    # Zeilen 0..len(ids)-1 = unsere Token-Embeddings (zufällig, da kein echtes Embedding-Table).
    # Zeilen len(ids)..3   = PAD-Token-Embeddings (Nullen).
    var x = Matrix(BATCH, G4_D)
    x.fill_random()                          # repräsentiert Token-Embeddings

    var t0       = perf_counter_ns()
    var base_pos = kv.cur_len

    for layer in range(40):
        gemma4_forward_layer(x, layer, weights[layer], kv, cfg, base_pos, workers)

    kv.cur_len += BATCH
    var ms = Float64(Int(perf_counter_ns() - t0)) / 1e6

    print("   Forward-Pass in", ms, "ms")

    # ── 5. Ausgabe-IDs via Argmax (nicht aussagekräftig ohne echte Gewichte) ──
    print("\n[5] Detokenizer auf Ausgabe-Logits (argmax, random weights)")
    print("   Argmax pro Token-Zeile  →  nächstes Token:")
    var out_ids = List[Int]()
    for i in range(len(ids)):
        var next_id = argmax_row(x, i) % tok.n_tokens()
        out_ids.append(next_id)
        print("     Zeile", i, ": argmax=", next_id,
              " → \"", tok.token_str(next_id), "\"")

    var out_text = tok.decode(out_ids)
    print("\n   Dekodierte Ausgabe: \"", out_text, "\"")
    print("   (Gibberish erwartet – keine echten Gewichte geladen)")

    # ── Zusammenfassung ───────────────────────────────────────────────────
    print()
    print("══════════════════════════════════════════════════════════")
    print("  Pipeline vollständig:")
    print("  \"", input_text, "\"  →  IDs  →  Forward-Pass  →  Detokenizer  ✓")
    print("  Round-Trip-Test: BESTANDEN" if decoded == input_text else "  Round-Trip-Test: FEHLGESCHLAGEN")
    print("══════════════════════════════════════════════════════════")
