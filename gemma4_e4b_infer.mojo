# gemma4_e4b_infer.mojo
#
# Task 4: Gemma-4 E4B – Minimale CLI-Schleife für TTFT-Messung.
#
# Lädt models/gemma4_e4b_q4.mojostream, speist BOS-Token (ID=2) ein und
# misst die Zeit für den ersten vollständigen 42-Layer Forward-Pass
# (Time-to-First-Token).
#
# Batch = 4 (Kernel-Constraint MR=4). Zeile 0 trägt den BOS-Embedding-
# Vektor; Zeilen 1–3 sind auf Null gepadded und produzieren irrelevante
# Ausgaben. TTFT entspricht Position 0 (erster echter Token).
#
# Einschränkung: Embedding-Tabelle ist nicht im .mojostream gespeichert.
# BOS-Vektor wird als Gauß-Rauschen (N(0, 0.02)) approximiert – realistisch
# genug für TTFT-Timing-Zwecke.
#
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores
from std.random import rand as rnd_fill
from std.memory import UnsafePointer

from src.linalg.kernels import Matrix, PtrT, DT, U8Ptr
from src.streaming.mojostream import MojoStreamFile
from src.inference.gemma4_e4b import (
    HybridKVCache, E4BLayerRef, load_e4b_layer_ref, e4b_forward_layer,
    E4B_D, E4B_N_LAYERS, E4B_BATCH,
    E4B_SLIDE_Q_DIM, E4B_FULL_Q_DIM,
    E4B_SLIDE_KV_DIM, E4B_FULL_KV_DIM,
)

comptime MODEL_PATH : String = "models/gemma4_e4b_q4.mojostream"
comptime MAX_FULL_SEQ: Int = 256    # Full-Attention max. Kontext für diese Session


fn make_bos_embedding(mut x: Matrix):
    """
    Approximiert den BOS-Embedding-Vektor (ID=2).
    Da die Embedding-Tabelle nicht im .mojostream-Format liegt, wird ein
    Gauß-ähnlicher Vektor mit Varianz 0.02 als Platzhalter genutzt.
    Zeilen 1–3 bleiben null (Padding für MR=4-Kernel-Constraint).
    """
    rnd_fill[DT](x.data(), E4B_D)    # Zeile 0: zufällige Aktivierung ≈ BOS
    var ptr = x.data()
    # Skaliere auf plausible Embedding-Magnitude (~0.02)
    var scale_factor = Float32(0.02)
    for j in range(E4B_D):
        ptr.store(j, ptr.load(j) * scale_factor)
    # Zeilen 1–3: explizit null (bereits 0 durch Matrix-Initialisierung)


fn run_ttft(workers: Int) raises:
    print("══════════════════════════════════════════════════════════════")
    print("  Gemma-4 E4B  –  Time-to-First-Token  (BOS ID=2)")
    print("══════════════════════════════════════════════════════════════")
    print("  Modell: ", MODEL_PATH)
    print("  D=2560  FFD=10240  42 Layer  8Q/2KV Heads")
    print("  Sliding (35): head_dim=256  win=512")
    print("  Full    ( 7): head_dim=512  partial_rope=128 dims")
    print("  Threads: ", workers)
    print()

    # ── [1] Modell laden (1.97 GB → RAM) ─────────────────────────────────
    print("[1/4] Lade .mojostream ...")
    var t_load = perf_counter_ns()
    var ms     = MojoStreamFile(MODEL_PATH)
    var load_ms = Float64(Int(perf_counter_ns() - t_load)) / 1e6
    print("  Geladen:", ms.file_size_mb() / 1e3, "GB  in", load_ms, "ms")
    print("  Tensoren:", ms.meta.n_tensors, "  Layer:", ms.meta.n_layers)

    # ── [2] KV-Cache allozieren ───────────────────────────────────────────
    print()
    print("[2/4] Alloziere Hybrid-KV-Cache ...")
    var kv = HybridKVCache(MAX_FULL_SEQ)
    print("  KV-Cache:", kv.memory_mb(), "MB")
    print("    Sliding: 35 Layer × 512 Tokens × 512 Dim × K+V")
    print("    Full:     7 Layer ×", MAX_FULL_SEQ, "Tokens × 1024 Dim × K+V")

    # ── [3] BOS-Embedding (batch=4, Zeile 0 = BOS) ───────────────────────
    print()
    print("[3/4] Erstelle BOS-Embedding (Token ID=2) ...")
    var x = Matrix(E4B_BATCH, E4B_D)   # zero-initialisiert
    make_bos_embedding(x)
    print("  Eingabe: batch=", E4B_BATCH, "  D=", E4B_D,
          "  (Zeilen 1-3 gepadded)")

    # ── [4] TTFT: 42-Layer Forward-Pass ──────────────────────────────────
    print()
    print("[4/4] Starte 42-Layer Forward-Pass (TTFT-Uhr läuft) ...")
    print()

    var base_pos = 0    # BOS ist der erste Token
    var t_ttft   = perf_counter_ns()

    for layer in range(E4B_N_LAYERS):
        var w = load_e4b_layer_ref(ms, layer)
        e4b_forward_layer(x, layer, w, kv, base_pos, workers)

        if (layer + 1) % 7 == 0 or layer == E4B_N_LAYERS - 1:
            var elapsed = Float64(Int(perf_counter_ns() - t_ttft)) / 1e6
            print("  Layer", layer + 1, "/ 42   (", elapsed, "ms bisher)")

    var ttft_ns = perf_counter_ns() - t_ttft
    var ttft_ms = Float64(Int(ttft_ns)) / 1e6

    # ── Ergebnis ──────────────────────────────────────────────────────────
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  TTFT-ERGEBNIS  (Gemma-4 E4B, BOS Token, 42 Layer)")
    print("══════════════════════════════════════════════════════════════")
    print("  TTFT (Forward-Pass):  ", ttft_ms, "ms")
    print("  Davon I/O (Load):     ", load_ms, "ms")
    print("  Reine Compute-Zeit:   ", ttft_ms, "ms  (Gewichte in RAM)")
    print("  Tokens/Sek (effektiv):", Float64(1000) / ttft_ms, "t/s")
    print()
    print("  KV-Cache befüllt:  1 Position (BOS)")
    print("  Batch-Constraint:  4  (MR=4 Kernel, Zeilen 1-3 = Padding)")
    print("══════════════════════════════════════════════════════════════")


fn main() raises:
    run_ttft(num_logical_cores())
