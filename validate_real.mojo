# validate_real.mojo – Validierung mit konvertierten Gemma-4-Gewichten
#
# Nutzung:
#   python3 scripts/gen_reference.py <model.mojostream>  # optionaler Ref-Output
#   pixi run mojo validate_real.mojo [model.mojostream]
#
# Was dieser Test prüft:
#   1. ShapeGuard + TensorGuard beim Laden (5 Kriterien pro Tensor)
#   2. Numerische Stabilität (kein NaN / Inf nach Layer-0-Forward-Pass)
#   3. Relative RMSE vs. NumPy-Referenz (falls layer0_ref.bin vorhanden)
#
# Mit echten Gewichten und korrekter Architektur:
#   [PASS] ShapeGuard/TensorGuard
#   [PASS] Stabilität: kein NaN, max_abs im realistischen Bereich
#   [PASS] rel. RMSE < 1e-4  (falls Referenz vorhanden)
#
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores

from src.streaming.stream_runner import (
    StreamingRunner, gemma4_forward_stream, rss_mb,
)
from src.tests.validate_layer import (
    has_nan_or_inf, check_matrix_stability,
    load_ref, compute_errors, PASS_TOL, VAL_BATCH,
)
from src.main import (
    Gemma4Config, KVCache,
)
from src.linalg.kernels import Matrix


fn run_real_validation(ms_path: String, ref_path: String, workers: Int) raises:
    print("══════════════════════════════════════════════════════════════")
    print("  MojoStream Real-Weight Validation")
    print("  Datei:", ms_path)
    print("══════════════════════════════════════════════════════════════")

    # ── 1. Laden + Strict Metadata Guard ──────────────────────────────────
    print("\n[1] ShapeGuard + TensorGuard ...")
    var rss0    = rss_mb()
    var runner  = StreamingRunner(ms_path)
    print("  ShapeGuard:  OK")
    print("  TensorGuard: OK  (", runner.meta.n_tensors, "Einträge geprüft)")
    print("  RSS nach Init:", rss_mb(), "MB  (vorher:", rss0, "MB)")

    var cfg = Gemma4Config(
        hidden     = runner.meta.hidden,
        kv_dim     = runner.meta.kv_dim,
        ffn_dim    = runner.meta.ffn_dim,
        n_layers   = runner.meta.n_layers,
        n_heads    = runner.meta.n_heads,
        n_kv_heads = runner.meta.n_kv_heads,
    )
    print("  Modell: D=", cfg.hidden, " KV=", cfg.kv_dim,
          " FFN=", cfg.ffn_dim, " Layers=", cfg.n_layers)

    # ── 2. Stabilitätstest: Layer-0 Forward-Pass ──────────────────────────
    print("\n[2] Numerischer Stabilitätstest (Layer 0, batch=4) ...")

    var kv = KVCache(cfg.n_layers, cfg.kv_dim, 8)
    var x  = Matrix(VAL_BATCH, cfg.hidden)
    x.fill_random()

    var t0 = perf_counter_ns()
    var w  = runner.load_layer(0)
    var dt_io = perf_counter_ns() - t0

    t0 = perf_counter_ns()
    gemma4_forward_stream(x, 0, w, kv, cfg, 0, workers)
    var dt_compute = perf_counter_ns() - t0

    print("  I/O:     ", Float64(Int(dt_io))      / 1e6, "ms")
    print("  Compute: ", Float64(Int(dt_compute))  / 1e6, "ms")

    var stable = check_matrix_stability(x, "Layer-0-Output")

    if stable:
        print("  [PASS] Numerische Stabilität: kein NaN / Inf ✓")
    else:
        print("  [FAIL] Numerische Instabilität — Forward-Pass produziert NaN/Inf!")
        print("         Mögliche Ursachen:")
        print("           • Quantisierungsskala zu groß (scale > 1000)")
        print("           • Falsche Gewichts-Transposition im Converter")
        print("           • Architekt-Parameter (D/KVD/FFD) stimmen nicht")
        runner.close()
        return

    # ── 3. Referenz-Vergleich (optional) ──────────────────────────────────
    var raw_check = List[UInt8]()
    var has_ref   = False
    try:
        with open(ref_path, "r") as f:
            raw_check = f.read_bytes()
        has_ref = True
    except:
        pass

    if has_ref:
        print("\n[3] Referenz-Vergleich ...")
        var ref_data = load_ref(ref_path)
        if ref_data.hidden != cfg.hidden:
            print("  WARNUNG: Referenz-hidden (", ref_data.hidden,
                  ") ≠ Modell-hidden (", cfg.hidden, ") — überspringe Vergleich")
        else:
            # Input aus Referenz-Datei laden
            var x2  = Matrix(VAL_BATCH, cfg.hidden)
            var kv2 = KVCache(cfg.n_layers, cfg.kv_dim, 8)
            var xd  = x2.data()
            var xr  = ref_data.input_ptr()
            for i in range(VAL_BATCH * cfg.hidden):
                xd.store(i, xr.load(i))

            var w2 = runner.load_layer(0)
            gemma4_forward_stream(x2, 0, w2, kv2, cfg, 0, workers)

            var errs = compute_errors(x2, ref_data.output_ptr(),
                                      VAL_BATCH * cfg.hidden)
            print("  abs. RMSE:  ", errs.abs_rmse)
            print("  rel. RMSE:  ", errs.rel_rmse, " (Schwelle:", PASS_TOL, ")")

            if errs.rel_rmse < PASS_TOL:
                print("  [PASS] Numerische Übereinstimmung ✓  Error:",
                      errs.rel_rmse)
            else:
                print("  [FAIL] rel. RMSE überschreitet Schwelle")
    else:
        print("\n[3] Kein Referenz-Output (layer0_ref.bin nicht gefunden)")
        print("    Generieren mit:  python3 scripts/gen_reference.py", ms_path)

    runner.close()

    # ── Zusammenfassung ────────────────────────────────────────────────────
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  ShapeGuard/TensorGuard: [PASS]")
    if stable:
        print("  Stabilität:             [PASS]")
    else:
        print("  Stabilität:             [FAIL]")
    print("  Peak RSS:              ", rss_mb(), "MB")
    print("══════════════════════════════════════════════════════════════")


fn main() raises:
    var ms_path  = String("model_converted.mojostream")
    var ref_path = String("layer0_ref.bin")

    # Einfaches argv-Handling (erstes Argument = mojostream-Pfad)
    from std.sys.info import num_logical_cores as _ncores
    run_real_validation(ms_path, ref_path, num_logical_cores())
