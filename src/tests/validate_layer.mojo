# src/tests/validate_layer.mojo
#
# Numerische Validierung – Layer-0-Forward-Pass gegen Python-Referenz.
#
# Pipeline:
#   1. model.mojostream laden → ShapeGuard-Check
#   2. layer0_ref.bin laden (Input + Python-Referenz-Output)
#   3. Mojo-Forward-Pass mit den SELBEN Gewichten und dem SELBEN Input
#   4. RMSE und relative Abweichung berechnen
#   5. PASS / FAIL-Urteil (Schwelle: relative RMSE < 1e-4)
#
# Fehlerquellen: float32 Rundungsfehler durch SIMD-Blocking (BK=128)
# vs. NumPy-float64-Akkumulation. Erwartet: rel. RMSE ≈ 1e-5 → [PASS]
#
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores
from std.math import sqrt

from src.streaming.mojostream import (
    MojoStreamFile, MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN,
)
from src.main import (
    Gemma4Config, KVCache, Gemma4LayerWeights,
    gemma4_forward_layer,
)
from src.bench.harness import load_weights_from_ms
from src.linalg.kernels import Matrix, PtrT, DT, SIMD_W

alias REF_MAGIC : UInt32 = 0x42464552   # "REFB"
alias VAL_BATCH : Int    = 4
alias PASS_TOL  : Float64 = 1e-4        # relative RMSE-Schwelle


# ── NaN / Inf Stabilitätsprüfung ─────────────────────────────────────────────

fn has_nan_or_inf(m: Matrix) -> Bool:
    """Gibt True zurück wenn die Matrix NaN oder Inf enthält (IEEE 754).
    Genutzt nach einem Forward-Pass mit echten Gewichten um Overflow/Underflow
    zu erkennen bevor mit dem Modell weitergearbeitet wird."""
    var ptr = m.data()
    var n   = m.rows * m.cols
    for i in range(n):
        var v = ptr.load(i)
        if v != v:   # NaN: IEEE 754 – NaN ist das einzige Float das ungleich sich selbst ist
            return True
        if v > Float32(1e30) or v < Float32(-1e30):   # +/- Inf
            return True
    return False


fn check_matrix_stability(m: Matrix, name: String) -> Bool:
    """Prüft Matrix auf NaN/Inf und gibt detaillierten Report.
    Returns True wenn stabil."""
    if has_nan_or_inf(m):
        print("  [INSTABIL] ", name, ": NaN oder Inf gefunden!")
        return False
    var ptr = m.data()
    var n   = m.rows * m.cols
    var max_abs = Float32(0)
    for i in range(n):
        var v = ptr.load(i)
        var av = v if v >= Float32(0) else -v
        if av > max_abs: max_abs = av
    print("  [OK]       ", name, ": max_abs =", max_abs)
    return True


# ── Referenzdatei laden ──────────────────────────────────────────────────────

struct RefData(Movable):
    var input_buf:  List[UInt8]   # batch * hidden float32
    var output_buf: List[UInt8]   # batch * hidden float32
    var hidden:     Int
    var batch:      Int

    fn __init__(out self):
        self.input_buf  = List[UInt8]()
        self.output_buf = List[UInt8]()
        self.hidden     = 0
        self.batch      = 0

    fn input_ptr(self) -> PtrT:
        return rebind[PtrT](self.input_buf.unsafe_ptr())

    fn output_ptr(self) -> PtrT:
        return rebind[PtrT](self.output_buf.unsafe_ptr())


fn load_ref(path: String) raises -> RefData:
    var raw = List[UInt8]()
    with open(path, "r") as f:
        raw = f.read_bytes()
    var bp = raw.unsafe_ptr()

    var magic  = bp.bitcast[UInt32]().load(0)
    var hidden = Int(bp.bitcast[UInt32]().load(2))
    var batch  = Int(bp.bitcast[UInt32]().load(3))

    if magic != REF_MAGIC:
        raise Error("layer0_ref.bin: ungültige Magic-Zahl")

    var n_floats = batch * hidden
    var data_pos = 16   # nach Header
    var in_bytes = n_floats * 4
    var out_bytes = n_floats * 4

    var r = RefData()
    r.hidden = hidden
    r.batch  = batch

    r.input_buf.resize(in_bytes, 0)
    r.output_buf.resize(out_bytes, 0)

    var src_in = bp + data_pos
    var src_out = bp + data_pos + in_bytes
    var dst_in  = r.input_buf.unsafe_ptr()
    var dst_out = r.output_buf.unsafe_ptr()

    for i in range(in_bytes):  dst_in.store(i,  src_in.load(i))
    for i in range(out_bytes): dst_out.store(i, src_out.load(i))

    return r^


# ── Fehlerberechnung ─────────────────────────────────────────────────────────

struct ErrorMetrics(Copyable, Movable):
    var abs_rmse: Float64
    var rms_ref:  Float64
    var rel_rmse: Float64

    fn __init__(out self, a: Float64, r: Float64, rel: Float64):
        self.abs_rmse = a; self.rms_ref = r; self.rel_rmse = rel

    fn copy(self) -> Self: return Self(self.abs_rmse, self.rms_ref, self.rel_rmse)^


fn compute_errors(mojo: Matrix, ref_ptr: PtrT, n: Int) -> ErrorMetrics:
    """Berechnet abs_rmse, rms_ref und rel_rmse zwischen Mojo-Output und Referenz."""
    var mojo_ptr    = mojo.data()
    var sum_sq_diff : Float64 = 0.0
    var sum_sq_ref  : Float64 = 0.0

    for i in range(n):
        var m = Float64(mojo_ptr.load(i))
        var rv = Float64(ref_ptr.load(i))
        var d = m - rv
        sum_sq_diff += d * d
        sum_sq_ref  += rv * rv

    var abs_rmse = Float64(sqrt(Float32(sum_sq_diff / Float64(n))))
    var rms_ref  = Float64(sqrt(Float32(sum_sq_ref  / Float64(n))))
    var rel_rmse = abs_rmse / (rms_ref + Float64(1e-12))
    return ErrorMetrics(abs_rmse, rms_ref, rel_rmse)


# ── Haupt-Validierungsroutine ────────────────────────────────────────────────

fn run_validation(ms_path: String, ref_path: String, workers: Int) raises:
    print("══════════════════════════════════════════════════════════════")
    print("  MojoStream Numerical Validation – Layer 0")
    print("══════════════════════════════════════════════════════════════")

    # ── 1. .mojostream laden + ShapeGuard ─────────────────────────────────
    print("\n[1] Lade", ms_path, "...")
    var ms = MojoStreamFile(ms_path)
    ms.validate_gemma4_shape()
    print("  ShapeGuard: OK  D=", ms.meta.hidden, " KV=", ms.meta.kv_dim,
          " FFN=", ms.meta.ffn_dim, " Heads=", ms.meta.n_heads, "/", ms.meta.n_kv_heads)

    # ── 2. Referenzdaten laden ─────────────────────────────────────────────
    print("\n[2] Lade Referenz", ref_path, "...")
    var ref_data = load_ref(ref_path)
    print("  Referenz: hidden=", ref_data.hidden, " batch=", ref_data.batch)

    if ref_data.hidden != ms.meta.hidden:
        raise Error("Referenz und Modell haben unterschiedliche hidden_size")

    # ── 3. Gemma4Config aus Modell-Metadaten ──────────────────────────────
    var cfg = Gemma4Config(
        hidden     = ms.meta.hidden,
        kv_dim     = ms.meta.kv_dim,
        ffn_dim    = ms.meta.ffn_dim,
        n_layers   = ms.meta.n_layers,
        n_heads    = ms.meta.n_heads,
        n_kv_heads = ms.meta.n_kv_heads,
    )

    # ── 4. Layer-0-Gewichte aus mojostream bauen ───────────────────────────
    print("\n[3] Baue Layer-0-Gewichte aus Puffer ...")
    var w = Gemma4LayerWeights()
    load_weights_from_ms(ms, 0, w, ms.meta.hidden, ms.meta.kv_dim, ms.meta.ffn_dim)
    print("  PLE-Skala:", w.ple_scale, "  Q-Skala:", w.scale_Q)

    # ── 5. Input aus Referenzdatei in Matrix laden ─────────────────────────
    var x = Matrix(VAL_BATCH, ms.meta.hidden)
    var x_dst = x.data()
    var x_src = ref_data.input_ptr()
    for i in range(VAL_BATCH * ms.meta.hidden):
        x_dst.store(i, x_src.load(i))

    var kv = KVCache(1, ms.meta.kv_dim, 8)   # mini-Cache nur für diesen Test

    # ── 6. Mojo Forward-Pass ──────────────────────────────────────────────
    print("\n[4] Führe Mojo-Forward-Pass durch (Layer 0) ...")
    var t0 = perf_counter_ns()
    gemma4_forward_layer(x, 0, w, kv, cfg, 0, workers)
    var ms_time = Float64(Int(perf_counter_ns() - t0)) / 1e6
    print("  Mojo-Zeit:", ms_time, "ms")

    # ── 7. Fehlerberechnung ───────────────────────────────────────────────
    var n_total = VAL_BATCH * ms.meta.hidden
    var errs = compute_errors(x, ref_data.output_ptr(), n_total)

    # ── 8. PASS / FAIL-Urteil ─────────────────────────────────────────────
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  VALIDIERUNGSERGEBNIS  Layer 0")
    print("══════════════════════════════════════════════════════════════")
    print("  Absolute RMSE:   ", errs.abs_rmse)
    print("  RMS(Referenz):   ", errs.rms_ref)
    print("  Relative RMSE:   ", errs.rel_rmse, " (Schwelle:", PASS_TOL, ")")

    if errs.rel_rmse < PASS_TOL:
        print()
        print("  [PASS] Validation Layer 0: Error:", errs.rel_rmse)
        print("  Numerische Übereinstimmung innerhalb float32-Toleranz ✓")
    else:
        print()
        print("  [FAIL] Validation Layer 0: Error:", errs.rel_rmse)
        print("  Relative RMSE überschreitet Schwelle", PASS_TOL)
        print("  → Implementierungsfehler im Forward-Pass")
    print("══════════════════════════════════════════════════════════════")
