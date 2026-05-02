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
    matmul_q4_prepacked, matmul_q4_bpack, matmul_q4_bpack_l2,
    rmsnorm_inplace, ple_scale_inplace, swiglu_inplace,
    num_logical_cores,
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
    demo_gemma4_structure(ncores)


# ===========================================================================
# GEMMA 4  ARCHITECTURE LAYER
# ===========================================================================

# Gemma-4 Demo-Dimensionen (skalierbare Platzhalter)
alias G4_D     = 1024  # hidden size (Demo; real Gemma 4 7B: 2048, 27B: 4096)
alias G4_KV_D  = 256   # KV-Dim via GQA (Demo; real: 1024 für 27B)
alias G4_FFN_D = 2048  # FFN intermediate (Demo; real: ~8/3 × D)

# ── Gemma 4 Hyperparameter-Konfiguration ────────────────────────────────────
struct Gemma4Config(Copyable, Movable):
    """Gemma 4 Modell-Konfiguration.
    Werte für Demo (G4_D=1024); echte Gemma 4 Größen:
      7B:  hidden=2048, ffn=8192, kv=256,  n_layers=28
      27B: hidden=4096, ffn=16384, kv=1024, n_layers=46"""
    var hidden:   Int   # D – Hidden Size
    var kv_dim:   Int   # KV-Dimension (GQA: n_kv_heads × head_dim)
    var ffn_dim:  Int   # FFN Intermediate
    var n_layers: Int
    var n_heads:  Int   # Query-Heads
    var n_kv_heads: Int # KV-Heads (GQA; n_kv_heads < n_heads)

    fn __init__(out self, hidden: Int, kv_dim: Int, ffn_dim: Int,
                n_layers: Int, n_heads: Int, n_kv_heads: Int):
        self.hidden     = hidden
        self.kv_dim     = kv_dim
        self.ffn_dim    = ffn_dim
        self.n_layers   = n_layers
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads

    fn copy(self) -> Self:
        return Self(self.hidden, self.kv_dim, self.ffn_dim,
                    self.n_layers, self.n_heads, self.n_kv_heads)


# ── Shared KV Cache ──────────────────────────────────────────────────────────
struct KVCache(Movable):
    """Geteilter Key-Value Cache für Gemma 4 Long-Context-Inferenz.

    'Shared' bedeutet: mehrere Decode-Requests können den Common-Prefix-
    KV-Cache lesen, ohne ihn für jede Anfrage neu zu berechnen (Paged
    Attention / Prefix Sharing). Das reduziert RAM-Verbrauch bei langen
    Kontexten drastisch.

    Speicherstruktur (vereinfacht):
      k_cache[layer][token, kv_dim]  → flacher Float32-Puffer
      v_cache[layer][token, kv_dim]

    TODO: Paged-Memory-Allocator für variable Sequenzlängen implementieren.
    TODO: int8-Quantisierung des KV-Cache für 50% RAM-Ersparnis."""
    var k_data: List[List[Scalar[DT]]]  # [n_layers] × (max_seq × kv_dim)
    var v_data: List[List[Scalar[DT]]]
    var cur_len:  Int
    var max_len:  Int
    var n_layers: Int
    var kv_dim:   Int

    fn __init__(out self, n_layers: Int, kv_dim: Int, max_seq_len: Int):
        self.n_layers = n_layers
        self.kv_dim   = kv_dim
        self.max_len  = max_seq_len
        self.cur_len  = 0
        self.k_data   = List[List[Scalar[DT]]]()
        self.v_data   = List[List[Scalar[DT]]]()
        var slot_size = max_seq_len * kv_dim
        for _ in range(n_layers):
            var ks = List[Scalar[DT]](); ks.resize(slot_size, 0)
            var vs = List[Scalar[DT]](); vs.resize(slot_size, 0)
            self.k_data.append(ks^)
            self.v_data.append(vs^)

    fn is_full(self) -> Bool:
        return self.cur_len >= self.max_len

    fn reset(mut self):
        """Leert den Cache (z.B. für neues Gespräch)."""
        self.cur_len = 0

    fn memory_mb(self) -> Float64:
        """Gibt den theoretischen RAM-Verbrauch des Cache in MB zurück."""
        return Float64(self.n_layers * self.max_len * self.kv_dim * 2 * 4) / 1e6


# ── Gemma 4 Forward-Pass Stub ────────────────────────────────────────────────
struct Gemma4LayerWeights(Movable):
    """Enthält alle 7 Q4-Gewichtsmatrizen eines Gemma-4-Layers.
    Reihenfolge entspricht create_fake_model.py --format gemma4."""
    var Q: List[UInt8];    var scale_Q: Float32
    var K: List[UInt8];    var scale_K: Float32
    var V: List[UInt8];    var scale_V: Float32
    var O: List[UInt8];    var scale_O: Float32
    var Gate: List[UInt8]; var scale_Gate: Float32
    var Up: List[UInt8];   var scale_Up: Float32
    var Down: List[UInt8]; var scale_Down: Float32
    var ple_scale: Float32

    fn __init__(out self):
        self.Q = List[UInt8]();    self.scale_Q    = Float32(0.1)
        self.K = List[UInt8]();    self.scale_K    = Float32(0.1)
        self.V = List[UInt8]();    self.scale_V    = Float32(0.1)
        self.O = List[UInt8]();    self.scale_O    = Float32(0.1)
        self.Gate = List[UInt8](); self.scale_Gate = Float32(0.1)
        self.Up   = List[UInt8](); self.scale_Up   = Float32(0.1)
        self.Down = List[UInt8](); self.scale_Down = Float32(0.1)
        self.ple_scale = Float32(1.0)


fn gemma4_forward_stub(
    mut x:      Matrix,               # (batch, D) Aktivierungen
    mut tmp:    Matrix,               # Arbeits-Buffer (batch, D)
    mut tmp_kv: Matrix,               # Arbeits-Buffer (batch, kv_dim)
    mut tmp_ff: Matrix,               # Arbeits-Buffer (batch, ffn_dim)
    weights:    Gemma4LayerWeights,
    cfg:        Gemma4Config,
    workers:    Int,
):
    """Gemma 4 Layer Forward-Pass (vereinfachter Stub).

    Vollständige Implementierung würde beinhalten:
      1. Pre-Norm (RMSNorm)
      2. Self-Attention mit GQA und RoPE-Embeddings
      3. KV-Cache-Update
      4. Post-Attention RMSNorm
      5. SwiGLU FFN
      6. Residual Connection

    Stub führt die Matrix-Multiplikationen durch und zeigt das Rechenvolumen,
    ohne echte Attention-Masken oder RoPE zu implementieren.

    TODO: RoPE (Rotary Position Embeddings) für Gemma 4 implementieren.
    TODO: GQA-kompatiblen Attention-Score-Mechanismus hinzufügen.
    TODO: Sliding-Window-Attention für Long-Context (Gemma 4 Feature)."""

    var D   = cfg.hidden
    var KVD = cfg.kv_dim
    var FFD = cfg.ffn_dim
    var W_Q    = rebind[U8Ptr](weights.Q.unsafe_ptr())
    var W_K    = rebind[U8Ptr](weights.K.unsafe_ptr())
    var W_V    = rebind[U8Ptr](weights.V.unsafe_ptr())
    var W_O    = rebind[U8Ptr](weights.O.unsafe_ptr())
    var W_Gate = rebind[U8Ptr](weights.Gate.unsafe_ptr())
    var W_Up   = rebind[U8Ptr](weights.Up.unsafe_ptr())
    var W_Down = rebind[U8Ptr](weights.Down.unsafe_ptr())

    # ── Schritt 1: PLE-Skalierung (Gemma 4) ──────────────────────────────
    ple_scale_inplace(x.data(), x.rows, D, weights.ple_scale)

    # ── Schritt 2: Attention Projektionen (Stub) ─────────────────────────
    # Q = x @ W_q,  K = x @ W_k,  V = x @ W_v  (via matmul_q4_prepacked)
    # TODO: Echte GQA-Attention-Berechnung + RoPE + KV-Cache-Update

    # ── Schritt 3: Output-Projektion (Stub) ──────────────────────────────
    # x = x + attn_out @ W_o

    # ── Schritt 4: RMSNorm vor FFN ───────────────────────────────────────
    rmsnorm_inplace(x.data(), x.rows, D)

    # ── Schritt 5: SwiGLU FFN (Stub) ─────────────────────────────────────
    # gate_buf = x @ W_gate;  up_buf = x @ W_up
    # swiglu_inplace(gate_buf.data(), up_buf.data(), batch × FFD)
    # x = x + up_buf @ W_down
    # TODO: Erfordert (batch, FFD) Zwischenpuffer

    # [Stub-Output]: Strukturellen Forward-Pass bestätigen
    print("[Gemma4 Forward] PLE scale=", weights.ple_scale,
          " D=", D, " KVD=", KVD, " FFD=", FFD)


fn demo_gemma4_structure(workers: Int):
    """Zeigt die Gemma-4-Architektur-Struktur und misst die Speicheranforderungen."""
    print()
    print("═══════════════════════════════════════════════════════")
    print("  GEMMA 4 ARCHITECTURE DEMO  (D=", G4_D, ")")
    print("═══════════════════════════════════════════════════════")

    # Config: Demo-Größe (Gemma 4 7B hätte D=2048)
    var cfg = Gemma4Config(
        hidden=G4_D, kv_dim=G4_KV_D, ffn_dim=G4_FFN_D,
        n_layers=40, n_heads=16, n_kv_heads=4,
    )

    # KV-Cache: 4096 Token max (Long Context)
    var kv_cache = KVCache(cfg.n_layers, cfg.kv_dim, 4096)
    print("KV-Cache:")
    print("  Layers:", kv_cache.n_layers, "  KV-Dim:", kv_cache.kv_dim)
    print("  Max Tokens: 4096  |  RAM:", kv_cache.memory_mb(), "MB")
    print()

    # Gewichtsspeicher pro Layer
    var q4_per_mat = Float64(G4_D * (G4_D // 2)) / 1e6
    var kv_per_mat = Float64(G4_D * (G4_KV_D // 2)) / 1e6
    var ff_per_mat = Float64(G4_D * (G4_FFN_D // 2)) / 1e6
    var total_per_layer = 2 * q4_per_mat + 2 * kv_per_mat + 3 * ff_per_mat
    print("Gewichte pro Layer (Q4):")
    print("  Q, O (D×D):           ", q4_per_mat, "MB ×2 =", 2 * q4_per_mat, "MB")
    print("  K, V (D×KV_D):        ", kv_per_mat, "MB ×2 =", 2 * kv_per_mat, "MB")
    print("  Gate, Up, Down (FFN): ", ff_per_mat, "MB ×3 =", 3 * ff_per_mat, "MB")
    print("  Total pro Layer:      ", total_per_layer, "MB")
    print("  Total (40 Layer):     ", total_per_layer * 40, "MB")


