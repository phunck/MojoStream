# src/inference/gemma4_e4b.mojo
#
# Gemma-4 E4B – Hybrid-Inferenz-Kernel
#
# Architektur (aus config.json):
#   D=2560, FFD=10240, 42 Layer, 8Q/2KV Heads (GQA)
#   Sliding-Attention (35 Layer): head_dim=256, window=512, rope_theta=1e4
#   Full-Attention    ( 7 Layer, Index (i+1)%6==0): head_dim=512,
#                        partial_rotary=128 Dims, rope_theta=1e6
#
# Aufgaben-Mapping:
#   Task 1  → dequantize_block[width] in src/linalg/kernels.mojo
#   Task 2  → compute_attention, HybridKVCache  (dieses File)
#   Task 3  → e4b_forward_layer, load_e4b_layer_ref  (dieses File)
#   Task 4  → gemma4_e4b_infer.mojo (Root-Entry-Point)
#
from std.math import sqrt as fsqrt, exp as fexp, log
from std.memory import UnsafePointer

from src.linalg.kernels import (
    Matrix, PtrT, U8Ptr, DT, SIMD_W,
    matmul_q4_bpack_raw,
    rmsnorm_inplace, ple_scale_inplace,
    swiglu_inplace, apply_rope_inplace,
)
from src.streaming.mojostream import (
    MojoStreamFile,
    MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN,
)

# ── E4B Architektur-Konstanten ────────────────────────────────────────────────

comptime E4B_D            : Int = 2560
comptime E4B_N_HEADS      : Int = 8
comptime E4B_N_KV_HEADS   : Int = 2
comptime E4B_KV_RATIO     : Int = 4      # N_HEADS // N_KV_HEADS
comptime E4B_FFD          : Int = 10240
comptime E4B_N_LAYERS     : Int = 42
comptime E4B_N_SLIDE      : Int = 35     # Anzahl sliding_attention Layer
comptime E4B_N_FULL       : Int = 7      # Anzahl full_attention Layer

# Sliding-Attention Dimensionen
comptime E4B_SLIDE_HD     : Int = 256
comptime E4B_SLIDE_Q_DIM  : Int = 2048   # 8 × 256
comptime E4B_SLIDE_KV_DIM : Int = 512    # 2 × 256
comptime E4B_SLIDE_WIN    : Int = 512    # Zirkuläre Fenstergröße

# Full-Attention Dimensionen
comptime E4B_FULL_HD      : Int = 512
comptime E4B_FULL_Q_DIM   : Int = 4096  # 8 × 512
comptime E4B_FULL_KV_DIM  : Int = 1024  # 2 × 512
comptime E4B_FULL_ROTARY  : Int = 128   # partial_rotary_factor=0.25: 512×0.25
comptime E4B_FULL_THETA   : Float32 = 1000000.0

comptime E4B_BATCH        : Int = 4     # Kernel-Constraint: Batch muss ≡ 0 (mod MR=4)


# ── Layer-Typ Hilfsfunktionen ─────────────────────────────────────────────────

@always_inline
fn is_full_attn(layer: Int) -> Bool:
    """Full-Attention-Layer bei Index (layer+1) % 6 == 0: 5,11,17,23,29,35,41."""
    return (layer + 1) % 6 == 0


@always_inline
fn slide_cache_idx(layer: Int) -> Int:
    """Sliding-KV-Cache-Index für Layer `layer` (nur für sliding Layer aufrufbar)."""
    return layer - (layer + 1) // 6


@always_inline
fn full_cache_idx(layer: Int) -> Int:
    """Full-KV-Cache-Index für Layer `layer` (nur für full Layer aufrufbar)."""
    return (layer + 1) // 6 - 1


# ── Hybrider KV-Cache ─────────────────────────────────────────────────────────
#
# slide_k/v: 35 Zirkular-Puffer (SLIDE_WIN × SLIDE_KV_DIM), write_pos modulo 512
# full_k/v:  7 lineare Puffer   (max_full_seq × FULL_KV_DIM), wächst mit seq_len

struct HybridKVCache(Movable):
    var slide_k:     List[List[Scalar[DT]]]   # [35] × (SLIDE_WIN × SLIDE_KV_DIM)
    var slide_v:     List[List[Scalar[DT]]]
    var full_k:      List[List[Scalar[DT]]]   # [7]  × (max_full_seq × FULL_KV_DIM)
    var full_v:      List[List[Scalar[DT]]]
    var seq_len:     Int    # globale Sequenzlänge (nächste Schreib-Position)
    var max_full_seq: Int

    fn __init__(out self, max_full_seq: Int):
        self.seq_len      = 0
        self.max_full_seq = max_full_seq
        self.slide_k      = List[List[Scalar[DT]]]()
        self.slide_v      = List[List[Scalar[DT]]]()
        self.full_k       = List[List[Scalar[DT]]]()
        self.full_v       = List[List[Scalar[DT]]]()

        var slide_sz = E4B_SLIDE_WIN * E4B_SLIDE_KV_DIM
        for _ in range(E4B_N_SLIDE):
            var ks = List[Scalar[DT]](); ks.resize(slide_sz, 0)
            var vs = List[Scalar[DT]](); vs.resize(slide_sz, 0)
            self.slide_k.append(ks^)
            self.slide_v.append(vs^)

        var full_sz = max_full_seq * E4B_FULL_KV_DIM
        for _ in range(E4B_N_FULL):
            var kf = List[Scalar[DT]](); kf.resize(full_sz, 0)
            var vf = List[Scalar[DT]](); vf.resize(full_sz, 0)
            self.full_k.append(kf^)
            self.full_v.append(vf^)

    fn memory_mb(self) -> Float64:
        var slide = Float64(E4B_N_SLIDE * E4B_SLIDE_WIN * E4B_SLIDE_KV_DIM * 2 * 4)
        var full  = Float64(E4B_N_FULL  * self.max_full_seq * E4B_FULL_KV_DIM * 2 * 4)
        return (slide + full) / 1e6


# ── Zero-Copy Layer-Gewicht-Referenz ─────────────────────────────────────────
#
# Zeiger direkt in das MojoStreamFile.raw-Puffer.
# Kein Kopieren – MojoStreamFile muss am Leben bleiben, solange E4BLayerRef
# in Benutzung ist.

struct E4BLayerRef(Copyable, Movable):
    var q_ptr: U8Ptr;  var q_pcols: Int;  var scale_Q:    Float32
    var k_ptr: U8Ptr;  var k_pcols: Int;  var scale_K:    Float32
    var v_ptr: U8Ptr;  var v_pcols: Int;  var scale_V:    Float32
    var o_ptr: U8Ptr;  var o_pcols: Int;  var scale_O:    Float32
    var g_ptr: U8Ptr;  var g_pcols: Int;  var scale_Gate: Float32
    var u_ptr: U8Ptr;  var u_pcols: Int;  var scale_Up:   Float32
    var d_ptr: U8Ptr;  var d_pcols: Int;  var scale_Down: Float32
    var ple_scale: Float32
    var is_full:  Bool
    var q_dim:    Int
    var kv_dim:   Int
    var head_dim: Int

    fn __init__(out self,
        q_ptr: U8Ptr, q_pcols: Int, scale_Q:    Float32,
        k_ptr: U8Ptr, k_pcols: Int, scale_K:    Float32,
        v_ptr: U8Ptr, v_pcols: Int, scale_V:    Float32,
        o_ptr: U8Ptr, o_pcols: Int, scale_O:    Float32,
        g_ptr: U8Ptr, g_pcols: Int, scale_Gate: Float32,
        u_ptr: U8Ptr, u_pcols: Int, scale_Up:   Float32,
        d_ptr: U8Ptr, d_pcols: Int, scale_Down: Float32,
        ple_scale: Float32, is_full: Bool,
        q_dim: Int, kv_dim: Int, head_dim: Int,
    ):
        self.q_ptr = q_ptr; self.q_pcols = q_pcols; self.scale_Q    = scale_Q
        self.k_ptr = k_ptr; self.k_pcols = k_pcols; self.scale_K    = scale_K
        self.v_ptr = v_ptr; self.v_pcols = v_pcols; self.scale_V    = scale_V
        self.o_ptr = o_ptr; self.o_pcols = o_pcols; self.scale_O    = scale_O
        self.g_ptr = g_ptr; self.g_pcols = g_pcols; self.scale_Gate = scale_Gate
        self.u_ptr = u_ptr; self.u_pcols = u_pcols; self.scale_Up   = scale_Up
        self.d_ptr = d_ptr; self.d_pcols = d_pcols; self.scale_Down = scale_Down
        self.ple_scale = ple_scale
        self.is_full  = is_full
        self.q_dim    = q_dim
        self.kv_dim   = kv_dim
        self.head_dim = head_dim

    fn copy(self) -> Self:
        return Self(
            self.q_ptr, self.q_pcols, self.scale_Q,
            self.k_ptr, self.k_pcols, self.scale_K,
            self.v_ptr, self.v_pcols, self.scale_V,
            self.o_ptr, self.o_pcols, self.scale_O,
            self.g_ptr, self.g_pcols, self.scale_Gate,
            self.u_ptr, self.u_pcols, self.scale_Up,
            self.d_ptr, self.d_pcols, self.scale_Down,
            self.ple_scale, self.is_full,
            self.q_dim, self.kv_dim, self.head_dim,
        )^


fn load_e4b_layer_ref(ms: MojoStreamFile, layer: Int) -> E4BLayerRef:
    """Gibt zero-copy Zeiger für Layer `layer` aus dem mojostream-Puffer zurück."""
    var full  = is_full_attn(layer)
    var q_d   = E4B_SLIDE_Q_DIM
    var kv_d  = E4B_SLIDE_KV_DIM
    var hd    = E4B_SLIDE_HD
    if full:
        q_d  = E4B_FULL_Q_DIM
        kv_d = E4B_FULL_KV_DIM
        hd   = E4B_FULL_HD

    var qt = ms.tensor_ptr(layer, MS_Q)
    var kt = ms.tensor_ptr(layer, MS_K)
    var vt = ms.tensor_ptr(layer, MS_V)
    var ot = ms.tensor_ptr(layer, MS_O)
    var gt = ms.tensor_ptr(layer, MS_GATE)
    var ut = ms.tensor_ptr(layer, MS_UP)
    var dt = ms.tensor_ptr(layer, MS_DOWN)

    return E4BLayerRef(
        qt.ptr, q_d   // 2, qt.scale,
        kt.ptr, kv_d  // 2, kt.scale,
        vt.ptr, kv_d  // 2, vt.scale,
        ot.ptr, E4B_D // 2, ot.scale,
        gt.ptr, E4B_FFD // 2, gt.scale,   # Gate: (D, FFD) packed_cols = FFD//2 ✓
        ut.ptr, E4B_FFD // 2, ut.scale,   # Up:   (D, FFD) packed_cols = FFD//2 ✓
        dt.ptr, E4B_D   // 2, dt.scale,   # Down: (FFD, D) packed_cols = D//2   ← war E4B_FFD//2 = falsch
        ms.ple_scale(layer), full,
        q_d, kv_d, hd,
    )^


# ── KV-Cache Schreiben ────────────────────────────────────────────────────────

fn write_kv_sliding(
    mut kv:  HybridKVCache,
    sidx:    Int,
    k_buf:   Matrix,
    v_buf:   Matrix,
    kv_d:    Int,
    base_pos: Int,
):
    """Schreibt K/V in den zirkulären Sliding-Window-Cache (slot = pos % WIN)."""
    for b in range(E4B_BATCH):
        var slot = (base_pos + b) % E4B_SLIDE_WIN
        var kptr = rebind[PtrT](kv.slide_k[sidx].unsafe_ptr()) + slot * kv_d
        var vptr = rebind[PtrT](kv.slide_v[sidx].unsafe_ptr()) + slot * kv_d
        var kb   = k_buf.data() + b * kv_d
        var vb   = v_buf.data() + b * kv_d
        for j in range(kv_d):
            kptr.store(j, kb.load(j))
            vptr.store(j, vb.load(j))


fn write_kv_full(
    mut kv:  HybridKVCache,
    fidx:    Int,
    k_buf:   Matrix,
    v_buf:   Matrix,
    kv_d:    Int,
    base_pos: Int,
):
    """Schreibt K/V linear in den Full-Attention-Cache (bis max_full_seq)."""
    for b in range(E4B_BATCH):
        var pos = base_pos + b
        if pos < kv.max_full_seq:
            var kptr = rebind[PtrT](kv.full_k[fidx].unsafe_ptr()) + pos * kv_d
            var vptr = rebind[PtrT](kv.full_v[fidx].unsafe_ptr()) + pos * kv_d
            var kb   = k_buf.data() + b * kv_d
            var vb   = v_buf.data() + b * kv_d
            for j in range(kv_d):
                kptr.store(j, kb.load(j))
                vptr.store(j, vb.load(j))


# ── Task 2: Hybride Attention ─────────────────────────────────────────────────
#
# Sliding: Zirkulärer KV-Cache, Fenster = 512 Tokens.
#          Token bei Abs-Position `t` liegt in Slot `t % SLIDE_WIN`.
# Full:    Linearer KV-Cache, volle Sequenzlänge bis max_full_seq.
# GQA:     kv_ratio = 4 → je 4 Query-Heads teilen einen KV-Head.

fn compute_sliding_attention(
    mut out:   Matrix,     # (BATCH × q_dim)  Ausgabe
    q:         Matrix,     # (BATCH × q_dim)  Query-Vektoren nach RoPE
    kd:        PtrT,       # K-Puffer: SLIDE_WIN × kv_dim
    vd:        PtrT,       # V-Puffer: SLIDE_WIN × kv_dim
    base_pos:  Int,
    q_dim:     Int,        # 2048
    kv_dim:    Int,        # 512
    head_dim:  Int,        # 256
    n_heads:   Int,        # 8
    n_kv_heads: Int,       # 2
):
    var kv_ratio = n_heads // n_kv_heads
    var scale    = Float32(1.0) / fsqrt(Float32(head_dim))

    # Scores-Puffer: max SLIDE_WIN Einträge
    var scores = List[Scalar[DT]]()
    scores.resize(E4B_SLIDE_WIN, 0)
    var sp = rebind[PtrT](scores.unsafe_ptr())

    for b in range(E4B_BATCH):
        var seq_len   = base_pos + b + 1
        var win_start = seq_len - E4B_SLIDE_WIN
        if win_start < 0: win_start = 0
        var attend_n  = seq_len - win_start   # ≤ SLIDE_WIN

        var q_base = q.data()   + b * q_dim
        var o_base = out.data() + b * q_dim

        for h_q in range(n_heads):
            var h_kv   = h_q // kv_ratio
            var q_head = q_base + h_q * head_dim

            # Q · K → scores (attend_n Positionen)
            for i in range(attend_n):
                var t      = win_start + i
                var slot   = t % E4B_SLIDE_WIN
                var k_head = kd + slot * kv_dim + h_kv * head_dim
                var dot    = SIMD[DT, SIMD_W](0)
                var j      = 0
                while j + SIMD_W <= head_dim:
                    dot += q_head.load[width=SIMD_W](j) * k_head.load[width=SIMD_W](j)
                    j   += SIMD_W
                var s = dot.reduce_add() * scale
                while j < head_dim:
                    s += q_head.load(j) * k_head.load(j) * scale
                    j += 1
                sp.store(i, s)

            # Numerisch stabiles Softmax
            var max_s = sp.load(0)
            for i in range(1, attend_n):
                var v = sp.load(i)
                if v > max_s: max_s = v
            var sum_e = Float32(0.0)
            for i in range(attend_n):
                var e = fexp(sp.load(i) - max_s)
                sp.store(i, e)
                sum_e += e
            var inv = Float32(1.0) / sum_e
            for i in range(attend_n):
                sp.store(i, sp.load(i) * inv)

            # Gewichtete V-Summe
            var o_head = o_base + h_q * head_dim
            for j in range(head_dim): o_head.store(j, Float32(0.0))
            for i in range(attend_n):
                var t      = win_start + i
                var slot   = t % E4B_SLIDE_WIN
                var v_head = vd + slot * kv_dim + h_kv * head_dim
                var prob   = sp.load(i)
                var j      = 0
                while j + SIMD_W <= head_dim:
                    o_head.store[width=SIMD_W](j,
                        o_head.load[width=SIMD_W](j) +
                        prob * v_head.load[width=SIMD_W](j))
                    j += SIMD_W
                while j < head_dim:
                    o_head.store(j, o_head.load(j) + prob * v_head.load(j))
                    j += 1


fn compute_full_attention(
    mut out:   Matrix,     # (BATCH × q_dim)
    q:         Matrix,     # (BATCH × q_dim)
    kd:        PtrT,       # K-Puffer: max_full_seq × kv_dim
    vd:        PtrT,       # V-Puffer: max_full_seq × kv_dim
    base_pos:  Int,
    max_seq:   Int,
    q_dim:     Int,        # 4096
    kv_dim:    Int,        # 1024
    head_dim:  Int,        # 512
    n_heads:   Int,        # 8
    n_kv_heads: Int,       # 2
):
    var kv_ratio = n_heads // n_kv_heads
    var scale    = Float32(1.0) / fsqrt(Float32(head_dim))

    var scores = List[Scalar[DT]]()
    scores.resize(max_seq, 0)
    var sp = rebind[PtrT](scores.unsafe_ptr())

    for b in range(E4B_BATCH):
        var seq_len = base_pos + b + 1
        if seq_len > max_seq: seq_len = max_seq

        var q_base = q.data()   + b * q_dim
        var o_base = out.data() + b * q_dim

        for h_q in range(n_heads):
            var h_kv   = h_q // kv_ratio
            var q_head = q_base + h_q * head_dim

            for t in range(seq_len):
                var k_head = kd + t * kv_dim + h_kv * head_dim
                var dot    = SIMD[DT, SIMD_W](0)
                var j      = 0
                while j + SIMD_W <= head_dim:
                    dot += q_head.load[width=SIMD_W](j) * k_head.load[width=SIMD_W](j)
                    j   += SIMD_W
                var s = dot.reduce_add() * scale
                while j < head_dim:
                    s += q_head.load(j) * k_head.load(j) * scale
                    j += 1
                sp.store(t, s)

            var max_s = sp.load(0)
            for t in range(1, seq_len):
                var v = sp.load(t)
                if v > max_s: max_s = v
            var sum_e = Float32(0.0)
            for t in range(seq_len):
                var e = fexp(sp.load(t) - max_s)
                sp.store(t, e)
                sum_e += e
            var inv = Float32(1.0) / sum_e
            for t in range(seq_len):
                sp.store(t, sp.load(t) * inv)

            var o_head = o_base + h_q * head_dim
            for j in range(head_dim): o_head.store(j, Float32(0.0))
            for t in range(seq_len):
                var v_head = vd + t * kv_dim + h_kv * head_dim
                var prob   = sp.load(t)
                var j      = 0
                while j + SIMD_W <= head_dim:
                    o_head.store[width=SIMD_W](j,
                        o_head.load[width=SIMD_W](j) +
                        prob * v_head.load[width=SIMD_W](j))
                    j += SIMD_W
                while j < head_dim:
                    o_head.store(j, o_head.load(j) + prob * v_head.load(j))
                    j += 1


fn compute_attention(
    mut out:  Matrix,
    q:        Matrix,
    mut kv:   HybridKVCache,
    layer:    Int,
    base_pos: Int,
    q_dim:    Int,
    kv_dim:   Int,
    head_dim: Int,
):
    """Dispatcht auf Sliding- oder Full-Attention je nach Layer-Typ."""
    if is_full_attn(layer):
        var fidx = full_cache_idx(layer)
        compute_full_attention(
            out, q,
            rebind[PtrT](kv.full_k[fidx].unsafe_ptr()),
            rebind[PtrT](kv.full_v[fidx].unsafe_ptr()),
            base_pos, kv.max_full_seq,
            q_dim, kv_dim, head_dim,
            E4B_N_HEADS, E4B_N_KV_HEADS,
        )
    else:
        var sidx = slide_cache_idx(layer)
        compute_sliding_attention(
            out, q,
            rebind[PtrT](kv.slide_k[sidx].unsafe_ptr()),
            rebind[PtrT](kv.slide_v[sidx].unsafe_ptr()),
            base_pos,
            q_dim, kv_dim, head_dim,
            E4B_N_HEADS, E4B_N_KV_HEADS,
        )


# ── Task 3: Vollständiger E4B Layer Forward-Pass ──────────────────────────────
#
# Ablauf:
#   1. PLE-Skala + Pre-RMSNorm
#   2. Q/K/V Projektionen (matmul_q4_bpack_raw, zero-copy Gewichte)
#   3. RoPE auf Q und K (sliding: voll; full: partial_rotary=128 Dims)
#   4. K/V in den Hybrid-Cache schreiben
#   5. compute_attention (Sliding oder Full, GQA 8/2)
#   6. O-Projektion + Residual
#   7. Post-Attention RMSNorm
#   8. SwiGLU FFN (Gate + Up → SwiGLU → Down) + Residual

fn e4b_forward_layer(
    mut x:    Matrix,        # (BATCH × D) Aktivierungen, in-place Update
    layer:    Int,
    w:        E4BLayerRef,   # zero-copy Zeiger in mojostream-Puffer
    mut kv:   HybridKVCache,
    base_pos: Int,
    workers:  Int,
):
    var D    = E4B_D
    var q_d  = w.q_dim
    var kv_d = w.kv_dim
    var hd   = w.head_dim
    var xd   = x.data()

    # ── 1. PLE + Pre-RMSNorm ─────────────────────────────────────────────
    ple_scale_inplace(xd, E4B_BATCH, D, w.ple_scale)
    rmsnorm_inplace(xd, E4B_BATCH, D)

    # ── 2. Q/K/V Projektionen ────────────────────────────────────────────
    # Matrizen werden fresh alloziert (zero-initialisiert) damit der
    # akkumulierende Kernel nicht auf Vorwerte aufläuft.
    var q_buf = Matrix(E4B_BATCH, q_d)
    var k_buf = Matrix(E4B_BATCH, kv_d)
    var v_buf = Matrix(E4B_BATCH, kv_d)

    matmul_q4_bpack_raw(q_buf, x, w.q_ptr, w.q_pcols, w.scale_Q,    workers)
    matmul_q4_bpack_raw(k_buf, x, w.k_ptr, w.k_pcols, w.scale_K,    workers)
    matmul_q4_bpack_raw(v_buf, x, w.v_ptr, w.v_pcols, w.scale_V,    workers)

    # ── 3. RoPE auf Q und K ──────────────────────────────────────────────
    if w.is_full:
        # Partial RoPE: nur die ersten E4B_FULL_ROTARY=128 Dims jedes Heads,
        # Theta=1e6 (proportional). Restliche 384 Dims unverändert.
        for b in range(E4B_BATCH):
            var q_b = q_buf.data() + b * q_d
            var k_b = k_buf.data() + b * kv_d
            for h in range(E4B_N_HEADS):
                apply_rope_inplace(q_b + h * hd, 1, E4B_FULL_ROTARY,
                                   base_pos + b, E4B_FULL_THETA)
            for h in range(E4B_N_KV_HEADS):
                apply_rope_inplace(k_b + h * hd, 1, E4B_FULL_ROTARY,
                                   base_pos + b, E4B_FULL_THETA)
    else:
        # Volle RoPE über alle head_dim=256 Dims, theta=10000 (Default)
        for b in range(E4B_BATCH):
            apply_rope_inplace(q_buf.data() + b * q_d,  E4B_N_HEADS,    hd, base_pos + b)
            apply_rope_inplace(k_buf.data() + b * kv_d, E4B_N_KV_HEADS, hd, base_pos + b)

    # ── 4. K/V in den Cache schreiben ────────────────────────────────────
    if w.is_full:
        write_kv_full(kv, full_cache_idx(layer), k_buf, v_buf, kv_d, base_pos)
    else:
        write_kv_sliding(kv, slide_cache_idx(layer), k_buf, v_buf, kv_d, base_pos)

    # ── 5. Hybride Attention ─────────────────────────────────────────────
    var attn_out = Matrix(E4B_BATCH, q_d)
    compute_attention(attn_out, q_buf, kv, layer, base_pos, q_d, kv_d, hd)

    # ── 6. O-Projektion + Residual ───────────────────────────────────────
    # O-Proj: (BATCH × q_dim) × (q_dim × D) → (BATCH × D)
    var proj_out = Matrix(E4B_BATCH, D)
    matmul_q4_bpack_raw(proj_out, attn_out, w.o_ptr, w.o_pcols, w.scale_O, workers)
    var pd = proj_out.data()
    for j in range(E4B_BATCH * D): xd.store(j, xd.load(j) + pd.load(j))

    # ── 7. Post-Attention RMSNorm ────────────────────────────────────────
    rmsnorm_inplace(xd, E4B_BATCH, D)

    # ── 8. SwiGLU FFN + Residual ─────────────────────────────────────────
    var gate_buf = Matrix(E4B_BATCH, E4B_FFD)
    var up_buf   = Matrix(E4B_BATCH, E4B_FFD)
    matmul_q4_bpack_raw(gate_buf, x, w.g_ptr, w.g_pcols, w.scale_Gate, workers)
    matmul_q4_bpack_raw(up_buf,   x, w.u_ptr, w.u_pcols, w.scale_Up,   workers)
    swiglu_inplace(gate_buf.data(), up_buf.data(), E4B_BATCH * E4B_FFD)

    # Down-Proj: (BATCH × FFD) × (FFD × D) → (BATCH × D)
    var ffn_out = Matrix(E4B_BATCH, D)
    matmul_q4_bpack_raw(ffn_out, up_buf, w.d_ptr, w.d_pcols, w.scale_Down, workers)
    var fd = ffn_out.data()
    for j in range(E4B_BATCH * D): xd.store(j, xd.load(j) + fd.load(j))

    kv.seq_len += E4B_BATCH
