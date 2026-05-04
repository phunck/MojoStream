# src/inference/lm_head.mojo
#
# LM-Head Projektion für Gemma-4 E4B.
#
# Nutzt matmul_q4_bpack_raw (gleicher Kernel wie die 7 Layer-Matrizen),
# um hidden_state (4×2560) × W_vocab_q4 (2560×vocab_n) → logits (4×vocab_n)
# zu berechnen. Temperature-Sampling über die Logit-Verteilung Row 0.
#
# Dateiformat lm_head_proto.bin (von scripts/export_lm_head.py):
#   [4B: vocab_n (uint32)]
#   [4B: scale (float32)]
#   [2560×4B: BOS-Embedding FP32]
#   [2560×vocab_n/2 B: Q4-Packed-Daten]
#
from std.math import exp as fexp
from std.time import perf_counter_ns

from src.linalg.kernels import (
    Matrix, U8Ptr, PtrT, DT, SIMD_W,
    matmul_q4_bpack_raw, matmul_fp32_raw, rmsnorm_weighted_inplace,
)
from src.inference.gemma4_e4b import E4B_D, E4B_BATCH


fn _read_u32_le(ptr: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int:
    return (Int(ptr.load(pos)) | Int(ptr.load(pos+1)) << 8
            | Int(ptr.load(pos+2)) << 16 | Int(ptr.load(pos+3)) << 24)


fn _read_f32_le(ptr: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Float32:
    # Bit-Reinterpretation via tmp-Puffer: uint8-Bytes → IEEE-754 float32
    var tmp = List[UInt8]()
    tmp.resize(4, 0)
    var tp = tmp.unsafe_ptr()
    for i in range(4): tp.store(i, ptr.load(pos + i))
    return rebind[UnsafePointer[Float32, MutAnyOrigin]](tp).load(0)


comptime LM_HEAD_V2: Int = 2   # Format-Version mit FP32-Gewichten + final_norm_gamma

struct LMHead(Movable):
    """LM-Head Projektor: Q4 (v1) oder FP32 (v2) + final RMSNorm gamma."""
    var vocab_n:          Int
    var scale:            Float32      # Q4-Skala (0.0 wenn use_fp32=True)
    var use_fp32:         Bool         # True → FP32 LM-Head statt Q4
    var bos_emb:          List[Float32]
    var final_norm_gamma: List[Float32]   # model.norm.weight [D]
    var weights:          List[UInt8]     # Q4 packed (v1): (D × vocab_n / 2) Bytes
    var weights_fp32:     List[Float32]   # FP32 (v2): (D × vocab_n) Floats

    fn __init__(out self):
        self.vocab_n          = 0
        self.scale            = Float32(1.0)
        self.use_fp32         = False
        self.bos_emb          = List[Float32]()
        self.final_norm_gamma = List[Float32]()
        self.weights          = List[UInt8]()
        self.weights_fp32     = List[Float32]()


fn load_lm_head(path: String) raises -> LMHead:
    """Lädt lm_head_proto.bin (v1=Q4 oder v2=FP32+final_norm_gamma)."""
    var lm  = LMHead()
    var raw = List[UInt8]()
    with open(path, "r") as f:
        raw = f.read_bytes()

    var bp  = rebind[UnsafePointer[UInt8, MutAnyOrigin]](raw.unsafe_ptr())
    var pos = 0

    lm.vocab_n = _read_u32_le(bp, pos);  pos += 4
    lm.scale   = _read_f32_le(bp, pos);  pos += 4

    # Auto-detect version: v2 hat nach scale ein uint32 == LM_HEAD_V2
    var v2_expected_size = 12 + E4B_D * 4 + E4B_D * 4 + E4B_D * lm.vocab_n * 4
    lm.use_fp32 = (len(raw) >= v2_expected_size)

    if lm.use_fp32:
        # V2: [version=2] [BOS FP32] [gamma FP32] [weights FP32]
        pos += 4   # version-Feld überspringen

    # BOS-Embedding: E4B_D × 4B FP32
    lm.bos_emb.resize(E4B_D, 0)
    var ep = rebind[UnsafePointer[Float32, MutAnyOrigin]](lm.bos_emb.unsafe_ptr())
    for j in range(E4B_D):
        ep.store(j, _read_f32_le(bp, pos));  pos += 4

    if lm.use_fp32:
        # V2: final_norm_gamma [E4B_D × 4B]
        lm.final_norm_gamma.resize(E4B_D, Float32(1.0))
        var gp = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            lm.final_norm_gamma.unsafe_ptr()
        )
        for j in range(E4B_D):
            gp.store(j, _read_f32_le(bp, pos));  pos += 4

        # V2: FP32 Gewichte [E4B_D × vocab_n]
        var n_fp32 = E4B_D * lm.vocab_n
        lm.weights_fp32.resize(n_fp32, 0)
        var wp = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            lm.weights_fp32.unsafe_ptr()
        )
        for j in range(n_fp32):
            wp.store(j, _read_f32_le(bp, pos));  pos += 4
    else:
        # V1: Q4-Gewichte [E4B_D × vocab_n / 2 Bytes]
        # Gamma = 1.0 (keine echten Gewichte in v1)
        lm.final_norm_gamma.resize(E4B_D, Float32(1.0))
        var data_size = E4B_D * lm.vocab_n // 2
        lm.weights.resize(data_size, 0)
        var dst = rebind[U8Ptr](lm.weights.unsafe_ptr())
        var src = bp + pos
        for i in range(data_size):
            dst.store(i, src.load(i))

    return lm^


fn apply_final_norm(mut x: Matrix, lm: LMHead):
    """Wendet model.norm.weight (final RMSNorm mit gelernten Gamma-Gewichten)
    auf den Hidden-State nach Layer 41 an, bevor er in den LM-Head geht."""
    var gp = rebind[UnsafePointer[Float32, MutAnyOrigin]](
        lm.final_norm_gamma.unsafe_ptr()
    )
    rmsnorm_weighted_inplace(
        x.data(),
        rebind[PtrT](gp),
        E4B_BATCH,
        E4B_D,
    )


fn apply_bos_embedding(mut x: Matrix, lm: LMHead):
    """Setzt Zeile 0 von x auf das echte BOS-Embedding (Token-ID 2)."""
    var xp = x.data()
    var ep = rebind[UnsafePointer[Float32, MutAnyOrigin]](
        lm.bos_emb.unsafe_ptr()
    )
    for j in range(E4B_D):
        xp.store(j, ep.load(j))
    # Zeilen 1-3: Null-Padding (Kernel-Constraint MR=4)
    for j in range(E4B_D, E4B_BATCH * E4B_D):
        xp.store(j, Float32(0.0))


fn project_lm_head(
    mut logits: Matrix,   # (BATCH, vocab_n) — muss vor dem Aufruf zero() sein
    x:          Matrix,   # (BATCH, D) hidden states
    lm:         LMHead,
    workers:    Int,
):
    """Logit-Berechnung: x(BATCH×D) @ W_vocab(D×vocab_n) → logits(BATCH×vocab_n).
    Dispatch: FP32 (v2) für höhere Präzision oder Q4 (v1) legacy."""
    if lm.use_fp32:
        matmul_fp32_raw(
            logits, x,
            rebind[PtrT](lm.weights_fp32.unsafe_ptr()),
            lm.vocab_n,
            workers,
        )
    else:
        matmul_q4_bpack_raw(
            logits, x,
            rebind[U8Ptr](lm.weights.unsafe_ptr()),
            lm.vocab_n // 2,
            lm.scale,
            workers,
        )


fn get_token_embedding(
    token_id: Int,
    lm:       LMHead,
    out_ptr:  PtrT,        # (E4B_D,) Ausgabe-Puffer
):
    """Extrahiert embed_tokens.weight[token_id] als FP32-Vektor.
    FP32 (v2): direkte Spalten-Kopie aus dem FP32-Gewichts-Puffer.
    Q4  (v1):  Dequantisierung via (nibble − 8) × scale."""
    if lm.use_fp32:
        # FP32-Layout: (D, vocab_n) row-major → Spalte token_id
        var fp32_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            lm.weights_fp32.unsafe_ptr()
        )
        for row in range(E4B_D):
            out_ptr.store(row, fp32_ptr.load(row * lm.vocab_n + token_id))
    else:
        var packed_cols = lm.vocab_n // 2
        var byte_col    = token_id // 2
        var is_hi       = (token_id % 2) == 1
        var wptr        = rebind[U8Ptr](lm.weights.unsafe_ptr())
        var sc          = lm.scale
        for row in range(E4B_D):
            var byte_val = Int(wptr.load(row * packed_cols + byte_col))
            var nibble   = (byte_val >> 4) & 0x0F if is_hi else byte_val & 0x0F
            out_ptr.store(row, Float32(nibble - 8) * sc)


fn temperature_sampling(
    logits:           Matrix,      # (BATCH, vocab_n) — Zeile 0 wird genutzt
    temperature:      Float32,
    generated:        List[Int],   # bereits generierte Token-IDs
    penalty:          Float32 = Float32(1.0),  # repetition_penalty (1.0 = kein Effekt)
) -> Int:
    """
    Wählt ein Token aus der Logit-Verteilung (Zeile 0).
    temperature <= 0.01 → Greedy (argmax).
    penalty > 1.0 → Bestraft Wiederholungen (Standard: 1.2).
    """
    var ptr = logits.data()   # Zeile 0
    var n   = logits.cols

    # Repetition Penalty: Logits für bereits gesehene Token abschwächen
    if penalty > Float32(1.0):
        for i in range(len(generated)):
            var tid = generated[i]
            if tid < n:
                var v = ptr.load(tid)
                # Positive Logits dividieren, negative multiplizieren
                if v > Float32(0.0):
                    ptr.store(tid, v / penalty)
                else:
                    ptr.store(tid, v * penalty)

    if temperature <= Float32(0.01):
        var best   = 0
        var best_v = ptr.load(0)
        for j in range(1, n):
            var v = ptr.load(j)
            if v > best_v: best_v = v; best = j
        return best

    var inv_t = Float32(1.0) / temperature
    var max_v = ptr.load(0)
    for j in range(1, n):
        if ptr.load(j) > max_v: max_v = ptr.load(j)

    var sum_e = Float32(0.0)
    for j in range(n):
        var e = fexp((ptr.load(j) - max_v) * inv_t)
        ptr.store(j, e)
        sum_e += e
    var inv_s = Float32(1.0) / sum_e
    for j in range(n):
        ptr.store(j, ptr.load(j) * inv_s)

    var seed   = Int(perf_counter_ns())
    var lcg    = (seed * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFF
    var target = Float32(lcg) / Float32(0x7FFFFFFF)

    var cdf = Float32(0.0)
    for j in range(n):
        cdf += ptr.load(j)
        if cdf >= target: return j
    return n - 1
