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

from src.linalg.kernels import Matrix, U8Ptr, PtrT, DT, SIMD_W, matmul_q4_bpack_raw
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


struct LMHead(Movable):
    """Q4-quantisierter Vocabulary-Projektor (embed_tokens.weight transponiert)."""
    var vocab_n:  Int          # Anzahl der Vocab-Einträge in diesem Proto
    var scale:    Float32      # globale Q4-Skala
    var bos_emb:  List[Float32]  # BOS-Embedding (Token-ID 2), FP32
    var weights:  List[UInt8]  # Q4 packed: (2560 × vocab_n / 2) Bytes

    fn __init__(out self):
        self.vocab_n = 0
        self.scale   = Float32(1.0)
        self.bos_emb = List[Float32]()
        self.weights = List[UInt8]()


fn load_lm_head(path: String) raises -> LMHead:
    """Lädt lm_head_proto.bin und gibt ein LMHead-Struct zurück."""
    var lm  = LMHead()
    var raw = List[UInt8]()
    with open(path, "r") as f:
        raw = f.read_bytes()

    var bp = rebind[UnsafePointer[UInt8, MutAnyOrigin]](raw.unsafe_ptr())
    var pos = 0

    lm.vocab_n = _read_u32_le(bp, pos);  pos += 4
    lm.scale   = _read_f32_le(bp, pos);  pos += 4

    # BOS-Embedding: 2560 × 4 Bytes FP32
    lm.bos_emb.resize(E4B_D, 0)
    var ep = rebind[UnsafePointer[Float32, MutAnyOrigin]](
        lm.bos_emb.unsafe_ptr()
    )
    for j in range(E4B_D):
        ep.store(j, _read_f32_le(bp, pos));  pos += 4

    # Q4-Gewichte: 2560 × vocab_n / 2 Bytes
    var data_size = E4B_D * lm.vocab_n // 2
    lm.weights.resize(data_size, 0)
    var dst = rebind[U8Ptr](lm.weights.unsafe_ptr())
    var src = bp + pos
    for i in range(data_size):
        dst.store(i, src.load(i))

    return lm^


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
    """
    Logit-Berechnung: x(BATCH×D) @ W_vocab_q4(D×vocab_n) → logits(BATCH×vocab_n).
    Nutzt matmul_q4_bpack_raw — identisch zu den Layer-Projektionen.
    SIMD-Dequantisierung via dequantize_block[width] on-the-fly im Kernel.
    """
    matmul_q4_bpack_raw(
        logits, x,
        rebind[U8Ptr](lm.weights.unsafe_ptr()),
        lm.vocab_n // 2,   # packed_cols = output_cols / 2
        lm.scale,
        workers,
    )


fn get_token_embedding(
    token_id: Int,
    lm:       LMHead,
    out_ptr:  PtrT,        # (E4B_D,) Ausgabe-Puffer
):
    """
    Dequantisiert die Embedding-Spalte für token_id aus dem Q4 LM-Head.
    Direkte Nutzung von dequantize_block-Logik: (nibble − 8) × scale.
    Ergebnis = embed_tokens.weight[token_id] in FP32.
    """
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
    logits:      Matrix,    # (BATCH, vocab_n) — Zeile 0 wird genutzt
    temperature: Float32,
) -> Int:
    """
    Wählt ein Token aus der Logit-Verteilung (Zeile 0).
    temperature <= 0.01 → Greedy (argmax).
    temperature  = 0.7  → Kreative Verteilung (Standard).
    temperature  = 1.0  → Unmodifizierte Verteilung.
    """
    var ptr   = logits.data()   # Zeile 0
    var n     = logits.cols

    if temperature <= Float32(0.01):
        # Greedy: argmax
        var best   = 0
        var best_v = ptr.load(0)
        for j in range(1, n):
            var v = ptr.load(j)
            if v > best_v: best_v = v; best = j
        return best

    # Temperature-Skalierung: logits / T
    var inv_t = Float32(1.0) / temperature
    var max_v = ptr.load(0)
    for j in range(1, n):
        if ptr.load(j) > max_v: max_v = ptr.load(j)

    # Numerisch stabiles Softmax: exp((logit - max) / T)
    var sum_e = Float32(0.0)
    for j in range(n):
        var e = fexp((ptr.load(j) - max_v) * inv_t)
        ptr.store(j, e)
        sum_e += e
    var inv_s = Float32(1.0) / sum_e
    for j in range(n):
        ptr.store(j, ptr.load(j) * inv_s)

    # CDF-Sampling (perf_counter_ns als Pseudo-Zufallsquelle)
    var seed   = Int(perf_counter_ns())
    var lcg    = (seed * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFF
    var target = Float32(lcg) / Float32(0x7FFFFFFF)

    var cdf = Float32(0.0)
    for j in range(n):
        cdf += ptr.load(j)
        if cdf >= target: return j
    return n - 1
