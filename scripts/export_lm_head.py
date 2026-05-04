#!/usr/bin/env python3
"""
Exportiert Proto-Vocab und Q4-quantisierten LM-Head für Gemma-4 E4B.

Outputs:
  vocab_proto.bin    – Token-ID → ASCII-Text-Map (kompakt binär)
  lm_head_proto.bin  – Q4-Gewichte für die ersten VOCAB_N Tokens +
                       BOS-Embedding (FP32, für korrekte TTFT-Init)

Format vocab_proto.bin:
  [4B: n_tokens (uint32)]
  [für jeden Token:]
    [4B: text_len (uint32)]
    [text_len Bytes: ASCII-Text]

Format lm_head_proto.bin:
  [4B: vocab_n (uint32)]
  [4B: scale (float32)]
  [2560 × 4B: BOS-Embedding FP32 (Token-ID 2)]
  [2560 × vocab_n / 2 Bytes: Q4-Packed-Daten]

Nutzung:
  python3 scripts/export_lm_head.py
"""
import json, struct, sys, os
import numpy as np

sys.path.insert(0, 'scripts')
from convert import quantize_q4_blockwise

TOKENIZER_PATH = "weights/gemma-4-e4b-raw/tokenizer.json"
SAFETENSORS    = "weights/gemma-4-e4b-raw/model.safetensors"
VOCAB_N        = 8192     # Erste N Token-IDs für LM-Head + Sampling
VOCAB_OUT      = "vocab_proto.bin"
LM_HEAD_OUT    = "lm_head_proto.bin"
HIDDEN         = 2560

print("══════════════════════════════════════════════════════════════")
print("  LM-Head Export  –  Gemma-4 E4B")
print(f"  Vocab: {VOCAB_N} Tokens   |  Hidden: {HIDDEN}")
print("══════════════════════════════════════════════════════════════")

# ── [1] Vocab-Map laden ──────────────────────────────────────────────────────

print("\n[1/4] Lade tokenizer.json ...")
with open(TOKENIZER_PATH) as f:
    tok_data = json.load(f)

vocab  = tok_data["model"]["vocab"]   # piece → id
id2raw = {v: k for k, v in vocab.items()}

# Spezial-Token aus added_tokens überschreiben
for at in tok_data.get("added_tokens", []):
    id2raw[at["id"]] = at["content"]

def decode_piece(piece: str) -> str:
    """Dekodiert SentencePiece-Piece zu ASCII (Sonderzeichen → '?')."""
    # Byte-Fallback: <0xNN> → chr(0xNN)
    if piece.startswith("<0x") and piece.endswith(">"):
        try:
            byte_val = int(piece[3:-1], 16)
            return chr(byte_val) if byte_val >= 32 else "?"
        except ValueError:
            return "?"
    # SentencePiece Leerzeichen-Marker
    piece = piece.replace("▁", " ")   # ▁ → Leerzeichen
    # Nicht-ASCII → '?'
    return piece.encode("ascii", errors="replace").decode("ascii")

# ── [2] Vocab-Datei schreiben ─────────────────────────────────────────────────

print("[2/4] Schreibe vocab_proto.bin ...")
entries = []
for token_id in range(VOCAB_N):
    raw   = id2raw.get(token_id, f"[UNK:{token_id}]")
    text  = decode_piece(raw)
    entries.append(text)

with open(VOCAB_OUT, "wb") as f:
    f.write(struct.pack("<I", len(entries)))
    for text in entries:
        tb = text.encode("ascii", errors="replace")
        f.write(struct.pack("<I", len(tb)))
        f.write(tb)

print(f"  → {VOCAB_OUT}: {len(entries)} Tokens, "
      f"{os.path.getsize(VOCAB_OUT) / 1e3:.1f} KB")
# Stichprobe
for sample_id in [0, 1, 2, 3, 270, 1000, 1003, 2000]:
    if sample_id < len(entries):
        print(f"    Token {sample_id:5d}: {repr(entries[sample_id])}")

# ── [3] embed_tokens.weight lesen (partial pread) ────────────────────────────

print(f"\n[3/4] Lese embed_tokens.weight (erste {VOCAB_N} Zeilen) ...")
with open(SAFETENSORS, "rb") as f:
    hdr_len   = struct.unpack("<Q", f.read(8))[0]
    hdr_json  = f.read(hdr_len)
    data_base = 8 + hdr_len   # absolute Byte-Offset wo Tensor-Daten beginnen

import json as json2
hdr_meta  = json2.loads(hdr_json)
et_info   = hdr_meta["model.language_model.embed_tokens.weight"]
et_start  = et_info["data_offsets"][0]
et_dtype  = et_info["dtype"]   # BF16
et_shape  = et_info["shape"]   # [262144, 2560]

row_bytes  = HIDDEN * 2        # BF16 = 2 Bytes/Element
read_bytes = VOCAB_N * row_bytes

print(f"  Tensor-Offset:  {data_base + et_start:,} Bytes in der Datei")
print(f"  Lese:           {read_bytes / 1e6:.1f} MB  "
      f"(von {HIDDEN * et_shape[0] * 2 / 1e9:.2f} GB gesamt)")

with open(SAFETENSORS, "rb") as f:
    f.seek(data_base + et_start)
    raw = f.read(read_bytes)

# BF16 → FP32
u16     = np.frombuffer(raw, dtype="<u2")
u32     = u16.astype(np.uint32) << 16
w_fp32  = u32.view(np.float32).reshape(VOCAB_N, HIDDEN).copy()
print(f"  Shape: {w_fp32.shape}  Min: {w_fp32.min():.4f}  Max: {w_fp32.max():.4f}")

# BOS-Embedding (Token-ID 2) für korrekte TTFT-Initialisierung
bos_embedding = w_fp32[2].copy()   # shape (2560,)
print(f"  BOS-Embedding (ID=2): norm={float(np.linalg.norm(bos_embedding)):.4f}")

# ── [4] LM-Head quantisieren + exportieren ────────────────────────────────────

print("\n[4/4] Quantisiere LM-Head (INT4) und schreibe lm_head_proto.bin ...")

# Transponieren: (VOCAB_N, HIDDEN) → (HIDDEN, VOCAB_N) für matmul_q4_bpack_raw
w_t = w_fp32.T.copy()   # (2560, 8192)
print(f"  Transponiert: {w_t.shape}")

packed, scale = quantize_q4_blockwise(w_t)
data_bytes = packed.tobytes()
print(f"  Q4 packed: {len(data_bytes) / 1e6:.2f} MB  Scale: {scale:.6f}")

with open(LM_HEAD_OUT, "wb") as f:
    # Header
    f.write(struct.pack("<I", VOCAB_N))           # vocab_n
    f.write(struct.pack("<f", float(scale)))       # global scale
    # BOS-Embedding FP32
    f.write(bos_embedding.astype(np.float32).tobytes())
    # Q4-Daten
    f.write(data_bytes)

final_size = os.path.getsize(LM_HEAD_OUT)
print(f"  → {LM_HEAD_OUT}: {final_size / 1e6:.2f} MB")
print()
print("══════════════════════════════════════════════════════════════")
print("  Export abgeschlossen.")
print(f"  Nächster Schritt:  pixi run e4b-infer")
print("══════════════════════════════════════════════════════════════")
