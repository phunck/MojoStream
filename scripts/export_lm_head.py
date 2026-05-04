#!/usr/bin/env python3
"""
Exportiert Proto-Vocab und FP32-LM-Head (v2) für Gemma-4 E4B.

Outputs:
  vocab_proto.bin    – Token-ID → ASCII-Text-Map (kompakt binär)
  lm_head_proto.bin  – v2: FP32-Gewichte + final_norm_gamma + BOS-Embedding

Format vocab_proto.bin:
  [4B: n_tokens (uint32)]
  [für jeden Token:]
    [4B: text_len (uint32)]
    [text_len Bytes: ASCII-Text]

Format lm_head_proto.bin (v2):
  [4B: vocab_n (uint32)]
  [4B: scale  (float32)]   = 0.0 (v2 nutzt FP32)
  [4B: version (uint32)]   = 2
  [HIDDEN × 4B: BOS-Embedding FP32 (Token-ID 2)]
  [HIDDEN × 4B: final_norm_gamma FP32 (model.language_model.norm.weight)]
  [HIDDEN × vocab_n × 4B: LM-Head FP32, Shape (HIDDEN, VOCAB_N)]

PLE-Projektion:
  model.language_model.per_layer_model_projection.weight existiert [10752, 2560]
  und model.language_model.layers.N.per_layer_projection.weight [2560, 256].
  Diese werden bereits im .mojostream-Format als PLE-Skalen gespeichert.
  Eine zusätzliche Matrix-Multiplikation vor dem LM-Head ist nicht vorgesehen
  (die Projektion ist layer-intern, nicht LM-Head-intern).

Nutzung:
  python3 scripts/export_lm_head.py
"""
import json, struct, sys, os
import numpy as np

TOKENIZER_PATH = "weights/gemma-4-e4b-raw/tokenizer.json"
SAFETENSORS    = "weights/gemma-4-e4b-raw/model.safetensors"
VOCAB_N        = 8192     # Erste N Token-IDs für LM-Head + Sampling
HIDDEN         = 2560
VOCAB_OUT      = "vocab_proto.bin"
LM_HEAD_OUT    = "lm_head_proto.bin"
LM_HEAD_V2     = 2

print("══════════════════════════════════════════════════════════════")
print("  LM-Head Export v2  –  Gemma-4 E4B (FP32 + final_norm_gamma)")
print(f"  Vocab: {VOCAB_N} Tokens   |  Hidden: {HIDDEN}")
print("══════════════════════════════════════════════════════════════")

# ── [1] Vocab-Map laden ──────────────────────────────────────────────────────

print("\n[1/5] Lade tokenizer.json ...")
with open(TOKENIZER_PATH) as f:
    tok_data = json.load(f)

vocab  = tok_data["model"]["vocab"]   # piece → id
id2raw = {v: k for k, v in vocab.items()}

for at in tok_data.get("added_tokens", []):
    id2raw[at["id"]] = at["content"]

def decode_piece(piece: str) -> str:
    if piece.startswith("<0x") and piece.endswith(">"):
        try:
            byte_val = int(piece[3:-1], 16)
            return chr(byte_val) if byte_val >= 32 else "?"
        except ValueError:
            return "?"
    piece = piece.replace("▁", " ")
    return piece.encode("ascii", errors="replace").decode("ascii")

# ── [2] Vocab-Datei schreiben ─────────────────────────────────────────────────

print("[2/5] Schreibe vocab_proto.bin ...")
entries = []
for token_id in range(VOCAB_N):
    raw  = id2raw.get(token_id, f"[UNK:{token_id}]")
    text = decode_piece(raw)
    entries.append(text)

with open(VOCAB_OUT, "wb") as f:
    f.write(struct.pack("<I", len(entries)))
    for text in entries:
        tb = text.encode("ascii", errors="replace")
        f.write(struct.pack("<I", len(tb)))
        f.write(tb)

print(f"  → {VOCAB_OUT}: {len(entries)} Tokens, "
      f"{os.path.getsize(VOCAB_OUT) / 1e3:.1f} KB")
for sample_id in [0, 1, 2, 3, 270, 603, 1003, 2000, 5012]:
    if sample_id < len(entries):
        print(f"    Token {sample_id:5d}: {repr(entries[sample_id])}")

# ── [3] Safetensors Header lesen ─────────────────────────────────────────────

print(f"\n[3/5] Lese Safetensors-Header ...")
with open(SAFETENSORS, "rb") as f:
    hdr_len  = struct.unpack("<Q", f.read(8))[0]
    hdr_json = f.read(hdr_len)
    data_base = 8 + hdr_len

hdr_meta = json.loads(hdr_json)

def read_tensor_bf16(key: str) -> np.ndarray:
    info      = hdr_meta[key]
    offset    = info["data_offsets"][0]
    shape     = info["shape"]
    n_elems   = 1
    for d in shape: n_elems *= d
    with open(SAFETENSORS, "rb") as f:
        f.seek(data_base + offset)
        raw = f.read(n_elems * 2)
    u16  = np.frombuffer(raw, dtype="<u2")
    u32  = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape).copy()

# ── [4] Tensoren laden ───────────────────────────────────────────────────────

# embed_tokens.weight: [vocab_size, HIDDEN] → wir lesen nur VOCAB_N Zeilen
print(f"[4/5] Lese embed_tokens.weight ({VOCAB_N} Zeilen) ...")
et_info    = hdr_meta["model.language_model.embed_tokens.weight"]
et_start   = et_info["data_offsets"][0]
et_shape   = et_info["shape"]
row_bytes  = HIDDEN * 2
read_bytes = VOCAB_N * row_bytes
print(f"  Tensor-Offset: {data_base + et_start:,} B | Lese: {read_bytes / 1e6:.1f} MB")

with open(SAFETENSORS, "rb") as f:
    f.seek(data_base + et_start)
    raw = f.read(read_bytes)
u16    = np.frombuffer(raw, dtype="<u2")
u32    = u16.astype(np.uint32) << 16
w_fp32 = u32.view(np.float32).reshape(VOCAB_N, HIDDEN).copy()
print(f"  Shape: {w_fp32.shape}  Min: {w_fp32.min():.4f}  Max: {w_fp32.max():.4f}")

bos_embedding = w_fp32[2].copy()
print(f"  BOS-Embedding (ID=2): norm={float(np.linalg.norm(bos_embedding)):.4f}")

# model.language_model.norm.weight: [HIDDEN] – final RMSNorm gamma
print("  Lese model.language_model.norm.weight ...")
norm_gamma = read_tensor_bf16("model.language_model.norm.weight")
print(f"  norm.weight: shape={norm_gamma.shape}  "
      f"min={norm_gamma.min():.4f}  max={norm_gamma.max():.4f}  "
      f"mean={norm_gamma.mean():.4f}")

# PLE-Projektion: existiert als model.language_model.per_layer_model_projection.weight
# Shape [10752, 2560] – layer-interne Projektion, KEIN extra Matmul vor LM-Head nötig
ple_key = "model.language_model.per_layer_model_projection.weight"
if ple_key in hdr_meta:
    ple_shape = hdr_meta[ple_key]["shape"]
    print(f"  PLE-Projektion gefunden: {ple_key} shape={ple_shape}")
    print("  → Layer-intern im .mojostream gespeichert, kein zusätzlicher LM-Head-Matmul.")
else:
    print("  Kein model.embed_tokens.projection → kein zusätzlicher Matmul vor LM-Head.")

# ── [5] LM-Head FP32 exportieren ─────────────────────────────────────────────

print(f"\n[5/5] Schreibe lm_head_proto.bin v2 (FP32) ...")
# Layout: (HIDDEN, VOCAB_N) für matmul_fp32_raw: x(BATCH×D) @ W(D×vocab_n)
w_t = w_fp32.T.astype(np.float32).copy()   # (2560, 8192)
print(f"  Transponiert: {w_t.shape}  ({w_t.nbytes / 1e6:.1f} MB FP32)")

with open(LM_HEAD_OUT, "wb") as f:
    f.write(struct.pack("<I", VOCAB_N))           # vocab_n
    f.write(struct.pack("<f", 0.0))               # scale = 0.0 (FP32 mode)
    f.write(struct.pack("<I", LM_HEAD_V2))        # version = 2
    f.write(bos_embedding.astype(np.float32).tobytes())    # BOS-Embedding
    f.write(norm_gamma.astype(np.float32).tobytes())       # final_norm_gamma
    f.write(w_t.tobytes())                                 # LM-Head FP32

final_size = os.path.getsize(LM_HEAD_OUT)
print(f"  → {LM_HEAD_OUT}: {final_size / 1e6:.1f} MB")
print()
print("══════════════════════════════════════════════════════════════")
print("  Export v2 abgeschlossen.")
print(f"  LM-Head: FP32 ({w_t.nbytes / 1e6:.0f} MB) statt Q4 (10 MB)")
print(f"  final_norm_gamma: model.language_model.norm.weight geladen")
print(f"  Nächster Schritt:  pixi run e4b-infer")
print("══════════════════════════════════════════════════════════════")
