#!/usr/bin/env python3
"""
MojoStream Referenz-Generator – Layer-0 Numerical Fidelity

Liest model.mojostream, dequantisiert die Q4-Gewichte von Layer 0
und führt einen vollständigen Forward-Pass in NumPy durch.
Das Ergebnis (Input + Output) wird in layer0_ref.bin gespeichert
und dient als Ground-Truth für validate_layer.mojo.

Binärformat layer0_ref.bin:
  [4]  Magic 0x42464552 ("REFB")
  [4]  version = 1
  [4]  hidden_size
  [4]  batch = 4
  [batch*hidden*4]  input  float32  (row-major)
  [batch*hidden*4]  output float32  (row-major, nach vollem Layer-Forward)

Dequant-Formel (row-major Q4):
  W[k, 2j]   = (raw[k,j] & 0x0F - 8) * scale   (low  nibble → gerade Spalten)
  W[k, 2j+1] = (raw[k,j] >> 4    - 8) * scale   (high nibble → ungerade Spalten)
"""
import os, struct, sys
import numpy as np

REF_MAGIC = 0x42464552  # "REFB"


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def parse_mojostream(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()

    n_layers, hidden, kv_dim, ffn_dim, n_heads, n_kv_heads = struct.unpack_from("<IIIIII", data, 16)
    n_tensors, dir_offset, data_start = struct.unpack_from("<QQQ", data, 40)

    entries = []
    for i in range(n_tensors):
        base = dir_offset + i * 32
        layer_id, mat_type, rows, cols = struct.unpack_from("<IIII", data, base)
        scale  = struct.unpack_from("<f", data, base + 16)[0]
        doff   = struct.unpack_from("<Q", data, base + 24)[0]
        entries.append({"layer": layer_id, "type": mat_type,
                        "rows": rows, "cols": cols,
                        "scale": scale, "offset": doff})

    return {"hidden": hidden, "kv_dim": kv_dim, "ffn_dim": ffn_dim,
            "n_heads": n_heads, "n_kv_heads": n_kv_heads, "n_layers": n_layers,
            "data": data, "entries": entries}


def dequant_q4(ms: dict, layer: int, mat_type: int) -> object:
    """Dequantisiert eine Q4-Matrix → float32 ndarray (rows, cols)."""
    for e in ms["entries"]:
        if e["layer"] == layer and e["type"] == mat_type:
            if e["offset"] == 0:           # PLE-Eintrag → nur Skala
                return float(e["scale"])
            rows, cols, scale = e["rows"], e["cols"], e["scale"]
            size = rows * (cols // 2)
            raw  = np.frombuffer(ms["data"][e["offset"]:e["offset"] + size],
                                 dtype=np.uint8).reshape(rows, cols // 2)
            lo   = (raw & 0x0F).astype(np.float32) - 8.0
            hi   = ((raw >> 4) & 0x0F).astype(np.float32) - 8.0
            W    = np.empty((rows, cols), dtype=np.float32)
            W[:, 0::2] = lo * scale   # low  nibble → gerade Spalten
            W[:, 1::2] = hi * scale   # high nibble → ungerade Spalten
            return W
    raise KeyError(f"Tensor nicht gefunden: layer={layer} type={mat_type}")


# ── NumPy-Reimplementierung des Mojo-Forward-Passes ──────────────────────────

def rmsnorm_f32(x: np.ndarray) -> np.ndarray:
    """Per-Row RMSNorm in float32 (identisch zu rmsnorm_inplace in Mojo)."""
    rms = np.sqrt((x.astype(np.float64)**2).mean(axis=-1, keepdims=True) + 1e-6)
    return (x / rms).astype(np.float32)


def apply_rope_f32(x: np.ndarray, n_heads: int, head_dim: int, pos: int,
                   base: float = 10000.0) -> np.ndarray:
    """RoPE in float32 – exakte Nachbildung von apply_rope_inplace."""
    half     = head_dim // 2
    log_base = np.float32(np.log(float(base)))
    inv_freq = np.exp(-np.arange(half, dtype=np.float32) * 2.0 / head_dim * log_base)
    theta    = np.float32(pos) * inv_freq         # (half,) float32
    cos_t    = np.cos(theta).astype(np.float32)
    sin_t    = np.sin(theta).astype(np.float32)
    # x: (n_heads, head_dim) – Paare (x[i], x[i+half])
    x_lo  = x[:, :half].copy()
    x_hi  = x[:, half:].copy()
    out   = np.empty_like(x)
    out[:, :half] = x_lo * cos_t - x_hi * sin_t
    out[:, half:] = x_lo * sin_t + x_hi * cos_t
    return out.astype(np.float32)


def swiglu_f32(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU: up * silu(gate),  silu(x) = x * sigmoid(x)."""
    sig = (1.0 / (1.0 + np.exp(-gate.astype(np.float64)))).astype(np.float32)
    return (up * gate * sig).astype(np.float32)


def forward_layer(x_input: np.ndarray, weights: dict, cfg: dict,
                  base_pos: int = 0) -> dict:
    """
    Vollständiger Gemma-4-Layer Forward-Pass in NumPy.
    Gibt Zwischen- und Endergebnis zurück.

    Matmul-Orientierung: C = A @ W  (W hat Form (K, N) wie in matmul_q4_bpack_raw)
    """
    D, KVD, FFD = cfg["hidden"], cfg["kv_dim"], cfg["ffn_dim"]
    NH, NKV     = cfg["n_heads"], cfg["n_kv_heads"]
    HD          = D // NH
    kv_ratio    = NH // NKV
    batch       = x_input.shape[0]

    x = x_input.astype(np.float32).copy()

    # ── 1. PLE + Pre-Norm ────────────────────────────────────────────────────
    x = (x * np.float32(weights["ple_scale"])).astype(np.float32)
    x = rmsnorm_f32(x)

    # ── 2. QKV Projektionen ──────────────────────────────────────────────────
    Q = (x @ weights["Q"]).astype(np.float32)    # (batch, D)
    K = (x @ weights["K"]).astype(np.float32)    # (batch, KVD)
    V = (x @ weights["V"]).astype(np.float32)    # (batch, KVD)

    # ── 3. RoPE auf Q und K ──────────────────────────────────────────────────
    Q_h = Q.reshape(batch, NH,  HD)
    K_h = K.reshape(batch, NKV, HD)
    V_h = V.reshape(batch, NKV, HD)
    for b in range(batch):
        Q_h[b] = apply_rope_f32(Q_h[b], NH,  HD, base_pos + b)
        K_h[b] = apply_rope_f32(K_h[b], NKV, HD, base_pos + b)

    # ── 4. GQA Kausale Attention (base_pos=0: Token b sieht 0..b) ────────────
    scale    = np.float32(1.0 / np.sqrt(HD))
    attn_out = np.zeros((batch, NH, HD), dtype=np.float32)

    for b in range(batch):
        seq_len = base_pos + b + 1          # kausale Maske
        for h_q in range(NH):
            h_kv = h_q // kv_ratio
            q    = Q_h[b, h_q]              # (HD,)
            # Scores (alle gecachten Positionen 0..seq_len-1)
            scores = np.array(
                [np.dot(q, K_h[t, h_kv]) * scale for t in range(seq_len)],
                dtype=np.float32)
            # Numerisch stabiles Softmax
            scores -= scores.max()
            exps = np.exp(scores.astype(np.float64)).astype(np.float32)
            probs = (exps / exps.sum()).astype(np.float32)
            # Gewichtete V-Summe
            for t in range(seq_len):
                attn_out[b, h_q] += probs[t] * V_h[t, h_kv]

    attn_out = attn_out.reshape(batch, D).astype(np.float32)
    attn_ref = (x + (attn_out @ weights["O"]).astype(np.float32)).astype(np.float32)

    # ── 5. Output Projektion + Residual (auf normiertem x!) ──────────────────
    o_proj = (attn_out @ weights["O"]).astype(np.float32)
    x      = (x + o_proj).astype(np.float32)

    # ── 6. Post-Attention Norm ───────────────────────────────────────────────
    x = rmsnorm_f32(x)

    # ── 7. SwiGLU FFN + Residual ─────────────────────────────────────────────
    gate    = (x @ weights["Gate"]).astype(np.float32)
    up      = (x @ weights["Up"]).astype(np.float32)
    swiglu  = swiglu_f32(gate, up)
    ffn_out = (swiglu @ weights["Down"]).astype(np.float32)
    x       = (x + ffn_out).astype(np.float32)

    return {"attn_out": attn_ref, "final_out": x}


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

def main():
    here    = os.path.dirname(os.path.abspath(__file__))
    ms_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "model.mojostream")
    out_path= sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "layer0_ref.bin")

    print(f"Lese: {ms_path}")
    ms = parse_mojostream(ms_path)
    cfg = {k: ms[k] for k in ("hidden", "kv_dim", "ffn_dim", "n_heads", "n_kv_heads")}

    print(f"Modell: D={cfg['hidden']}  KV={cfg['kv_dim']}  FFN={cfg['ffn_dim']}")

    # Gewichte Layer 0
    print("Dequantisiere Layer-0-Gewichte ...")
    weights = {
        "ple_scale": dequant_q4(ms, 0, 0),
        "Q":         dequant_q4(ms, 0, 1),
        "K":         dequant_q4(ms, 0, 2),
        "V":         dequant_q4(ms, 0, 3),
        "O":         dequant_q4(ms, 0, 4),
        "Gate":      dequant_q4(ms, 0, 5),
        "Up":        dequant_q4(ms, 0, 6),
        "Down":      dequant_q4(ms, 0, 7),
    }
    print(f"  PLE-Skala: {weights['ple_scale']:.4f}")
    print(f"  W_Q: {weights['Q'].shape}  W_Down: {weights['Down'].shape}")

    # Fester Test-Input (Seed 1234 → identisch zu validate_layer.mojo)
    BATCH = 4
    rng   = np.random.default_rng(1234)
    x_in  = rng.standard_normal((BATCH, cfg["hidden"])).astype(np.float32) * 0.1
    print(f"\nTest-Input:  shape={x_in.shape}  mean={x_in.mean():.4f}  std={x_in.std():.4f}")

    # Referenz-Forward-Pass
    print("Führe NumPy Forward-Pass durch ...")
    result = forward_layer(x_in, weights, cfg, base_pos=0)
    x_out  = result["final_out"]
    print(f"Output:      shape={x_out.shape}  mean={x_out.mean():.4f}  std={x_out.std():.4f}")

    # Sanity-Check: RMS der Differenz zwischen Input und Output
    diff_rms = float(np.sqrt(((x_in - x_out)**2).mean()))
    print(f"Input↔Output RMSE (erwartet ≠ 0): {diff_rms:.6f}")

    # Schreibe Referenzdatei
    hidden = cfg["hidden"]
    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIII", REF_MAGIC, 1, hidden, BATCH))
        f.write(x_in.astype(np.float32).tobytes())
        f.write(x_out.astype(np.float32).tobytes())
    size = os.path.getsize(out_path)
    print(f"\nGespeichert: {out_path}  ({size} Byte)")

    # Ausgabe der ersten Werte für manuellen Abgleich
    print(f"\nRef-Output[0, :4]: {x_out[0, :4]}")
    print(f"Ref-Output[1, :4]: {x_out[1, :4]}")


if __name__ == "__main__":
    main()
