#!/usr/bin/env python3
"""
MojoStream – SafeTensors Converter  (Gemma-4 / Gemma-2 kompatibel)

Konvertiert offizielle Gemma-4 SafeTensors in das .mojostream-Format:
  1. Multi-File SafeTensors → Layer-für-Layer (Double-Buffering, 1 Layer RAM)
  2. Symmetric INT4 mit block-weiser Skalierungs-Kalibrierung (block_size=32)
  3. Page-aligned Output (jeder Tensor startet auf 4096-Byte-Grenze)
  4. Seek-back Strategie: Skalenwerte werden nach Quantisierung rückgeschrieben

Voraussetzungen:
  pip install safetensors numpy

Nutzung:
  python3 scripts/convert.py <modell_verzeichnis> [ausgabe.mojostream]

Beispiel:
  python3 scripts/convert.py ~/models/gemma-4-27b  gemma4_27b.mojostream
  python3 scripts/convert.py ~/models/gemma-4-2b   gemma4_2b.mojostream
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

# ── SafeTensors Import ────────────────────────────────────────────────────────

def _check_safetensors():
    try:
        from safetensors import safe_open as _so   # noqa: F401
        return True
    except ImportError:
        print("ERROR: safetensors fehlt.")
        print("  pip install safetensors")
        sys.exit(1)

# ── Format-Konstanten (müssen mit mojostream.mojo übereinstimmen) ─────────────

MAGIC        = b"MOJOSTRM"
VERSION      = 1
FORMAT_TAG   = 1          # gemma4_q4_rowmajor
HEADER_BYTES = 128
DIR_E_BYTES  = 32
PAGE_ALIGN   = 4096

MS_PLE, MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN = range(8)

MAT_NAMES = {
    MS_Q: "Q", MS_K: "K", MS_V: "V", MS_O: "O",
    MS_GATE: "Gate", MS_UP: "Up", MS_DOWN: "Down",
}
MAT_ORDER = [MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN]

# ── HuggingFace → MojoStream Tensor-Name-Mapping ────────────────────────────
# Alle Gewichte werden transponiert: HF = (out_dim, in_dim) → unser Format = (in_dim, out_dim)
# da unsere Kernel C = A @ W nutzen (nicht C = A @ W.T wie PyTorch linear)

HF_SUFFIXES = {
    "self_attn.q_proj.weight": MS_Q,
    "self_attn.k_proj.weight": MS_K,
    "self_attn.v_proj.weight": MS_V,
    "self_attn.o_proj.weight": MS_O,
    "mlp.gate_proj.weight":    MS_GATE,
    "mlp.up_proj.weight":      MS_UP,
    "mlp.down_proj.weight":    MS_DOWN,
}

# Alternativen für verschiedene Gemma-Varianten
HF_SUFFIXES_ALT = {
    "self_attn.query_proj.weight": MS_Q,
    "self_attn.key_proj.weight":   MS_K,
    "self_attn.value_proj.weight": MS_V,
    "self_attn.out_proj.weight":   MS_O,
    "attention.q_proj.weight":     MS_Q,
    "attention.k_proj.weight":     MS_K,
    "attention.v_proj.weight":     MS_V,
    "attention.o_proj.weight":     MS_O,
}


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def align_up(n: int, align: int) -> int:
    return (n + align - 1) & ~(align - 1)


def load_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.json nicht gefunden in: {model_dir}")
    with open(path) as f:
        return json.load(f)


def extract_arch(config: dict) -> dict:
    """Extrahiert Architektur-Parameter aus config.json."""
    def get(keys, default=None):
        for k in (keys if isinstance(keys, list) else [keys]):
            if k in config:
                return config[k]
        return default

    hidden     = get("hidden_size")
    n_layers   = get("num_hidden_layers")
    n_heads    = get("num_attention_heads")
    n_kv_heads = get("num_key_value_heads", n_heads)
    ffn_dim    = get(["intermediate_size", "ffn_dim"])

    for name, val in [("hidden_size", hidden), ("num_hidden_layers", n_layers),
                      ("num_attention_heads", n_heads), ("intermediate_size", ffn_dim)]:
        if val is None:
            raise ValueError(f"Feld '{name}' fehlt in config.json")

    head_dim = hidden // n_heads
    kv_dim   = n_kv_heads * head_dim

    return dict(hidden=hidden, n_layers=n_layers, n_heads=n_heads,
                n_kv_heads=n_kv_heads, head_dim=head_dim,
                kv_dim=kv_dim, ffn_dim=ffn_dim)


def build_tensor_index(model_dir: str) -> dict:
    """
    Baut {tensor_name → file_path} Index.
    Unterstützt Einzel- und Sharded-Modelle.
    """
    from safetensors import safe_open

    # Sharded: model.safetensors.index.json
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        return {name: os.path.join(model_dir, fname)
                for name, fname in idx["weight_map"].items()}

    # Einzelne Datei
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        with safe_open(single, framework="numpy", device="cpu") as f:
            return {k: single for k in f.keys()}

    # Mehrere Dateien ohne Index
    files = sorted(p for p in os.listdir(model_dir) if p.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"Keine .safetensors-Dateien in: {model_dir}")

    index = {}
    for fname in files:
        fpath = os.path.join(model_dir, fname)
        with safe_open(fpath, framework="numpy", device="cpu") as f:
            for k in f.keys():
                index[k] = fpath
    return index


def load_weight(tensor_index: dict, name: str) -> np.ndarray:
    from safetensors import safe_open
    fpath = tensor_index[name]
    with safe_open(fpath, framework="numpy", device="cpu") as f:
        return np.array(f.get_tensor(name), dtype=np.float32)


def resolve_tensor_name(tensor_index: dict, layer: int, ms_type: int) -> str:
    """Findet den HuggingFace-Tensor-Namen für Layer/Typ."""
    prefix = f"model.layers.{layer}."

    # Primäre Namen
    for suffix, t in HF_SUFFIXES.items():
        if t == ms_type:
            name = prefix + suffix
            if name in tensor_index:
                return name

    # Alternative Namen
    for suffix, t in HF_SUFFIXES_ALT.items():
        if t == ms_type:
            name = prefix + suffix
            if name in tensor_index:
                return name

    available = sorted(k for k in tensor_index if k.startswith(prefix))
    raise KeyError(
        f"Tensor für Layer {layer}, Typ '{MAT_NAMES[ms_type]}' nicht gefunden.\n"
        f"  Verfügbar: {available[:10]}"
    )


# ── Quantisierung ────────────────────────────────────────────────────────────

def quantize_q4_blockwise(weight: np.ndarray, block_size: int = 32) -> tuple:
    """
    Symmetric INT4 Quantisierung mit block-weiser Skalierungs-Kalibrierung.

    Algorithmus:
      1. Flache Gewichte in Blöcke der Größe block_size teilen
      2. Pro Block: scale_i = max(|block_i|) / 7.0
      3. Globale Skala = max(scale_i) — garantiert kein Clipping
      4. q = clip(round(weight / scale), -8, 7)
      5. Packing: low nibble = gerade Spalten, high nibble = ungerade Spalten
         (identisch zu matmul_q4_bpack_raw Kernel-Erwartung)

    Format-Hinweis: .mojostream speichert eine Float32-Skala pro Tensor.
    True per-block Skalierung erfordert Format-Erweiterung (Roadmap).

    Returns: (packed_uint8 shape (rows*cols//2,), global_scale float32)
    """
    rows, cols = weight.shape
    assert cols % 2 == 0, f"cols muss gerade sein für Q4-Packing, erhalten: {cols}"

    flat = weight.ravel()

    # Block-weise Skalierungs-Kalibrierung
    block_scales = []
    for start in range(0, len(flat), block_size):
        block    = flat[start:start + block_size]
        max_abs  = float(np.abs(block).max())
        block_scales.append(max_abs / 7.0 if max_abs > 1e-8 else 1e-8)

    global_scale = float(max(block_scales)) if block_scales else 1e-8

    # Quantisierung mit globaler Skala
    q_f = np.clip(np.round(weight / global_scale), -8, 7).astype(np.int8)
    q_u = (q_f + 8).astype(np.uint8)   # Offset: -8..7 → 0..15

    # Bit-Packing: gerade Spalten → low nibble, ungerade → high nibble
    packed = np.zeros((rows, cols // 2), dtype=np.uint8)
    packed |= q_u[:, 0::2] & 0x0F          # low nibble
    packed |= (q_u[:, 1::2] & 0x0F) << 4   # high nibble

    return packed.reshape(-1), np.float32(global_scale)


# ── Offset-Berechnung (aus Architektur, ohne Daten zu laden) ─────────────────

def compute_layout(arch: dict) -> dict:
    """
    Berechnet alle Datei-Offsets aus der Architektur-Konfiguration.
    Kein Laden von Gewichten nötig — reine Dimensionsrechnung.
    """
    D, KVD, FFD = arch["hidden"], arch["kv_dim"], arch["ffn_dim"]
    n_layers     = arch["n_layers"]
    n_tensors    = n_layers * 8

    dir_bytes   = n_tensors * DIR_E_BYTES
    data_start  = align_up(HEADER_BYTES + dir_bytes, PAGE_ALIGN)

    # Tensor-Bytes pro Matrix-Typ
    mat_bytes = {
        MS_Q:    D * D   // 2,
        MS_K:    D * KVD // 2,
        MS_V:    D * KVD // 2,
        MS_O:    D * D   // 2,
        MS_GATE: D * FFD // 2,
        MS_UP:   D * FFD // 2,
        MS_DOWN: FFD * D // 2,
    }

    # Erwartete Dimensionen für TensorGuard-Konformität
    exp_dims = {
        MS_Q:    (D,   D),
        MS_K:    (D,   KVD),
        MS_V:    (D,   KVD),
        MS_O:    (D,   D),
        MS_GATE: (D,   FFD),
        MS_UP:   (D,   FFD),
        MS_DOWN: (FFD, D),
    }

    # Offsets für alle (layer, mat_type)-Paare
    offsets = {}   # (layer, mat_type) → data_offset
    cur     = data_start

    for layer in range(n_layers):
        for ms_type in MAT_ORDER:
            offsets[(layer, ms_type)] = cur
            cur = align_up(cur + mat_bytes[ms_type], PAGE_ALIGN)

    total_size = cur

    return dict(data_start=data_start, dir_bytes=dir_bytes,
                mat_bytes=mat_bytes, exp_dims=exp_dims,
                offsets=offsets, total_size=total_size)


# ── Datei-Writer (Seek-Back Strategie) ───────────────────────────────────────

def _dir_entry_offset(tensor_idx: int) -> int:
    """Absoluter Byte-Offset des Directory-Eintrags Nr. tensor_idx."""
    return HEADER_BYTES + tensor_idx * DIR_E_BYTES


def _scale_field_offset(tensor_idx: int) -> int:
    """Byte-Offset des Scale-Feldes (Byte 16 im Eintrag)."""
    return _dir_entry_offset(tensor_idx) + 16


def write_header_and_dir_placeholder(f, arch: dict, layout: dict):
    """Schreibt Header + Directory mit Platzhalter-Skalen (= 0.0)."""
    n_layers  = arch["n_layers"]
    n_tensors = n_layers * 8

    # Header (128 Byte)
    hdr = struct.pack("<8sIIIIIIIIQQQ",
        MAGIC, VERSION, FORMAT_TAG,
        n_layers,
        arch["hidden"], arch["kv_dim"], arch["ffn_dim"],
        arch["n_heads"], arch["n_kv_heads"],
        n_tensors,
        HEADER_BYTES,         # dir_offset
        layout["data_start"], # data_start
    )
    hdr += b"\x00" * (HEADER_BYTES - len(hdr))
    assert len(hdr) == HEADER_BYTES, len(hdr)
    f.write(hdr)

    # Directory: Platzhalter (rows/cols/offsets klar, scale=0.0 vorläufig)
    D, KVD, FFD = arch["hidden"], arch["kv_dim"], arch["ffn_dim"]
    exp_dims = layout["exp_dims"]

    for layer in range(n_layers):
        # PLE-Eintrag (kein Datenblock; Scale wird nach Laden rückgeschrieben)
        f.write(struct.pack("<IIIIfIQ", layer, MS_PLE, 1, 0, 0.0, 0, 0))

        for ms_type in MAT_ORDER:
            rows, cols  = exp_dims[ms_type]
            data_offset = layout["offsets"][(layer, ms_type)]
            f.write(struct.pack("<IIIIfIQ",
                layer, ms_type, rows, cols, 0.0, 0, data_offset))

    # Padding bis data_start
    pos = f.tell()
    assert pos == HEADER_BYTES + layout["dir_bytes"]
    f.write(b"\x00" * (layout["data_start"] - pos))


def write_scale_to_dir(f, tensor_global_idx: int, scale: float):
    """Schreibt die tatsächliche Skala an die richtige Stelle im Directory."""
    pos_after = f.tell()
    f.seek(_scale_field_offset(tensor_global_idx))
    f.write(struct.pack("<f", np.float32(scale)))
    f.seek(pos_after)


# ── Haupt-Konvertierungs-Funktion ────────────────────────────────────────────

def convert(model_dir: str, out_path: str, block_size: int = 32,
            verbose: bool = True):
    """Konvertiert SafeTensors → .mojostream (Layer-für-Layer, Double-Buffering)."""
    _check_safetensors()

    print("══════════════════════════════════════════════════════════════")
    print("  MojoStream SafeTensors Converter")
    print(f"  Quelle:  {model_dir}")
    print(f"  Ausgabe: {out_path}")
    print(f"  Q4 block_size={block_size}")
    print("══════════════════════════════════════════════════════════════")

    # [1] Konfiguration laden
    print("\n[1/4] Lade Modell-Konfiguration ...")
    config = load_config(model_dir)
    arch   = extract_arch(config)

    n_layers = arch["n_layers"]
    D, KVD, FFD = arch["hidden"], arch["kv_dim"], arch["ffn_dim"]
    print(f"  Typ:     {config.get('model_type', '?')}")
    print(f"  D={D}  KV={KVD}  FFN={FFD}  Heads={arch['n_heads']}/{arch['n_kv_heads']}")
    print(f"  Layers:  {n_layers}  head_dim={arch['head_dim']}")

    # [2] Tensor-Index aufbauen
    print("\n[2/4] Indexiere SafeTensors-Dateien ...")
    tensor_index = build_tensor_index(model_dir)
    print(f"  {len(tensor_index)} Tensoren indexiert")

    # [3] Layout vorausberechnen
    layout = compute_layout(arch)
    q4_gb  = sum(layout["mat_bytes"].values()) * n_layers / 1e9
    print(f"\n  Datei-Layout:")
    print(f"    data_start:  0x{layout['data_start']:X}  ({layout['data_start']} B)")
    print(f"    Q4-Gewichte: {q4_gb:.2f} GB")
    print(f"    Gesamt (est.): {layout['total_size'] / 1e9:.2f} GB")

    # [4] Datei schreiben (Seek-Back Strategie)
    print(f"\n[3/4] Quantisiere und schreibe {n_layers} Layer ...")
    t_start = time.perf_counter()

    with open(out_path, "w+b") as f:
        write_header_and_dir_placeholder(f, arch, layout)

        for layer in range(n_layers):
            elapsed = time.perf_counter() - t_start
            eta     = (elapsed / max(layer, 1)) * (n_layers - layer)
            print(f"\n  Layer {layer:3d}/{n_layers}  "
                  f"({elapsed:.0f}s  ETA {eta:.0f}s)", flush=True)

            # PLE-Skala: Gemma-4 spezifisch — Proxy via input_layernorm.weight
            ple_scale = 1.0
            ple_name  = f"model.layers.{layer}.input_layernorm.weight"
            if ple_name in tensor_index:
                w_norm    = load_weight(tensor_index, ple_name)
                ple_scale = float(np.mean(np.abs(w_norm)))
                ple_scale = max(ple_scale, 0.001)  # Sanity: scale > 0

            # PLE Scale in Directory rückschreiben (Eintrag: layer*8 + 0)
            write_scale_to_dir(f, layer * 8 + 0, ple_scale)
            if verbose:
                print(f"    PLE scale = {ple_scale:.4f}")

            # 7 Gewichtsmatrizen: laden → transponieren → quantisieren → schreiben
            for mi, ms_type in enumerate(MAT_ORDER):
                name = resolve_tensor_name(tensor_index, layer, ms_type)
                w_fp32 = load_weight(tensor_index, name)

                # Transponieren: HF (out_dim, in_dim) → MojoStream (in_dim, out_dim)
                w_fp32 = w_fp32.T
                rows, cols = w_fp32.shape

                # Dimensionen-Check (früh scheitern)
                exp_r, exp_c = layout["exp_dims"][ms_type]
                if (rows, cols) != (exp_r, exp_c):
                    raise ValueError(
                        f"Layer {layer} {MAT_NAMES[ms_type]}: "
                        f"Shape ({rows},{cols}) erwartet ({exp_r},{exp_c}). "
                        f"Prüfe Modell-Config (hidden/kv_dim/ffn_dim)."
                    )

                # Quantisierung
                packed, scale = quantize_q4_blockwise(w_fp32, block_size)
                del w_fp32  # FP32 sofort freigeben (Double-Buffering)

                # Daten an korrekte Position schreiben
                expected_pos = layout["offsets"][(layer, ms_type)]
                actual_pos   = f.tell()
                assert actual_pos == expected_pos, (
                    f"[Alignment] Layer {layer} {MAT_NAMES[ms_type]}: "
                    f"pos={actual_pos} erwartet={expected_pos}"
                )

                raw = packed.tobytes()
                f.write(raw)

                # Padding auf nächste Seiten-Grenze
                pad = align_up(len(raw), PAGE_ALIGN) - len(raw)
                if pad:
                    f.write(b"\x00" * pad)

                # Skala in Directory rückschreiben (Seek-Back)
                write_scale_to_dir(f, layer * 8 + 1 + mi, float(scale))

                if verbose:
                    print(f"    {MAT_NAMES[ms_type]:4s}: ({rows}×{cols})"
                          f"  Q4 {len(raw)/1e6:.1f} MB  scale={scale:.4f}")

    # [5] Verifikation
    actual_size = os.path.getsize(out_path)
    dt          = time.perf_counter() - t_start

    print(f"\n[4/4] Fertig.")
    print(f"  Dateigröße:  {actual_size / 1e9:.2f} GB")
    print(f"  Dauer:       {dt:.1f} s  ({actual_size / dt / 1e6:.0f} MB/s)")
    print()
    print("══════════════════════════════════════════════════════════════")
    print(f"  Ausgabe: {out_path}")
    print(f"  Nächster Schritt: pixi run mojo validate_real.mojo {out_path}")
    print("══════════════════════════════════════════════════════════════")

    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MojoStream SafeTensors Converter — Gemma-4 / Gemma-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("model_dir",
                    help="Verzeichnis mit SafeTensors + config.json")
    ap.add_argument("output", nargs="?", default="model_converted.mojostream",
                    help="Ausgabedatei (default: model_converted.mojostream)")
    ap.add_argument("--block-size", type=int, default=32,
                    help="Blockgröße für Q4-Kalibrierung (default: 32)")
    ap.add_argument("--quiet", action="store_true",
                    help="Weniger Ausgabe")
    args = ap.parse_args()

    convert(
        model_dir  = args.model_dir,
        out_path   = args.output,
        block_size = args.block_size,
        verbose    = not args.quiet,
    )
