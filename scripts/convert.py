#!/usr/bin/env python3
"""
MojoStream – SafeTensors Converter  (Gemma-4 E4B / Gemma-2 kompatibel)

Konvertiert offizielle Gemma-4 SafeTensors in das .mojostream-Format:
  1. BF16 SafeTensors → Layer-für-Layer (Double-Buffering, 1 Layer RAM)
  2. Symmetric INT4 mit block-weiser Skalierungs-Kalibrierung (block_size=32)
  3. Page-aligned Output (jeder Tensor startet auf 4096-Byte-Grenze)
  4. Seek-back Strategie: Skalenwerte nach Quantisierung rückgeschrieben
  5. Engram-Header-Erweiterung: PLE-Scales und KV-Cache-Metadaten reserviert

Nutzung:
  python3 scripts/convert.py <modell_verzeichnis> [ausgabe.mojostream]

Beispiel:
  python3 scripts/convert.py weights/gemma-4-e4b-raw models/gemma4_e4b_q4.mojostream
"""
import json
import os
import struct
import sys
import time

import numpy as np

# ── Format-Konstanten (müssen mit mojostream.mojo übereinstimmen) ─────────────

MAGIC        = b"MOJOSTRM"
VERSION      = 1
FORMAT_TAG   = 2          # gemma4_q4_gqa_layertype (v2: Engram-Header)
HEADER_BYTES = 128        # unveränderlich – Mojo-Reader hängt davon ab
DIR_E_BYTES  = 32
PAGE_ALIGN   = 4096

# Engram-Extension sitzt in ungenutzten Bytes 68–127 des vorhandenen Headers
ENGRAM_MAGIC = b"ENGR"
ENGRAM_VER   = 1

MS_PLE, MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN = range(8)

MAT_NAMES = {
    MS_Q: "Q", MS_K: "K", MS_V: "V", MS_O: "O",
    MS_GATE: "Gate", MS_UP: "Up", MS_DOWN: "Down",
}
MAT_ORDER = [MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN]

# ── HuggingFace Tensor-Suffix-Mapping ────────────────────────────────────────
# Transposiert: HF = (out_dim, in_dim) → unser Format = (in_dim, out_dim)

HF_SUFFIXES = {
    "self_attn.q_proj.weight": MS_Q,
    "self_attn.k_proj.weight": MS_K,
    "self_attn.v_proj.weight": MS_V,
    "self_attn.o_proj.weight": MS_O,
    "mlp.gate_proj.weight":    MS_GATE,
    "mlp.up_proj.weight":      MS_UP,
    "mlp.down_proj.weight":    MS_DOWN,
}

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


# ── SafeTensors BF16-sicherer Reader ─────────────────────────────────────────

class SafetensorsReader:
    """Liest SafeTensors mit BF16→F32-Konvertierung ohne numpy-dtype-Probleme."""

    def __init__(self, fpath: str):
        self.fpath = fpath
        with open(fpath, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            hdr_json = f.read(hdr_len).decode("utf-8")
        self._meta = json.loads(hdr_json)
        self._data_base = 8 + hdr_len   # byte-Offset wo Tensor-Daten beginnen
        self._f = open(fpath, "rb")     # offen halten für sequenzielle Reads

    def keys(self):
        return [k for k in self._meta if k != "__metadata__"]

    def get_shape(self, name: str):
        return self._meta[name]["shape"]

    def load_f32(self, name: str) -> np.ndarray:
        info  = self._meta[name]
        dtype = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]

        self._f.seek(self._data_base + start)
        raw = self._f.read(end - start)

        if dtype == "BF16":
            # BF16 = obere 16 Bits von Float32 (gleiche Byte-Reihenfolge LE)
            u16 = np.frombuffer(raw, dtype="<u2")
            u32 = u16.astype(np.uint32) << 16
            return u32.view(np.float32).reshape(shape).copy()
        elif dtype == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        elif dtype == "F16":
            return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
        else:
            raise ValueError(f"Unbekannter Tensor-dtype: {dtype} ({name})")

    def close(self):
        self._f.close()

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass


_readers: dict = {}   # fpath → SafetensorsReader (global Cache)


def _reader(fpath: str) -> SafetensorsReader:
    if fpath not in _readers:
        _readers[fpath] = SafetensorsReader(fpath)
    return _readers[fpath]


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
    """Extrahiert Architektur-Parameter für Gemma-4 (text_config) und ältere Gemma-Modelle."""
    # Gemma-4 multimodal: text_config ist verschachtelt
    tc = config.get("text_config", config)

    def get(*keys, default=None):
        for k in keys:
            if k in tc:
                return tc[k]
        return default

    hidden       = get("hidden_size")
    n_layers     = get("num_hidden_layers")
    n_heads      = get("num_attention_heads")
    n_kv_heads   = get("num_key_value_heads", default=n_heads)
    ffn_dim      = get("intermediate_size", "ffn_dim")
    head_dim     = get("head_dim")
    global_hd    = get("global_head_dim", default=head_dim)
    layer_types  = get("layer_types", default=[])
    num_kv_sh    = get("num_kv_shared_layers", default=0)
    ple_dim      = get("hidden_size_per_layer_input", default=0)
    sliding_win  = get("sliding_window", default=0)

    for name, val in [("hidden_size", hidden), ("num_hidden_layers", n_layers),
                      ("num_attention_heads", n_heads), ("intermediate_size", ffn_dim)]:
        if val is None:
            raise ValueError(f"Feld '{name}' fehlt in config.json")

    if head_dim is None:
        head_dim = hidden // n_heads
    if global_hd is None:
        global_hd = head_dim

    # Für rückwärtskompatible Header-Felder
    sliding_kv_dim = n_kv_heads * head_dim
    full_kv_dim    = n_kv_heads * global_hd

    return dict(
        hidden=hidden, n_layers=n_layers, n_heads=n_heads,
        n_kv_heads=n_kv_heads, head_dim=head_dim, global_head_dim=global_hd,
        ffn_dim=ffn_dim, layer_types=layer_types, num_kv_shared_layers=num_kv_sh,
        kv_dim=sliding_kv_dim,       # Feld im Haupt-Header (Sliding-Default)
        sliding_kv_dim=sliding_kv_dim,
        full_kv_dim=full_kv_dim,
        ple_dim=ple_dim,
        sliding_window=sliding_win,
    )


def get_layer_dims(arch: dict, layer: int) -> dict:
    """Gibt dimensions-dict für einen Layer zurück (GQA-bewusst, layer-typ-abhängig)."""
    D       = arch["hidden"]
    FFD     = arch["ffn_dim"]
    ltypes  = arch.get("layer_types", [])
    ltype   = ltypes[layer] if layer < len(ltypes) else "sliding_attention"

    hd    = arch["global_head_dim"] if ltype == "full_attention" else arch["head_dim"]
    q_dim = arch["n_heads"]    * hd   # 8 * head_dim
    kv_d  = arch["n_kv_heads"] * hd   # 2 * head_dim  (GQA)

    mat_bytes = {
        MS_Q:    D * q_dim // 2,
        MS_K:    D * kv_d  // 2,
        MS_V:    D * kv_d  // 2,
        MS_O:    q_dim * D // 2,    # O-Proj: q_dim → D
        MS_GATE: D * FFD   // 2,
        MS_UP:   D * FFD   // 2,
        MS_DOWN: FFD * D   // 2,
    }
    exp_dims = {
        MS_Q:    (D,     q_dim),
        MS_K:    (D,     kv_d),
        MS_V:    (D,     kv_d),
        MS_O:    (q_dim, D),
        MS_GATE: (D,     FFD),
        MS_UP:   (D,     FFD),
        MS_DOWN: (FFD,   D),
    }
    return dict(layer_type=ltype, q_dim=q_dim, kv_dim=kv_d,
                mat_bytes=mat_bytes, exp_dims=exp_dims)


def build_tensor_index(model_dir: str) -> dict:
    """Baut {tensor_name → file_path} Index (Einzel- und Sharded-Modelle)."""
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
        return {k: single for k in _reader(single).keys()}

    # Mehrere Dateien ohne Index
    files = sorted(p for p in os.listdir(model_dir) if p.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"Keine .safetensors-Dateien in: {model_dir}")

    index = {}
    for fname in files:
        fpath = os.path.join(model_dir, fname)
        for k in _reader(fpath).keys():
            index[k] = fpath
    return index


def detect_model_prefix(tensor_index: dict) -> str:
    """Erkennt HF-Tensor-Präfix (Gemma-4 vs. ältere Gemma-Modelle)."""
    for name in tensor_index:
        if name.startswith("model.language_model.layers."):
            return "model.language_model.layers."
        if name.startswith("model.layers."):
            return "model.layers."
    raise ValueError("Kein bekanntes Tensor-Präfix in SafeTensors gefunden")


def load_weight(tensor_index: dict, name: str) -> np.ndarray:
    return _reader(tensor_index[name]).load_f32(name)


def resolve_tensor_name(tensor_index: dict, layer: int,
                        ms_type: int, prefix: str) -> str:
    full_prefix = f"{prefix}{layer}."

    for suffix, t in HF_SUFFIXES.items():
        if t == ms_type:
            name = full_prefix + suffix
            if name in tensor_index:
                return name

    for suffix, t in HF_SUFFIXES_ALT.items():
        if t == ms_type:
            name = full_prefix + suffix
            if name in tensor_index:
                return name

    available = sorted(k for k in tensor_index if k.startswith(full_prefix))
    raise KeyError(
        f"Tensor für Layer {layer}, Typ '{MAT_NAMES[ms_type]}' nicht gefunden.\n"
        f"  Präfix: {full_prefix}\n"
        f"  Verfügbar: {available[:10]}"
    )


# ── Quantisierung ────────────────────────────────────────────────────────────

def quantize_q4_blockwise(weight: np.ndarray, block_size: int = 32) -> tuple:
    """
    Symmetric INT4-Quantisierung mit block-weiser Skalierungs-Kalibrierung.

    Packing: low nibble = gerade Spalten, high nibble = ungerade Spalten.
    Returns: (packed_uint8 shape (rows*cols//2,), global_scale float32)
    """
    rows, cols = weight.shape
    assert cols % 2 == 0, f"cols muss gerade sein für Q4-Packing, erhalten: {cols}"

    flat = weight.ravel()

    block_scales = []
    for start in range(0, len(flat), block_size):
        block   = flat[start:start + block_size]
        max_abs = float(np.abs(block).max())
        block_scales.append(max_abs / 7.0 if max_abs > 1e-8 else 1e-8)

    global_scale = float(max(block_scales)) if block_scales else 1e-8

    q_f = np.clip(np.round(weight / global_scale), -8, 7).astype(np.int8)
    q_u = (q_f + 8).astype(np.uint8)   # -8..7 → 0..15

    packed = np.zeros((rows, cols // 2), dtype=np.uint8)
    packed |= q_u[:, 0::2] & 0x0F
    packed |= (q_u[:, 1::2] & 0x0F) << 4

    return packed.reshape(-1), np.float32(global_scale)


# ── Layout-Berechnung ─────────────────────────────────────────────────────────

def compute_layout(arch: dict) -> dict:
    """Berechnet alle Datei-Offsets aus der Architektur (layer-typ-bewusst)."""
    n_layers  = arch["n_layers"]
    n_tensors = n_layers * 8

    dir_bytes  = n_tensors * DIR_E_BYTES
    data_start = align_up(HEADER_BYTES + dir_bytes, PAGE_ALIGN)

    offsets         = {}
    layer_dims_map  = {}
    cur             = data_start

    for layer in range(n_layers):
        ldims = get_layer_dims(arch, layer)
        layer_dims_map[layer] = ldims
        for ms_type in MAT_ORDER:
            offsets[(layer, ms_type)] = cur
            cur = align_up(cur + ldims["mat_bytes"][ms_type], PAGE_ALIGN)

    return dict(
        data_start=data_start, dir_bytes=dir_bytes,
        layer_dims=layer_dims_map, offsets=offsets, total_size=cur,
    )


# ── Datei-Writer ──────────────────────────────────────────────────────────────

def _dir_entry_offset(tensor_idx: int) -> int:
    return HEADER_BYTES + tensor_idx * DIR_E_BYTES

def _scale_field_offset(tensor_idx: int) -> int:
    return _dir_entry_offset(tensor_idx) + 16


def write_header_and_dir_placeholder(f, arch: dict, layout: dict):
    """Schreibt Header (inkl. Engram-Extension) + Directory mit Platzhalter-Skalen."""
    n_layers  = arch["n_layers"]
    n_tensors = n_layers * 8

    # ── Haupt-Header (Bytes 0–67, 68 Byte) ──────────────────────────────────
    # <8sIIIIIIIIQQQ> = 8+9×4+3×8 = 68 Byte
    main_hdr = struct.pack("<8sIIIIIIIIQQQ",
        MAGIC, VERSION, FORMAT_TAG,
        n_layers,
        arch["hidden"], arch["kv_dim"], arch["ffn_dim"],
        arch["n_heads"], arch["n_kv_heads"],
        n_tensors,
        HEADER_BYTES,          # dir_offset
        layout["data_start"],  # data_start
    )

    # ── Engram-Extension (Bytes 64–127, 64 Byte reserviert) ─────────────────
    # main_hdr = 8s + 8×I + 3×Q = 8+32+24 = 64 Byte
    # Reserviert PLE-Scales und KV-Cache-Metadaten für Engram-Integration.
    # <4sIIIIII> = 4+7×4 = 32 Byte, + 32 Byte Future-Reserved
    engram_hdr = struct.pack("<4sIIIIII",
        ENGRAM_MAGIC,
        ENGRAM_VER,
        arch.get("num_kv_shared_layers", 0),   # Anzahl shared-KV-Layer
        arch.get("sliding_kv_dim", 0),          # KV-Dim für sliding layers
        arch.get("full_kv_dim",    0),           # KV-Dim für full-attention layers
        arch.get("ple_dim",        0),           # hidden_size_per_layer_input
        arch.get("sliding_window", 0),           # Sliding-Window-Größe
    )
    engram_hdr += b"\x00" * (HEADER_BYTES - len(main_hdr) - len(engram_hdr))

    hdr = main_hdr + engram_hdr
    assert len(hdr) == HEADER_BYTES, f"Header-Größe: {len(hdr)} ≠ {HEADER_BYTES}"
    f.write(hdr)

    # ── Directory (Platzhalter, Skalen = 0.0) ────────────────────────────────
    for layer in range(n_layers):
        ldims = layout["layer_dims"][layer]
        # PLE-Eintrag (kein Datenblock; scale wird nach Laden rückgeschrieben)
        f.write(struct.pack("<IIIIfIQ", layer, MS_PLE, 1, 0, 0.0, 0, 0))

        for ms_type in MAT_ORDER:
            rows, cols  = ldims["exp_dims"][ms_type]
            data_offset = layout["offsets"][(layer, ms_type)]
            f.write(struct.pack("<IIIIfIQ",
                layer, ms_type, rows, cols, 0.0, 0, data_offset))

    # Padding bis data_start
    pos = f.tell()
    assert pos == HEADER_BYTES + layout["dir_bytes"]
    f.write(b"\x00" * (layout["data_start"] - pos))


def write_scale_to_dir(f, tensor_global_idx: int, scale: float):
    pos_after = f.tell()
    f.seek(_scale_field_offset(tensor_global_idx))
    f.write(struct.pack("<f", np.float32(scale)))
    f.seek(pos_after)


# ── Haupt-Konvertierung ───────────────────────────────────────────────────────

def convert(model_dir: str, out_path: str, block_size: int = 32,
            verbose: bool = True):
    """Konvertiert SafeTensors → .mojostream (Layer-für-Layer, BF16-sicher)."""

    print("══════════════════════════════════════════════════════════════")
    print("  MojoStream SafeTensors Converter  (Gemma-4 E4B / GQA)")
    print(f"  Quelle:  {model_dir}")
    print(f"  Ausgabe: {out_path}")
    print(f"  Q4 block_size={block_size}")
    print("══════════════════════════════════════════════════════════════")

    # [1] Konfiguration
    print("\n[1/4] Lade Modell-Konfiguration ...")
    config = load_config(model_dir)
    arch   = extract_arch(config)

    n_layers = arch["n_layers"]
    D, KVD, FFD = arch["hidden"], arch["kv_dim"], arch["ffn_dim"]
    ltype_counts = {}
    for lt in arch["layer_types"]:
        ltype_counts[lt] = ltype_counts.get(lt, 0) + 1

    print(f"  Typ:      {config.get('model_type', '?')}")
    print(f"  D={D}  kv(slide)={KVD}  kv(full)={arch['full_kv_dim']}"
          f"  FFN={FFD}  Heads={arch['n_heads']}/{arch['n_kv_heads']}")
    print(f"  Layers:   {n_layers}  "
          + "  ".join(f"{lt}={cnt}" for lt, cnt in sorted(ltype_counts.items())))
    print(f"  PLE-Dim:  {arch['ple_dim']}  "
          f"KV-Shared: {arch['num_kv_shared_layers']}")

    # [2] Tensor-Index + Prefix erkennen
    print("\n[2/4] Indexiere SafeTensors-Dateien ...")
    tensor_index = build_tensor_index(model_dir)
    prefix       = detect_model_prefix(tensor_index)
    print(f"  {len(tensor_index)} Tensoren indexiert")
    print(f"  Tensor-Präfix: '{prefix}'")

    # [3] Layout vorausberechnen
    layout = compute_layout(arch)
    q4_total = sum(
        sum(layout["layer_dims"][l]["mat_bytes"].values())
        for l in range(n_layers)
    )
    print(f"\n  Datei-Layout:")
    print(f"    data_start:    0x{layout['data_start']:X}  ({layout['data_start']} B)")
    print(f"    Q4-Gewichte:   {q4_total / 1e9:.2f} GB")
    print(f"    Gesamt (est.): {layout['total_size'] / 1e9:.2f} GB")

    # Output-Verzeichnis anlegen
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # [4] Quantisierung + Schreiben
    print(f"\n[3/4] Quantisiere und schreibe {n_layers} Layer ...")
    t_start = time.perf_counter()

    with open(out_path, "w+b") as f:
        write_header_and_dir_placeholder(f, arch, layout)

        for layer in range(n_layers):
            elapsed = time.perf_counter() - t_start
            eta     = (elapsed / max(layer, 1)) * (n_layers - layer) if layer > 0 else 0
            ldims   = layout["layer_dims"][layer]
            print(f"\n  Layer {layer:3d}/{n_layers}  [{ldims['layer_type'][:7]}]  "
                  f"({elapsed:.0f}s  ETA {eta:.0f}s)", flush=True)

            # PLE-Skala: input_layernorm.weight als Proxy
            ple_scale = 1.0
            ple_name  = f"{prefix}{layer}.input_layernorm.weight"
            if ple_name in tensor_index:
                w_norm    = load_weight(tensor_index, ple_name)
                ple_scale = float(np.mean(np.abs(w_norm)))
                ple_scale = max(ple_scale, 0.001)

            write_scale_to_dir(f, layer * 8, ple_scale)
            if verbose:
                print(f"    PLE scale = {ple_scale:.4f}")

            # 7 Gewichtsmatrizen: laden → transponieren → quantisieren → schreiben
            for mi, ms_type in enumerate(MAT_ORDER):
                name   = resolve_tensor_name(tensor_index, layer, ms_type, prefix)
                w_fp32 = load_weight(tensor_index, name)

                # HF-Format: (out_dim, in_dim) → MojoStream: (in_dim, out_dim)
                w_fp32 = w_fp32.T
                rows, cols = w_fp32.shape

                exp_r, exp_c = ldims["exp_dims"][ms_type]
                if (rows, cols) != (exp_r, exp_c):
                    raise ValueError(
                        f"Layer {layer} {MAT_NAMES[ms_type]}: "
                        f"Shape ({rows},{cols}) erwartet ({exp_r},{exp_c}). "
                        f"Prüfe Architektur-Config."
                    )

                packed, scale = quantize_q4_blockwise(w_fp32, block_size)
                del w_fp32

                expected_pos = layout["offsets"][(layer, ms_type)]
                actual_pos   = f.tell()
                assert actual_pos == expected_pos, (
                    f"[Alignment] Layer {layer} {MAT_NAMES[ms_type]}: "
                    f"pos={actual_pos} erwartet={expected_pos}"
                )

                raw = packed.tobytes()
                f.write(raw)

                pad = align_up(len(raw), PAGE_ALIGN) - len(raw)
                if pad:
                    f.write(b"\x00" * pad)

                write_scale_to_dir(f, layer * 8 + 1 + mi, float(scale))

                if verbose:
                    print(f"    {MAT_NAMES[ms_type]:4s}: ({rows}×{cols})"
                          f"  Q4 {len(raw)/1e6:.1f} MB  scale={scale:.4f}")

    # [5] Abschluss
    actual_size = os.path.getsize(out_path)
    dt          = time.perf_counter() - t_start

    print(f"\n[4/4] Fertig.")
    print(f"  Dateigröße:    {actual_size / 1e9:.3f} GB  ({actual_size} B)")
    print(f"  Dauer:         {dt:.1f} s  ({actual_size / dt / 1e6:.0f} MB/s)")
    print()
    print("══════════════════════════════════════════════════════════════")
    print(f"  Ausgabe: {out_path}")
    print(f"  Nächster Schritt:")
    print(f"    pixi run mojo src/tests/validate_real.mojo {out_path}")
    print("══════════════════════════════════════════════════════════════")

    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="MojoStream SafeTensors Converter — Gemma-4 E4B / GQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("model_dir",
                    help="Verzeichnis mit SafeTensors + config.json")
    ap.add_argument("output", nargs="?",
                    default="models/gemma4_e4b_q4.mojostream",
                    help="Ausgabedatei (default: models/gemma4_e4b_q4.mojostream)")
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
