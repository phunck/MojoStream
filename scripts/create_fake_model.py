#!/usr/bin/env python3
"""
MojoStream – Fake-Modell Generator

Unterstützte Formate:
  --format row-major      Standard Q4 (matmul_q4_bpack, 1 Matrix/Layer)
  --format pre-packed     Tile-Layout Q4 (matmul_q4_prepacked, 1 Matrix/Layer)
  --format gemma4         7-Matrizen-Struktur pro Layer (Gemma 4 kompatibel)
  --format mojostream     Einzelne .mojostream-Datei (page-aligned, mit Tensor-Verzeichnis)

.mojostream Dateiformat (binär, little-endian):
  HEADER  (128 Byte, fest):
    [8 B]  ASCII  "MOJOSTRM"
    [4 B]  uint32 Version (1)
    [4 B]  uint32 Format-Tag (1 = gemma4_q4_rowmajor)
    [4 B]  uint32 n_layers
    [4 B]  uint32 hidden_size
    [4 B]  uint32 kv_dim
    [4 B]  uint32 ffn_dim
    [4 B]  uint32 n_heads
    [4 B]  uint32 n_kv_heads
    [8 B]  uint64 n_tensors
    [8 B]  uint64 dir_offset  (= 128)
    [8 B]  uint64 data_start  (= erstes 4096-aligntes Byte nach dem Verzeichnis)
    [64 B] zeros  (reserviert)
  TENSOR-VERZEICHNIS  (n_tensors × 32 Byte):
    Pro Eintrag:
    [4 B]  uint32 layer_id
    [4 B]  uint32 mat_type  (0=PLE, 1=Q, 2=K, 3=V, 4=O, 5=Gate, 6=Up, 7=Down)
    [4 B]  uint32 rows
    [4 B]  uint32 cols  (logische Spaltenzahl)
    [4 B]  float32 scale  (Q4-Skala; für PLE = PLE-Skala)
    [4 B]  uint32 flags  (0 = row-major Q4)
    [8 B]  uint64 data_offset  (absoluter Dateioffset; 0 für PLE-Einträge)
  PADDING  (Nullbytes bis zur nächsten 4096-Byte-Grenze)
  TENSOR-DATEN  (jeder Tensor beginnt auf 4096-Byte-Grenze):
    rows × (cols // 2) Byte uint8 gepackte Q4-Gewichte
    gefolgt von Padding bis zur nächsten Seitengrenze

Gemma 4 Demo-Dimensionen (skalierbare Platzhalter):
  D       = hidden_size    (z.B. 4096 für 27B, 1024 für Demo)
  KV_D    = kv_dim         (GQA: n_kv_heads × head_dim; ca. D//4)
  FFN_D   = ffn_intermediate (SwiGLU: ca. 8/3 × D ≈ 2.67D; Demo: 2D)
"""
import argparse
import os
import struct
import time
import numpy as np

# Standard-Dimensionen (echte Gewichte: D=4096 für 27B, D=2048 für 7B)
D     = 4096   # für row-major/pre-packed Benchmark
N_LAYERS = 40
BK    = 128
NR    = 16
HALF_W = NR // 2

# Gemma-4 Demo-Dimensionen (kleiner für schnelle Generierung)
G4_D          = 1024   # hidden size (Demo; real Gemma 4: 4096 / 2048)
G4_KV_D       = 256    # KV-Dim (GQA: 8 KV-Heads × 128 = 1024; Demo: 4 × 64 = 256)
G4_FFN_D      = 2048   # FFN intermediate (Demo; real: ~8/3 × D)
G4_LAYERS     = 40
G4_N_HEADS    = 16
G4_N_KV_HEADS = 4

# .mojostream Konstanten
PAGE_SIZE           = 4096
HEADER_BYTES        = 128
DIR_ENTRY_BYTES     = 32
TENSORS_PER_LAYER   = 8   # 1 PLE + 7 Matrizen
# Matrix-Typ-Tags (müssen mit mojostream.mojo übereinstimmen)
MS_PLE, MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN = range(8)


def align_up(n: int, align: int) -> int:
    return (n + align - 1) & ~(align - 1)

# Feste Reihenfolge der 7 Gemma-4-Matrizen pro Layer
GEMMA4_MATRICES = [
    ("Q",    G4_D,     G4_D),      # Query Projektion
    ("K",    G4_D,     G4_KV_D),   # Key Projektion (GQA: kleiner als Q)
    ("V",    G4_D,     G4_KV_D),   # Value Projektion
    ("O",    G4_D,     G4_D),      # Output Projektion
    ("Gate", G4_D,     G4_FFN_D),  # SwiGLU Gate
    ("Up",   G4_D,     G4_FFN_D),  # SwiGLU Up
    ("Down", G4_FFN_D, G4_D),      # FFN Down (Down hat transponierte Dims)
]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def pack_row_major(rng, rows, cols):
    scale = float(rng.uniform(0.04, 0.15))
    data  = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    return scale, data.reshape(-1)


def pack_tiles(rng, rows, cols):
    """Pre-Tiled Layout: (n_kt, n_nt, BK, NR//2) → sequenzieller Kernel-Zugriff."""
    scale  = float(rng.uniform(0.04, 0.15))
    B_q4   = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    n_kt   = max(1, rows // BK)
    n_nt   = max(1, cols // NR)
    bk_eff = rows // n_kt
    nt_eff = cols // n_nt

    packed = np.empty((n_kt, n_nt, bk_eff, HALF_W), dtype=np.uint8)
    for ki in range(n_kt):
        for ni in range(n_nt):
            byte_start = ni * HALF_W
            packed[ki, ni] = B_q4[ki * bk_eff:(ki + 1) * bk_eff,
                                  byte_start:byte_start + HALF_W]
    return scale, packed.reshape(-1)


# ---------------------------------------------------------------------------
# Generatoren
# ---------------------------------------------------------------------------

def generate_standard(out_dir, fmt):
    """row-major / pre-packed: eine Matrix pro Layer."""
    os.makedirs(out_dir, exist_ok=True)
    rng   = np.random.default_rng(42)
    total = 0
    t0    = time.perf_counter()

    pack_fn = pack_tiles if fmt == "pre-packed" else pack_row_major
    print(f"Format: {fmt}  N={D}×{D}  {N_LAYERS} Layer")

    for i in range(N_LAYERS):
        scale, data = pack_fn(rng, D, D)
        path = os.path.join(out_dir, f"layer_{i}.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<f", scale))
            f.write(data.tobytes())
        total += os.path.getsize(path)

    dt = time.perf_counter() - t0
    print(f"Fertig: {total / 1e6:.1f} MB in {dt:.2f} s  ({total / dt / 1e6:.0f} MB/s)")


def generate_gemma4(out_dir):
    """7-Matrizen pro Layer im Gemma-4-Format."""
    os.makedirs(out_dir, exist_ok=True)
    rng       = np.random.default_rng(42)
    total     = 0
    t0        = time.perf_counter()
    mat_bytes = sum(rows * (cols // 2) for _, rows, cols in GEMMA4_MATRICES)

    print(f"Format: gemma4  Layers={G4_LAYERS}  D={G4_D}  KV={G4_KV_D}  FFN={G4_FFN_D}")
    print(f"  7 Matrizen/Layer: Q, K, V, O, Gate, Up, Down")
    print(f"  ~{(4 + 7 * 4 + mat_bytes) / 1e6:.2f} MB / Layer")

    for i in range(G4_LAYERS):
        # Per-Layer Embedding Skala (PLE) – Gemma 4 spezifisch
        ple_scale = float(rng.uniform(0.8, 1.2))  # nahe 1.0, layer-spezifisch

        path = os.path.join(out_dir, f"layer_{i}.bin")
        with open(path, "wb") as f:
            # Header: PLE-Skala
            f.write(struct.pack("<f", ple_scale))

            # 7 Matrizen in fixer Reihenfolge
            for name, rows, cols in GEMMA4_MATRICES:
                scale, data = pack_tiles(rng, rows, cols)
                f.write(struct.pack("<f", scale))
                f.write(data.tobytes())

        fsize = os.path.getsize(path)
        total += fsize
        if i < 3 or i == G4_LAYERS - 1:  # erste 3 + letzte ausgeben
            print(f"  layer_{i:02d}.bin  ple={ple_scale:.4f}  {fsize / 1e6:.2f} MB")

    dt = time.perf_counter() - t0
    print(f"Fertig: {total / 1e6:.1f} MB in {dt:.2f} s  ({total / dt / 1e6:.0f} MB/s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# .mojostream Generator – eine Datei, page-aligned, mit Tensor-Verzeichnis
# ---------------------------------------------------------------------------

def generate_mojostream(out_path: str):
    """Erzeugt eine einzige model.mojostream-Datei im page-aligned Format."""
    rng = np.random.default_rng(42)

    n_tensors = G4_LAYERS * TENSORS_PER_LAYER
    dir_size  = n_tensors * DIR_ENTRY_BYTES
    data_start = align_up(HEADER_BYTES + dir_size, PAGE_SIZE)

    # ── Phase 1: Verzeichnis + Daten vorberechnen ─────────────────────────
    entries   = []   # Dicts mit Metadaten
    tensor_data = [] # numpy arrays (None für PLE-Einträge)
    cur_offset = data_start

    for layer in range(G4_LAYERS):
        # PLE-Skalierung (nur im Verzeichnis, kein Datenblock)
        ple_val = float(rng.uniform(0.8, 1.2))
        entries.append(dict(layer=layer, mat_type=MS_PLE, rows=1, cols=0,
                            scale=ple_val, flags=0, data_offset=0))
        tensor_data.append(None)

        # 7 Gewichtsmatrizen im row-major Q4-Format (für matmul_q4_bpack_raw)
        for mat_type, (name, rows, cols) in enumerate(GEMMA4_MATRICES, start=1):
            scale, data = pack_row_major(rng, rows, cols)
            tensor_bytes = rows * (cols // 2)
            entries.append(dict(layer=layer, mat_type=mat_type, rows=rows, cols=cols,
                                scale=scale, flags=0, data_offset=cur_offset))
            tensor_data.append(data)
            # Alle Demo-Dimensionen sind natürlich page-aligned (Potenzen von 2 × 512)
            cur_offset = align_up(cur_offset + tensor_bytes, PAGE_SIZE)

    total_size  = cur_offset
    weight_mb   = (total_size - data_start) / 1e6
    dir_pad     = data_start - (HEADER_BYTES + dir_size)

    print("Format: mojostream  (Einzel-Datei, 4096-Byte-aligned)")
    print(f"  Layers={G4_LAYERS}  D={G4_D}  KV={G4_KV_D}  FFN={G4_FFN_D}")
    print(f"  Header={HEADER_BYTES} B  Verzeichnis={dir_size} B  "
          f"Padding={dir_pad} B  Daten={weight_mb:.1f} MB")
    print(f"  data_start=0x{data_start:X}  Gesamt={total_size / 1e6:.1f} MB")
    print(f"  Schreibe {out_path} ...")

    t0 = time.perf_counter()

    with open(out_path, "wb") as f:
        # ── HEADER (128 Byte) ──────────────────────────────────────────────
        hdr = struct.pack("<8sIIIIIIIIQQQ",
            b"MOJOSTRM",
            1,               # version
            1,               # format_tag: gemma4_q4_rowmajor
            G4_LAYERS,
            G4_D, G4_KV_D, G4_FFN_D,
            G4_N_HEADS, G4_N_KV_HEADS,
            n_tensors,
            HEADER_BYTES,    # dir_offset
            data_start,      # data_start
        )
        hdr += b"\x00" * (HEADER_BYTES - len(hdr))
        assert len(hdr) == HEADER_BYTES
        f.write(hdr)

        # ── TENSOR-VERZEICHNIS (n_tensors × 32 Byte) ──────────────────────
        for e in entries:
            entry_bytes = struct.pack("<IIIIfIQ",
                e["layer"], e["mat_type"], e["rows"], e["cols"],
                e["scale"], e["flags"], e["data_offset"],
            )
            assert len(entry_bytes) == DIR_ENTRY_BYTES
            f.write(entry_bytes)

        # ── PADDING bis data_start ─────────────────────────────────────────
        pos = f.tell()
        assert pos == HEADER_BYTES + dir_size
        f.write(b"\x00" * (data_start - pos))

        # ── TENSOR-DATEN (jeder Block auf 4096-Byte-Grenze) ───────────────
        for e, data in zip(entries, tensor_data):
            if data is None:           # PLE-Eintrag → kein Datenblock
                continue
            pos = f.tell()
            assert pos == e["data_offset"], \
                f"Alignment-Fehler layer={e['layer']} mat={e['mat_type']}: " \
                f"pos={pos} erwartet={e['data_offset']}"
            raw = data.tobytes()
            f.write(raw)
            # Padding zur nächsten Seitengrenze
            pad = align_up(len(raw), PAGE_SIZE) - len(raw)
            if pad:
                f.write(b"\x00" * pad)

    dt   = time.perf_counter() - t0
    size = os.path.getsize(out_path)
    mb_s = size / dt / 1e6
    print(f"  Fertig: {size / 1e6:.1f} MB in {dt:.2f} s  ({mb_s:.0f} MB/s)")
    print(f"  Alignment-Check bestanden: alle Tensoren auf 4096-Byte-Grenzen ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="MojoStream Fake-Modell Generator")
    ap.add_argument("--format",
                    choices=["row-major", "pre-packed", "gemma4", "mojostream"],
                    default="mojostream",
                    help="Datei-Layout (default: mojostream)")
    ap.add_argument("--out", default=None,
                    help="Ausgabedatei/-verzeichnis (default: auto)")
    args = ap.parse_args()

    if args.format == "mojostream":
        out = args.out or os.path.join(here, "..", "model.mojostream")
        generate_mojostream(out)
    elif args.format == "gemma4":
        out = args.out or os.path.join(here, "..", "model_weights_gemma4")
        generate_gemma4(out)
    else:
        out = args.out or os.path.join(here, "..", f"model_weights_{args.format.replace('-', '_')}")
        generate_standard(out, args.format)
