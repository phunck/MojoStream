#!/usr/bin/env python3
"""
MojoStream – Fake-Modell Generator

Unterstützte Formate:
  --format row-major      Standard Q4 (matmul_q4_bpack, 1 Matrix/Layer)
  --format pre-packed     Tile-Layout Q4 (matmul_q4_prepacked, 1 Matrix/Layer)
  --format gemma4         7-Matrizen-Struktur pro Layer (Gemma 4 kompatibel)

Gemma 4 Layer-Struktur (pro Layer-Datei):
  [4 B] float32  PLE-Skalierung (Per-Layer Embedding)
  Dann 7 Matrizen in fixer Reihenfolge: Q, K, V, O, Gate, Up, Down
  Pro Matrix:
    [4 B] float32  Matrix-Skala (Q4-Quantisierung)
    [rows × cols//2 B] uint8  gepackte 4-bit Gewichte (pre-packed Tile-Layout)

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
G4_D     = 1024   # hidden size (Demo; real Gemma 4: 4096 / 2048)
G4_KV_D  = 256    # KV-Dim (GQA: 8 KV-Heads × 128 = 1024; Demo: 4 × 64 = 256)
G4_FFN_D = 2048   # FFN intermediate (Demo; real: ~8/3 × D)
G4_LAYERS = 40

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

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="MojoStream Fake-Modell Generator")
    ap.add_argument("--format", choices=["row-major", "pre-packed", "gemma4"],
                    default="gemma4",
                    help="Datei-Layout (default: gemma4)")
    ap.add_argument("--out", default=None,
                    help="Ausgabeverzeichnis (default: model_weights_<format>)")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(here, "..", f"model_weights_{args.format.replace('-', '_')}")

    if args.format == "gemma4":
        generate_gemma4(args.out)
    else:
        generate_standard(args.out, args.format)
