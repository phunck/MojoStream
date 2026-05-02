#!/usr/bin/env python3
"""
Erzeugt synthetische LLM-Gewichte als Q4-Binärdateien.

Zwei Formate wählbar über --format:
  row-major   (Standard, für matmul_q4_bpack)
  pre-packed  (Tile-Layout für matmul_q4_prepacked, gleiche Größe!)

Pre-Packed Tile-Layout (kein Stride-Zugriff im Kernel):
  Für jedes (kt_idx, nt_idx)-Tile: BK × (NR//2) Bytes hintereinander.
  tile_base = (kt_idx * n_nt + nt_idx) * BK * (NR//2)
  kl_byte   = tile_base + k_local * (NR//2)
  → Kernel lädt sequential, Hardware-Prefetcher arbeitet maximal effizient.

Dateiformat (beide Modi identisch):
  [4 Bytes]  float32 scale (little-endian)
  [K*N/2 B]  uint8 Gewichte (Anordnung je nach Format)
"""
import argparse
import os
import struct
import time
import numpy as np

N        = 4096
N_LAYERS = 40
BK       = 128    # muss mit kernels.mojo BK übereinstimmen
NR       = 16     # muss mit kernels.mojo NR übereinstimmen
HALF_W   = NR // 2  # = 8 Bytes pro k_local pro nt-Tile


def make_row_major(rng, K: int, N: int):
    """Standard row-major uint8[K, N//2]."""
    scale  = float(rng.uniform(0.04, 0.15))
    B_q4   = rng.integers(0, 256, size=(K, N // 2), dtype=np.uint8)
    return scale, B_q4.reshape(-1)


def make_pre_packed(rng, K: int, N: int):
    """Pre-Tiled: (n_kt, n_nt, BK, NR//2) → gleiche Bytes, anderes Layout."""
    scale  = float(rng.uniform(0.04, 0.15))
    B_q4   = rng.integers(0, 256, size=(K, N // 2), dtype=np.uint8)
    n_kt   = K // BK
    n_nt   = N // NR
    packed = np.empty((n_kt, n_nt, BK, HALF_W), dtype=np.uint8)
    for ki in range(n_kt):
        for ni in range(n_nt):
            byte_start = ni * HALF_W
            packed[ki, ni] = B_q4[ki * BK:(ki + 1) * BK,
                                  byte_start:byte_start + HALF_W]
    return scale, packed.reshape(-1)


def generate(out_dir: str, fmt: str):
    os.makedirs(out_dir, exist_ok=True)
    rng    = np.random.default_rng(42)
    total  = 0
    t0     = time.perf_counter()

    print(f"Format: {fmt}  N={N}  K={N}  {N_LAYERS} Layer  "
          f"(~{(4 + N * N // 2) / 1e6:.1f} MB/Layer)")

    for i in range(N_LAYERS):
        if fmt == "pre-packed":
            scale, data = make_pre_packed(rng, N, N)
        else:
            scale, data = make_row_major(rng, N, N)

        path = os.path.join(out_dir, f"layer_{i}.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<f", scale))
            f.write(data.tobytes())
        total += os.path.getsize(path)

    dt = time.perf_counter() - t0
    print(f"Fertig: {total / 1e6:.1f} MB in {dt:.2f} s  "
          f"({total / dt / 1e6:.0f} MB/s)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--format",  choices=["row-major", "pre-packed"],
                    default="pre-packed")
    ap.add_argument("--out",     default=os.path.join(here, "..", "model_weights_packed"))
    args = ap.parse_args()

    generate(args.out, args.format)
