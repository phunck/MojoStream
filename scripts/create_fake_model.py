#!/usr/bin/env python3
"""
Erzeugt 40 synthetische LLM-Layer-Gewichte als Q4-Binärdateien.

Format pro Datei (model_weights/layer_N.bin):
  [4 Bytes] float32 scale (little-endian)
  [N * N/2 Bytes] uint8 gepackte 4-bit Gewichte
  Gesamtgröße bei N=4096: 4 + 4096*2048 = 8.000.004 Bytes ≈ 7.6 MiB
"""
import os
import struct
import sys
import time
import numpy as np

N          = 4096
N_LAYERS   = 40
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_weights")
PACKED_COLS = N // 2

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    total_bytes = 0

    print(f"Erzeuge {N_LAYERS} Layer-Dateien (N={N}×{N}, ~{(4 + N*PACKED_COLS)/1e6:.1f} MB/Datei)")
    t0 = time.perf_counter()

    for i in range(N_LAYERS):
        scale   = float(rng.uniform(0.04, 0.15))
        weights = rng.integers(0, 256, size=(N, PACKED_COLS), dtype=np.uint8)
        path    = os.path.join(OUT_DIR, f"layer_{i}.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<f", scale))
            f.write(weights.tobytes())
        total_bytes += os.path.getsize(path)
        print(f"  layer_{i:02d}.bin  scale={scale:.5f}  {os.path.getsize(path)/1e6:.1f} MB")

    dt = time.perf_counter() - t0
    print(f"\nFertig: {total_bytes/1e6:.1f} MB in {dt:.2f} s  "
          f"({total_bytes/dt/1e6:.0f} MB/s Schreibrate)")

if __name__ == "__main__":
    main()
