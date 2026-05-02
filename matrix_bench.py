#!/usr/bin/env python3
"""Reine-Python vs. NumPy Matrixmultiplikation – 1024x1024 (float32) + Q4-Bench."""
import os
import sys
import time

# Keranzahl VOR numpy-Import setzen damit BLAS/OpenBLAS die richtige Zahl erhält
_N_CORES = os.cpu_count() or 1
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_N_CORES))

import numpy as np

try:
    from threadpoolctl import threadpool_limits as _tpl
    _HAS_TPCTL = True
except ImportError:
    _HAS_TPCTL = False

N = 1024
N_Q = 1024
FLOPS = 2.0 * N * N * N


def matmul_pure_python(A, B, N):
    C = [[0.0] * N for _ in range(N)]
    for i in range(N):
        Ai = A[i]
        Ci = C[i]
        for k in range(N):
            aik = Ai[k]
            Bk = B[k]
            for j in range(N):
                Ci[j] += aik * Bk[j]
    return C


def bench_numpy(N):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((N, N), dtype=np.float32)
    B = rng.standard_normal((N, N), dtype=np.float32)
    _ = A @ B  # warm-up
    t0 = time.perf_counter()
    C = A @ B
    dt = time.perf_counter() - t0
    return dt, C[0, 0]


def bench_pure_python(N_small=128):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((N_small, N_small)).tolist()
    B = rng.standard_normal((N_small, N_small)).tolist()
    t0 = time.perf_counter()
    C = matmul_pure_python(A, B, N_small)
    dt = time.perf_counter() - t0
    flops = 2.0 * N_small ** 3
    return dt, flops, C[0][0]


def gflops(flops, seconds):
    return flops / seconds / 1e9


# ----------------------------------------------------------------------
# Q4 (4-bit) Quantisierung – symmetrisch, eine globale Skala pro Matrix
# Speicherung: uint8[K, N/2]; byte = (high_nibble << 4) | low_nibble
# Dequantisiert: w = (nibble - 8) * scale  ->  Wertebereich [-8..7] * scale
# ----------------------------------------------------------------------

def make_q4_data(N, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N), dtype=np.float32)
    B_f32 = rng.standard_normal((N, N), dtype=np.float32)
    scale = float(np.max(np.abs(B_f32)) / 7.0)
    q = np.clip(np.round(B_f32 / scale).astype(np.int32) + 8, 0, 15).astype(np.uint8)
    low  = q[:, 0::2]
    high = q[:, 1::2]
    B_q4 = (low | (high << 4)).astype(np.uint8)  # shape (N, N/2)
    return A, B_f32, B_q4, scale


def dequantize_full(B_q4, scale, N):
    """Expansion: uint8[K,N/2] -> float32[K,N]. Schreibt 4*N*N Bytes neu in den RAM."""
    low  = (B_q4 & 0x0F).astype(np.int32) - 8
    high = ((B_q4 >> 4) & 0x0F).astype(np.int32) - 8
    out = np.empty((N, N), dtype=np.float32)
    out[:, 0::2] = low.astype(np.float32) * scale
    out[:, 1::2] = high.astype(np.float32) * scale
    return out


def bench_numpy_q4_expansion(N):
    A, _, B_q4, scale = make_q4_data(N)
    # warm-up (Cache + CPU-Boost)
    _ = A @ dequantize_full(B_q4, scale, N)
    # Hauptmessung: Expansion + Matmul = ehrliche Gesamtlatenz
    t0 = time.perf_counter()
    B_full = dequantize_full(B_q4, scale, N)
    C = A @ B_full
    dt = time.perf_counter() - t0
    # Aufschluesselung
    t1 = time.perf_counter()
    _ = dequantize_full(B_q4, scale, N)
    dt_unpack = time.perf_counter() - t1
    t2 = time.perf_counter()
    _ = A @ B_full
    dt_matmul = time.perf_counter() - t2
    return dt, dt_unpack, dt_matmul, C, scale


def write_q4_artifacts(path_dir, N, A, B_q4, scale, C_ref):
    """Schreibt Eingabedaten + Referenz fuer den Mojo-Lauf."""
    A.astype(np.float32).tofile(os.path.join(path_dir, "A.bin"))
    B_q4.astype(np.uint8).tofile(os.path.join(path_dir, "B_q4.bin"))
    # nur ein paar Referenzwerte (erste/letzte Zelle der Output-Matrix)
    with open(os.path.join(path_dir, "meta.txt"), "w") as f:
        f.write(f"{N}\n{scale}\n{float(C_ref[0,0])}\n{float(C_ref[N-1,N-1])}\n")


if __name__ == "__main__":
    print(f"=== Python-Benchmarks (N={N} fuer NumPy, N=128 fuer Pure-Python) ===")

    dt_np, _ = bench_numpy(N)
    g_np = gflops(FLOPS, dt_np)
    print(f"NumPy        : {dt_np*1000:9.2f} ms   {g_np:8.2f} GFLOPS  (N={N})")

    dt_py, flops_py, _ = bench_pure_python(128)
    g_py = gflops(flops_py, dt_py)
    print(f"Pure Python  : {dt_py*1000:9.2f} ms   {g_py:8.4f} GFLOPS  (N=128)")

    scale = (N / 128) ** 3
    dt_py_extrapolated = dt_py * scale
    g_py_ex = gflops(FLOPS, dt_py_extrapolated)
    print(f"Pure Python* : {dt_py_extrapolated*1000:9.2f} ms   {g_py_ex:8.4f} GFLOPS  "
          f"(N={N}, hochgerechnet)")

    print(f"RESULT_NUMPY_MS={dt_np*1000:.4f}")
    print(f"RESULT_NUMPY_GFLOPS={g_np:.4f}")
    print(f"RESULT_PYTHON_MS={dt_py_extrapolated*1000:.4f}")
    print(f"RESULT_PYTHON_GFLOPS={g_py_ex:.6f}")

    # ---------- Q4 (4-bit Dequant) Benchmark ----------
    print()
    print(f"=== Q4-Benchmark (N={N_Q}) ===")
    A, _, B_q4, scale = make_q4_data(N_Q)
    print(f"Skala (max(|B|)/7) = {scale:.6f},  "
          f"B_packed = {B_q4.nbytes/1024:.1f} KiB,  "
          f"B_dequant = {B_q4.nbytes*8/1024:.1f} KiB (8x mehr im RAM)")
    # Multi-Thread (alle Kerne, Standard-Modus)
    dt_q, dt_unpack, dt_matmul, C_ref, _ = bench_numpy_q4_expansion(N_Q)
    flops_q = 2.0 * N_Q ** 3
    g_q = gflops(flops_q, dt_q)
    print(f"NumPy Q4 Expansion MT ({_N_CORES}T): {dt_q*1000:8.2f} ms   {g_q:7.2f} GFLOPS  "
          f"(unpack {dt_unpack*1000:.2f} ms, matmul {dt_matmul*1000:.2f} ms)")

    # Single-Thread
    if _HAS_TPCTL:
        with _tpl(limits=1):
            dt_1t, _, _, _, _ = bench_numpy_q4_expansion(N_Q)
        g_1t = gflops(flops_q, dt_1t)
        print(f"NumPy Q4 Expansion 1T:  {dt_1t*1000:8.2f} ms   {g_1t:7.2f} GFLOPS")
    else:
        dt_1t = dt_q  # fallback: kein threadpoolctl
        g_1t  = g_q
        print("NumPy 1T: threadpoolctl nicht verfuegbar, verwende MT-Zeit als Fallback")

    here = os.path.dirname(os.path.abspath(__file__)) or "."
    write_q4_artifacts(here, N_Q, A, B_q4, scale, C_ref)
    print(f"Artefakte fuer Mojo geschrieben in: {here}")

    print(f"RESULT_N_CORES={_N_CORES}")
    print(f"RESULT_Q4_NUMPY_MS={dt_q*1000:.4f}")
    print(f"RESULT_Q4_NUMPY_GFLOPS={g_q:.4f}")
    print(f"RESULT_Q4_NUMPY_1T_MS={dt_1t*1000:.4f}")
    print(f"RESULT_Q4_NUMPY_1T_GFLOPS={g_1t:.4f}")
    print(f"RESULT_Q4_UNPACK_MS={dt_unpack*1000:.4f}")
    print(f"RESULT_Q4_MATMUL_MS={dt_matmul*1000:.4f}")
