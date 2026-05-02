# MojoStream

> **MojoStream: High-performance LLM inference engine written in Mojo. Optimized for Layer-by-Layer Streaming — run 14B+ models on consumer hardware without loading the full model into RAM. Built for Gemma 4 (PLE/GQA) and zero-latency SATA/NVMe throughput.**

---

## Project Status

```
Branch: main  |  Mojo: 0.26.2  |  Target: x86-64 AVX2  |  License: Apache 2.0
```

| Component | Status | Last Verified |
|---|---|---|
| Q4 B-Pack Kernel (AVX2) | ✅ Stable | 2026-05 |
| Layer Streaming + Double-Buffer | ✅ Stable | 2026-05 |
| Multithreading Scaling (4T) | ✅ Stable | 2026-05 |
| `.mojostream` Format + ShapeGuard | ✅ Stable | 2026-05 |
| Gemma 4 Full Forward Pass (RoPE, GQA, SwiGLU) | ✅ Implemented | 2026-05 |
| BPE Tokenizer (vocab load + encode/decode) | ✅ Implemented | 2026-05 |
| Numerical Validation Layer 0 | ✅ **[PASS]** rel. RMSE 1.77e-6 | 2026-05 |
| P95 Benchmark Harness + JSON output | ✅ Stable | 2026-05 |
| Real Gemma 4 Weights (SafeTensors loader) | 📋 Planned | — |
| All 40 Layers Validated | 📋 Pending real weights | — |

---

## Performance at a Glance

Hardware reference: **Intel i7-7500U** (2 physical cores / 4 logical, AVX2, 16 GB DDR4, SATA SSD 470 MB/s)

### Kernel Throughput

| Kernel | N=1024 | N=4096 | Threads |
|---|---:|---:|---:|
| `matmul_q4_bpack` (B-Pack, L1) | **128 GFLOPS** | 83 GFLOPS | 4 |
| `matmul_q4_bpack_l2` (A-Panel, L2) | 121 GFLOPS | **105 GFLOPS** | 4 |
| `matmul_q4_prepacked` (tile-layout) | 115 GFLOPS | 79 GFLOPS | 4 |
| Naive Q4 fused | 12 GFLOPS | 9 GFLOPS | 4 |

> **12× speedup** over naive fused kernel; equals BLAS SGEMM on the same hardware for Q4 workloads.

### Gemma 4 End-to-End (Demo dims: D=1024, 40 layers)

| Mode | Latency P50 | P95 | t/s (batch=4) | t/s (single) |
|---|---:|---:|---:|---:|
| First Light (3 steps) | — | — | 24.6 | **6.2** |
| Harness 100 steps (seq grows) | 206 ms | 241 ms | 19.4 | 4.9 |

I/O pressure: **2.5%** — system is 97.5% compute-bound on SATA SSD.

### Numerical Validation

```
[PASS] Layer 0  –  relative RMSE: 1.77e-06  (threshold: 1e-04)
       Absolute RMSE: 0.022  |  RMS(ref): 12553
       → float32 SIMD blocking matches NumPy to < 2 ppm
```

---

## Architecture

```
MojoStream/
│
├── src/
│   ├── linalg/
│   │   └── kernels.mojo          Q4 matmul (BPack/Prepacked/L2), RoPE,
│   │                             RMSNorm, PLE, SwiGLU, SiLU
│   ├── streaming/
│   │   ├── loader.mojo           Double-buffer ModelRunner (SATA-optimal)
│   │   └── mojostream.mojo       .mojostream reader + ShapeGuard
│   ├── nlp/
│   │   └── tokenizer.mojo        BPE tokenizer (vocab load, encode, decode)
│   ├── bench/
│   │   └── harness.mojo          TTFT, P95, RAM estimate, I/O pressure, JSON
│   ├── tests/
│   │   └── validate_layer.mojo   Numerical fidelity check vs. NumPy reference
│   └── main.mojo                 Gemma 4 forward pass (full layer pipeline)
│
├── scripts/
│   ├── create_fake_model.py      Q4 weight generator (row-major / mojostream)
│   ├── create_vocab.py           Demo BPE vocabulary generator
│   └── gen_reference.py          NumPy reference output for validation
│
├── first_light.mojo              First-token benchmark entry point
├── bench.mojo                    P95 harness entry point
├── tokenizer_test.mojo           End-to-end tokenizer pipeline test
├── validate.mojo                 Layer-0 numerical validation entry point
└── pixi.toml                     Build environment (Mojo 0.26, c-compiler)
```

### Gemma 4 Layer Pipeline (per layer, batch=4)

```
Input (4, D)
  │
  ├─ PLE scaling  (per-layer scalar)
  ├─ RMSNorm (pre-norm, SIMD float32)
  │
  ├─── Q, K, V projections  (matmul_q4_bpack_raw, row-major Q4)
  ├─── RoPE  (apply_rope_inplace, cos/sin table + SIMD rotation)
  ├─── KV-Cache write  (causal: token b writes position base_pos+b)
  ├─── GQA Attention  (causal softmax, seq_len = base_pos+b+1 per token)
  ├─── Output projection + residual
  │
  ├─ RMSNorm (post-attention)
  │
  ├─── Gate, Up projections  (matmul_q4_bpack_raw)
  ├─── SwiGLU  (up * gate * sigmoid(gate), SIMD exp)
  └─── Down projection + residual
       │
Output (4, D)
```

---

## .mojostream File Format

Single-file model format replacing 40 loose `.bin` files.

```
HEADER   128 B    Magic "MOJOSTRM" + version + arch params (D, KV, FFN, heads)
DIR      n×32 B   Tensor directory: layer, type, dims, Q4-scale, offset
PADDING           Zero-fill to next 4096-byte boundary
DATA              Q4 tensors, each block starting at 4096-byte SSD page boundary
```

**Advantage over loose files:**
- 1 file open vs. 40 → eliminates seek overhead on SATA/NVMe
- Sequential read saturates disk bandwidth (DMA-friendly)
- ShapeGuard validates all dimensions on load before any inference

Generate demo model (D=1024, 40 layers, 178 MB):
```bash
python3 scripts/create_fake_model.py --format mojostream
```

---

## Quick Start

### Prerequisites

- Linux x86-64 with AVX2 (check: `grep avx2 /proc/cpuinfo`)
- Python ≥ 3.10 + NumPy
- [pixi](https://pixi.sh) package manager

### Setup

```bash
# Clone
git clone https://github.com/phunck/MojoStream.git
cd MojoStream

# Install Mojo 0.26 via pixi (downloads ~500 MB)
pixi install

# Verify
pixi run mojo --version   # should print: Mojo 0.26.2.0
```

### Generate Test Data

```bash
# Fake model weights (mojostream format, 178 MB)
python3 scripts/create_fake_model.py --format mojostream

# BPE vocabulary (114 tokens, 1.2 KB)
python3 scripts/create_vocab.py

# NumPy reference output for Layer-0 validation
python3 scripts/gen_reference.py
```

### Run Benchmarks

```bash
# First-Light: full 40-layer forward pass (3 steps, batch=4)
pixi run mojo first_light.mojo

# P95 Harness: 100-step latency distribution + JSON output
pixi run mojo bench.mojo
cat bench_result.json

# Tokenizer round-trip + forward pass demo
pixi run mojo tokenizer_test.mojo
```

### Run Validation

```bash
# Numerical fidelity: Mojo vs. NumPy reference (Layer 0)
python3 scripts/gen_reference.py   # generates layer0_ref.bin
pixi run mojo validate.mojo
# Expected: [PASS] Layer 0 - Error: ~1.8e-06
```

---

## Testing Strategy

MojoStream uses a three-tier validation approach:

| Tier | Tool | What it checks |
|---|---|---|
| **Kernel correctness** | `test_import.mojo` | Smoke test: 128×128 Q4 matmul compiles and runs |
| **Architecture correctness** | `tokenizer_test.mojo` | BPE encode→decode round-trip identity |
| **Numerical fidelity** | `validate.mojo` | Layer output matches NumPy reference to < 1e-4 relative RMSE |

Run all three:
```bash
pixi run mojo test_import.mojo    # smoke
pixi run mojo tokenizer_test.mojo # tokenizer
pixi run mojo validate.mojo       # numerical
```

### Extending to All 40 Layers

The current validation covers Layer 0 (the hardest to get right: first RoPE position, cold KV-cache). To unlock all 40:

```bash
# When real Gemma 4 weights are available:
python3 scripts/gen_reference.py /path/to/model.mojostream layer0_ref.bin --layer 0
pixi run mojo validate.mojo   # [PASS] → clear layer 1, 2, ...
```

The `validate_gemma4_shape()` ShapeGuard runs automatically on every load — wrong dimensions fail fast before any computation.

---

## Benchmark JSON Schema

`bench.mojo` outputs a JSON file for direct comparison with llama.cpp:

```json
{
  "format": "mojostream",
  "model": { "n_layers": 40, "hidden": 1024, "kv_dim": 256, "ffn_dim": 2048 },
  "file_mb": 178.3,
  "load_time_ms": 543.6,
  "ttft_ms": 956.0,
  "latency_p50_ms": 206.5,
  "latency_p95_ms": 241.3,
  "latency_p99_ms": 257.2,
  "latency_best_ms": 175.4,
  "peak_ram_mb": 398.5,
  "io_pressure_pct": 2.54,
  "tokens_per_sec_p50": 19.4,
  "tokens_per_sec_best": 22.8
}
```

---

## Development History

| Date | Milestone | Key Metric |
|---|---|---|
| 2026-04 | **Stable Q4 B-Pack Kernel** — MR=4 register-blocked, L1-tiled | 115 GFLOPS (N=1024) |
| 2026-04 | **A-Panel L2 Kernel** — stride elimination, MC=128 L2 blocking | 105 GFLOPS (N=4096) |
| 2026-04 | **Layer Streaming** — 40-layer inference, double-buffer I/O model | 5.6 t/s SATA |
| 2026-04 | **Gemma 4 Architecture** — PLE, KV-Cache, 7-matrix layout (Q/K/V/O/Gate/Up/Down) | structs + stubs |
| 2026-05 | **Gemma 4 Math Core** — RoPE, GQA causal attention, real SwiGLU | First Light: 6.2 t/s |
| 2026-05 | **.mojostream Format** — single-file, 4096-byte aligned, tensor directory | 1 read vs. 40 |
| 2026-05 | **P95 Benchmark Harness** — TTFT, latency distribution, I/O pressure, JSON | I/O: 2.5% |
| 2026-05 | **BPE Tokenizer** — binary vocab, encode/decode, round-trip verified | "Hallo Mojo" ✓ |
| 2026-05 | **Numerical Validation** — ShapeGuard + Layer-0 NumPy cross-check | **[PASS] 1.77e-6** |

---

## Technical Reference

### Kernel Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `SIMD_W` | 8 (AVX2) / 16 (AVX-512) | compile-time via `simd_width_of` |
| `MR` | 4 | micro-kernel row tiles (4 accumulators fit AVX2 registers) |
| `NR` | 16 | column tiles per B-tile (2 × SIMD_W) |
| `BK` | 128 | K-tile: 8 KB B-buffer fits L1d |
| `MC` | 128 | M-tile: 64 KB A-panel fits half of L2 (256 KB, 2 HT threads) |
| Q4 format | row-major | `B[k, n//2]`: low nibble = even col, high nibble = odd col |
| Dequant bias | 8 | nibble ∈ [0,15] → (nibble − 8) × scale ∈ [−8s, +7s] |

### Mojo 0.26 Compatibility Notes

These are non-obvious API quirks discovered during development:

| Issue | Workaround |
|---|---|
| `alias` deprecated | use `comptime` (or ignore warning) |
| `List[T]` access returns `StringSlice`, not `String` | wrap with `String(list[i])` |
| No tuple returns from functions | use single-field struct or `mut` out params |
| `from sys import argv` fails | use `from std.sys import argv` |
| Tuple destructuring `var (a, b) = f()` | not supported; use struct fields |
| `List[T]` requires explicit `.copy()` | implicit copy only for `ImplicitlyCopyable` |
| `with open(...) as f: var x = f.read_bytes()` | declare `x` before `with` block |
| `exp`/`cos`/`sin` on SIMD | `from std.math import exp, cos, sin` + works natively |
| `Dict[String, Int]` | `from std.collections import Dict` (not `std.dictionary`) |

---

## Roadmap

### Phase 4 — Real Weight Integration (next)
- [ ] SafeTensors loader (`scripts/load_safetensors.py`) for Gemma 4 weights
- [ ] Update `.mojostream` writer for real Gemma 4 dims (D=4096, 46 layers)
- [ ] Validate all 40/46 layers against HuggingFace reference outputs
- [ ] Quantize real weights with per-tensor scale calibration

### Phase 5 — Production Inference
- [ ] Greedy sampler + temperature scaling
- [ ] Real embedding lookup (weight-tied)
- [ ] Continuous batching (variable sequence lengths)
- [ ] KV-cache paged allocator (sliding window for Gemma 4 long context)
- [ ] INT8 KV-cache (50% RAM reduction)

### Phase 6 — Edge Deployment
- [ ] ARM NEON support (Raspberry Pi 5, Apple Silicon fallback)
- [ ] NPU offload hooks (Intel NPU on 13th-gen+)
- [ ] WASM target for browser inference

---

## Contributing

```bash
# Fork → clone → branch
git checkout -b feature/your-feature

# Develop against Mojo 0.26
pixi run mojo your_file.mojo

# Before PR: run full validation suite
python3 scripts/gen_reference.py
pixi run mojo validate.mojo      # must be [PASS]
pixi run mojo tokenizer_test.mojo

# Submit PR to main
```

**PR checklist:**
- [ ] `validate.mojo` still [PASS] (rel. RMSE < 1e-4)
- [ ] `tokenizer_test.mojo` round-trip passes
- [ ] No new Mojo compiler errors (warnings from `alias` are acceptable)
- [ ] Benchmark numbers documented if kernel changed

---

## Hardware Targets

| Device | CPU | AVX2 | Expected t/s (demo dims) |
|---|---|---|---:|
| Dev machine (this project) | i7-7500U 2C/4T | ✅ | 6.2 |
| Modern laptop | i7-12th gen 6C/12T | ✅ | ~18 |
| Desktop | Ryzen 9 5900X 12C | ✅ | ~35 |
| Raspberry Pi 5 | ARM Cortex-A76 | ❌ NEON | roadmap |

For 14B real-weight inference (D=4096, 46 layers):  
estimated **~0.8 t/s** on the dev machine (SATA, single token decode).

---

## Tokenizer Note

> The current tokenizer is an ASCII/BPE prototype for pipeline validation. A full UTF-8/SentencePiece implementation is on the roadmap.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Paul Hunck

---

*MojoStream — Paul Hunck, 2026.*
