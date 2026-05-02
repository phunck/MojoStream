# MojoStream

> **High-Performance LLM-Inference on Edge Hardware via Mojo.**

Run quantized large language models on laptops and embedded devices — no GPU, no cloud, no Python overhead. MojoStream bringt BLAS-nahe Performance direkt auf die CPU.

---

## Status

| Meilenstein | Status |
|---|---|
| Q4-Kernel (B-Pack, AVX2) | ✅ Stable — April 2026 |
| Layer-Streaming (Double-Buffer) | ✅ Stable — April 2026 |
| Multithreading Scaling | ✅ Stable — April 2026 |
| Gemma 4 Forward-Pass (PLE + KV-Cache) | 🚧 In Progress |
| Weight Converter (gguf → MojoStream) | 📋 Planned |

---

## Highlights

### ⚡ 4-bit Quantized Kernels (AVX2-optimized)

```
N=1024  matmul_q4_bpack:  120 GFLOPS  (4 Threads, i7-7500U)
N=4096  matmul_q4_bpack:   82 GFLOPS  (same hardware)
```

Der **B-Pack-Kernel** pre-dequantisiert Q4-Gewichte (uint8) in einen 8-KB-L1-Puffer und eliminiert damit die Dequant-Redundanz. Das Ergebnis: 12× höhere Effizienz als der naive Fused-Kernel, bei identischer numerischer Genauigkeit (Δ < 10⁻⁵).

### 🔄 Zero-Latency Layer Streaming

Jeder Layer wird Layer-für-Layer von der SSD geladen — das Modell muss **nie vollständig in den RAM**. 40 Layer × 8 MB = 320 MB I/O pro Token-Schritt, bei 0 ms gemessener Wait-Zeit (Compute >> I/O auf SATA-SSD).

### 🗄 Double-Buffering I/O

```
Schritt i:   Compute(Layer i)  ‖  Load(Layer i+1)   ← parallel
Schritt i+1: Compute(Layer i+1) ‖ Load(Layer i+2)
```

Analytisches Overlap-Modell zeigt: I/O hält den Kernel auf jeder gängigen SSD satt.

### 📊 Token/s (i7-7500U, vereinfachter Forward-Pass, batch=4)

| Modus | Zeit/Step | t/s |
|---|---:|---:|
| Compute-only (RAM) | 232 ms | 17 t/s |
| Streaming (SATA-SSD 0.47 GB/s) | 708 ms | 5.6 t/s |
| Extrapoliert auf 14B (7 Matrizen/Layer) | — | ~0.8 t/s |

---

## Architektur

```
mojomatrixtest/
├── src/
│   ├── linalg/
│   │   └── kernels.mojo        ← B-Pack Kernel, RMSNorm, PLE-Skalierung
│   ├── streaming/
│   │   └── loader.mojo         ← ModelRunner, Double-Buffer, LayerStats
│   └── main.mojo               ← Forward-Pass, Token/s Benchmark
├── scripts/
│   └── create_fake_model.py    ← Q4-Gewicht-Generator (row-major + pre-packed)
├── test_import.mojo            ← Smoke-Test (128×128 Q4-Matmul)
├── matrix_bench.mojo           ← Vollständiger Benchmark-Harness
└── model_weights*/             ← Generierte Testdaten (.bin)
```

---

## Quick Start

```bash
# 1. Toolbox einrichten (Fedora)
chmod +x setup_toolbox.sh && ./setup_toolbox.sh
toolbox enter mojo-bench

# 2. Testdaten generieren
python3 scripts/create_fake_model.py --format pre-packed

# 3. Inferenz-Simulation
cd /home/$USER/Documents/vcode/mojomatrixtest
pixi run --manifest-path /home/$USER/mojo-bench/pixi.toml \
    mojo build src/main.mojo -o mojo_inference
./mojo_inference

# 4. Vollständiger Kernel-Benchmark
cd /home/$USER/mojo-bench && ./run_all.sh
```

---

## Roadmap: Gemma 4

Gemma 4 (April 2026) bringt zwei für MojoStream relevante Features:

**Per-Layer Embeddings (PLE)** — jeder Layer skaliert die Aktivierungen mit einem eigenen Float32-Wert. Das erhöht den nutzbaren Dynamikbereich der 4-bit-Quantisierung ohne Mehrkosten beim RAM.

**7-Matrizen-Struktur** — pro Layer: Q, K, V, O (Attention) + Gate, Up, Down (SwiGLU-FFN). MojoStream hat dafür bereits das Streaming-Gerüst (eine Matrix per Load-Call) und Placeholder-Funktionen.

**Shared KV Cache** — cross-request KV-Cache-Sharing für niedrigen RAM-Verbrauch bei langen Kontexten.

---

## Technische Details

| Parameter | Wert |
|---|---|
| Mojo | 0.26.2 |
| Ziel-Hardware | x86-64 mit AVX2 |
| Quantisierung | INT4 symmetric, globale Skala |
| Kernel | B-Pack + MR=4 Register-Blocked |
| L1-Puffer | 8 KB (BK=128 × NR=16 × float32) |
| L2-Puffer | 64 KB A-Panel (MC=128 × BK=128) |
| Thread-Modell | `std.algorithm.parallelize` |

---

## Lizenz

MIT — Eigenentwicklung für Forschungs- und Bildungszwecke.
