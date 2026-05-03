# bench_q4.mojo – Q4 Kernel Benchmark Entry Point
#
# Vergleicht B-Pack (klassisch) vs. Fused-AVX2 (32-Byte-breiter INT4-Kernel).
# Ausführen: pixi run mojo bench_q4.mojo
from std.sys.info import num_logical_cores
from src.bench.benchmark_q4 import run_q4_benchmark

fn main() raises:
    run_q4_benchmark(num_logical_cores())
