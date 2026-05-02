# stream.mojo – True Streaming Mode Entry Point
#
# Layer-für-Layer Inferenz via pread() — physischer RAM ≈ sizeof(1 Layer).
# Ausführen: pixi run mojo stream.mojo
from std.sys.info import num_logical_cores
from src.streaming.stream_runner import run_streaming_bench

fn main() raises:
    run_streaming_bench("model.mojostream", num_logical_cores())
