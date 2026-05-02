# bench.mojo – MojoStream Benchmark-Harness Entry Point
#
# Ausführen:  pixi run mojo bench.mojo
# Pfad:       model.mojostream (im Projektverzeichnis, erzeugt von create_fake_model.py)
from std.sys.info import num_logical_cores
from src.bench.harness import run_harness

fn main() raises:
    run_harness("model.mojostream", num_logical_cores())
