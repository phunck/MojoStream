# first_light.mojo – Standalone First-Light Benchmark
#
# Führt einen vollständigen Gemma-4 Forward-Pass (40 Layer, Demo-Dimensionen,
# zufällige Q4-Gewichte) durch und misst Tokens/Sekunde.
#
# Ausführen: mojo first_light.mojo
from std.sys.info import num_logical_cores
from src.main import demo_first_light

fn main() raises:
    demo_first_light(num_logical_cores())
