# validate.mojo – Numerical Fidelity Check
#
# Ausführen (nach python3 scripts/gen_reference.py):
#   pixi run mojo validate.mojo
from std.sys.info import num_logical_cores
from src.tests.validate_layer import run_validation

fn main() raises:
    run_validation("model.mojostream", "layer0_ref.bin", num_logical_cores())
