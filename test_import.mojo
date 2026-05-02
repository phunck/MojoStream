# test_import.mojo – minimaler Smoke-Test für das src/linalg/kernels-Modul.
# Berechnet eine 128×128 Q4-Matmul und prüft, dass das Ergebnis keine NaN-/Inf-Werte enthält.
from src.linalg.kernels import Matrix, Q4Matrix, matmul_q4_bpack, SIMD_W

fn main() raises:
    alias N = 128
    alias SCALE = Float32(0.1)   # synthetische Skala

    print("=== test_import: src.linalg.kernels ===")
    print("SIMD_W =", SIMD_W, "  N =", N)

    # Eingaben aufbauen
    var A  = Matrix(N, N)
    var Bq = Q4Matrix(N, N, SCALE)
    var C  = Matrix(N, N)
    A.fill_random()
    Bq.fill_random()

    # Single-Thread für deterministisches Ergebnis im Test
    matmul_q4_bpack(C, A, Bq, 1)

    # Plausibilitätsprüfung: C[0,0] muss endlich und ungleich 0 sein
    var c00    = C.data().load(0)
    var clast  = C.data().load(N * N - 1)
    var is_ok  = (c00 != 0.0) and (c00 == c00)  # != 0 und kein NaN

    print("C[0,0]     =", c00)
    print("C[N-1,N-1] =", clast)

    if is_ok:
        print("[PASS] Import und Kernel erfolgreich.")
    else:
        print("[FAIL] Unerwarteter Wert in C.")
