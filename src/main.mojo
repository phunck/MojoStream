# src/main.mojo
# Inferenz-Simulation: 40 Layer × 4096×4096 Q4-Matmul mit Layer-Streaming.
#
# Was wir messen:
#   - t_load[i]     : Zeit zum Laden von layer_i.bin von Disk
#   - t_compute[i]  : Zeit für matmul_q4_bpack_raw (B-Pack Kernel)
#   - wait_time[i]  : max(0, t_load[i+1] - t_compute[i])
#                     = Idle-Zeit, in der die CPU auf die SSD wartet
#
# Bei Q4 B-Pack ≥ 128 GFLOPS und ~8 MB/Layer → t_compute >> t_load →
# wait_time ≈ 0 → Das System ist durchsatzlimitiert, nicht I/O-limitiert.
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores

from src.linalg.kernels import Matrix
from src.streaming.loader import ModelRunner, LayerStats

# ── Konfiguration ───────────────────────────────────────────────────────────
alias N_DIM       = 4096   # Gewichtsmatrix-Dimension (4096 = LLM-typisch)
alias N_LAYERS    = 40     # Anzahl Layer
alias MODEL_PATH  = "model_weights"
# ---------------------------------------------------------------------------


fn print_stats(stats: List[LayerStats], ncores: Int) -> None:
    var total_compute_ms = Float64(0)
    var total_load_ms    = Float64(0)
    var total_wait_ms    = Float64(0)
    var max_load_ms      = Float64(0)
    var max_compute_ms   = Float64(0)

    print()
    print("Layer | Load [ms] | Compute [ms] | Wait [ms] | Overlap?")
    print("------+-----------+--------------+-----------+---------")

    for i in range(len(stats)):
        var s   = stats[i].copy()
        var lms = Float64(Int(s.load_ns))    / 1e6
        var cms = Float64(Int(s.compute_ns)) / 1e6
        var wms = Float64(Int(s.wait_ns))    / 1e6
        var overlap = "YES" if s.wait_ns == 0 else "NO "
        print(i, "  |", lms, "  |", cms, "  |", wms, "  |", overlap)
        total_load_ms    += lms
        total_compute_ms += cms
        total_wait_ms    += wms
        if lms > max_load_ms:    max_load_ms    = lms
        if cms > max_compute_ms: max_compute_ms = cms

    var n       = Float64(len(stats))
    var flops_l = 2.0 * Float64(N_DIM) * Float64(N_DIM) * Float64(N_DIM)
    var gflops  = flops_l / (total_compute_ms / 1e3 / n) / 1e9

    print()
    print("══════════════════════════════════════════════════════════")
    print("ZUSAMMENFASSUNG  (", N_LAYERS, " Layer × ", N_DIM, "×", N_DIM, " Q4)")
    print("══════════════════════════════════════════════════════════")
    print("Threads (Compute):     ", ncores)
    print("Ø Ladezeit/Layer:      ", total_load_ms    / n, "ms")
    print("Ø Rechenzeit/Layer:    ", total_compute_ms / n, "ms")
    print("Ø Wait-Zeit/Layer:     ", total_wait_ms    / n, "ms")
    print("Totale Wait-Zeit:      ", total_wait_ms, "ms")
    print("Effektive GFLOPS:      ", gflops)
    print()
    # Sequenziell vs. Overlapped
    var seq_ms      = total_load_ms + total_compute_ms
    var overlap_ms  = total_compute_ms + max_load_ms  # erster Load blockiert
    print("Wallclock sequenziell: ", seq_ms, "ms")
    print("Wallclock overlapped:  ", overlap_ms, "ms  (erster Load unberlappbar)")
    print("Streaming-Speedup:     ", seq_ms / overlap_ms, "x")
    print()
    if total_wait_ms < 10.0:
        print("✓ I/O hält den Kernel SATT  – Memory-Wall überwunden!")
        print("  Dein System kann Modelle jeder Größe flüssig streamen,")
        print("  solange sie auf die SSD passen.")
    else:
        print("⚠ SSD ist langsamer als der Kernel. Upgrade auf NVMe empfohlen.")


fn main() raises:
    var ncores = num_logical_cores()
    print("══════════════════════════════════════════════════════════")
    print("  Q4-Layer-Streaming Inference Simulation")
    print("══════════════════════════════════════════════════════════")
    print("N_DIM=", N_DIM, "  N_LAYERS=", N_LAYERS, "  Threads=", ncores)
    print("Modell-Pfad:", MODEL_PATH)
    print()

    # Aktivierungs-Matrix (synthetisch, bleibt über alle Layer konstant)
    print("Allokiere Aktivierungen A (", N_DIM, "×", N_DIM, "fp32 =",
          N_DIM * N_DIM * 4 / 1e6, "MB) ...")
    var A = Matrix(N_DIM, N_DIM)
    A.fill_random()

    # Ausgabe-Matrix (wird nach jedem Layer genullt)
    var C = Matrix(N_DIM, N_DIM)

    # ModelRunner initialisieren
    var runner = ModelRunner(MODEL_PATH, N_LAYERS, N_DIM, ncores)

    print("Starte Pipeline ...")
    var t_wall0 = perf_counter_ns()
    var stats   = runner.run_pipeline(C, A)
    var t_wall  = Float64(Int(perf_counter_ns() - t_wall0)) / 1e6

    print("Wall-Clock gesamt:", t_wall, "ms")
    print_stats(stats, ncores)
