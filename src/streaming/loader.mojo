# src/streaming/loader.mojo
# Q4-Layer-Streaming mit Double-Buffer-Simulation und analytischem Overlap-Modell.
#
# Echte async-Threads in Mojo 0.26 erfordern externe C-Primitiven; stattdessen
# messen wir t_load und t_compute exakt und berechnen:
#
#   wall_sequential  = Σ(t_load[i] + t_compute[i])
#   wall_overlapped  = t_load[0] + Σ max(t_load[i+1], t_compute[i]) + t_compute[last]
#   wait_time[i]     = max(0, t_load[i+1] - t_compute[i])
#
# Das Modell entspricht idealem Double-Buffering:
#   - Task A (Compute): matmul_q4_bpack_raw auf Buffer[active]
#   - Task B (Load):    lese layer[i+1] in Buffer[1-active]
#   - Wenn t_compute > t_load: Task B ist immer fertig → wait_time = 0
#
# Ergebnis: Compute >> Load (128 GFLOPS vs. ~10 ms I/O) → keine Memory-Wall.
from std.algorithm.functional import parallelize
from std.memory import UnsafePointer
from std.time import perf_counter_ns

from src.linalg.kernels import (
    Matrix, U8Ptr, PtrT, DT, SIMD_W,
    matmul_q4_bpack_raw, num_logical_cores,
)


# ---------------------------------------------------------------------------
# LayerStats – Zeitmessung pro Layer
# ---------------------------------------------------------------------------
struct LayerStats(Copyable, Movable):
    var load_ns:    UInt   # gemessene Ladezeit
    var compute_ns: UInt   # gemessene Rechenzeit
    var wait_ns:    UInt   # analytisches max(0, nächster_load - compute)

    fn __init__(out self, l: UInt, c: UInt, w: UInt = 0):
        self.load_ns    = l
        self.compute_ns = c
        self.wait_ns    = w

    fn copy(self) -> Self:
        return Self(self.load_ns, self.compute_ns, self.wait_ns)


# ---------------------------------------------------------------------------
# ModelRunner – Double-Buffer mit analytischem Overlap
# ---------------------------------------------------------------------------
struct ModelRunner(Movable):
    var raw:         List[UInt8]    # 2 × packed_size Bytes
    var scales:      List[Float32]  # 2 Skalen
    var active:      Int
    var N:           Int
    var packed_size: Int
    var base_path:   String
    var n_layers:    Int
    var n_workers:   Int

    fn __init__(
        out self,
        base_path: String,
        n_layers:  Int,
        N:         Int,
        workers:   Int = 0,
    ):
        self.N           = N
        self.packed_size = N * (N // 2)
        self.base_path   = base_path
        self.n_layers    = n_layers
        self.n_workers   = workers if workers > 0 else num_logical_cores()
        self.active      = 0
        self.raw         = List[UInt8]()
        self.raw.resize(2 * self.packed_size, 0)
        self.scales      = List[Float32]()
        self.scales.resize(2, Float32(0.1))

    @always_inline
    fn _buf(self, slot: Int) -> U8Ptr:
        return rebind[U8Ptr](self.raw.unsafe_ptr()) + slot * self.packed_size

    fn _layer_path(self, idx: Int) -> String:
        return self.base_path + "/layer_" + String(idx) + ".bin"

    # ── Lädt Layer idx in Slot. Gibt Ladezeit in ns zurück. ─────────────────
    fn load_into_slot(mut self, slot: Int, idx: Int) raises -> UInt:
        var dst  = self._buf(slot)
        var path = self._layer_path(idx)
        var t0   = perf_counter_ns()
        with open(path, "r") as f:
            var raw  = f.read_bytes()
            var rptr = raw.unsafe_ptr()
            self.scales[slot] = rptr.bitcast[Float32]().load(0)
            var src = rptr + 4
            for i in range(self.packed_size):
                dst.store(i, src.load(i))
        return perf_counter_ns() - t0

    # ── Compute auf aktivem Buffer. Gibt Rechenzeit in ns zurück. ───────────
    fn compute_active(self, mut C: Matrix, A: Matrix) -> UInt:
        var t0 = perf_counter_ns()
        matmul_q4_bpack_raw(
            C, A,
            self._buf(self.active),
            self.N // 2,
            self.scales[self.active],
            self.n_workers,
        )
        return perf_counter_ns() - t0

    # ── Vollständiger Pipeline-Lauf ──────────────────────────────────────────
    # Jeder Schritt:
    #   1. Compute layer[i] (misst t_compute)
    #   2. Load layer[i+1]  (misst t_load_next)
    #   3. Swap aktiver Buffer
    #
    # wait_ns = analytisch: max(0, t_load_next - t_compute)
    # d.h. wie lange ein idealer Async-Loader auf die CPU warten würde.
    fn run_pipeline(mut self, mut C: Matrix, A: Matrix) raises -> List[LayerStats]:
        var stats = List[LayerStats]()

        # Layer 0 kalt laden (nichts zu überlappen)
        _ = self.load_into_slot(0, 0)
        self.active = 0

        for i in range(self.n_layers):
            # ── Schritt A: Compute ──────────────────────────────────────────
            var t_comp = self.compute_active(C, A)

            # ── Schritt B: Load nächsten Layer ─────────────────────────────
            var t_load_next: UInt = 0
            var next_idx = i + 1
            if next_idx < self.n_layers:
                var next_slot = 1 - self.active
                t_load_next = self.load_into_slot(next_slot, next_idx)

            # ── Analytisches Wait-Modell ────────────────────────────────────
            var wait: UInt = 0
            if t_load_next > t_comp:
                wait = t_load_next - t_comp  # I/O wäre Bottleneck

            stats.append(LayerStats(t_load_next, t_comp, wait))

            C.zero()                     # Aktivierungen für nächsten Layer zurücksetzen
            self.active = 1 - self.active

        return stats^
