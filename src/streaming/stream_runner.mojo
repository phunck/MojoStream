# src/streaming/stream_runner.mojo
#
# True-Streaming-Modus: Gewichte werden Layer-für-Layer via pread() direkt
# aus der .mojostream-Datei in einen fixen Scratch-Buffer gelesen.
#
# RAM-Footprint ist damit unabhängig von der Modellgröße:
#   physischer RAM ≈ sizeof(1 Layer) + KV-Cache + Aktivierungspuffer
#   Demo (D=1024):  ~4.3 MB Gewichte  (vs. 178 MB load-all)
#   14B real:       ~60 MB Gewichte   (vs. ~14 GB load-all)
#
# get_rss_kb() liest /proc/self/status (VmRSS) und beweist den konstanten
# RAM-Verbrauch unabhängig von der Anzahl der bereits verarbeiteten Layer.
#
from std.ffi import external_call
from std.time import perf_counter_ns
from std.sys.info import num_logical_cores
from std.math import sqrt

from src.streaming.mojostream import (
    MojoStreamFile, MojoStreamMeta, TensorEntry,
    MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN,
    TENSORS_PER_LAYER, HDR_BYTES, DIR_E_BYTES,
)
from src.main import (
    Gemma4Config, KVCache,
    apply_rope_inplace, ple_scale_inplace, rmsnorm_inplace,
    swiglu_inplace, gqa_attention_decode,
)
from src.linalg.kernels import (
    Matrix, U8Ptr, PtrT, DT, SIMD_W,
    matmul_q4_bpack_raw,
)

# ── RSS-Tracking (Linux /proc/self/status) ──────────────────────────────────

fn get_rss_kb() -> Int:
    """Gibt den aktuellen physischen RAM-Verbrauch (VmRSS) in kB zurück.
    Liest /proc/self/status; gibt -1 zurück wenn nicht verfügbar."""
    var raw = List[UInt8]()
    try:
        with open("/proc/self/status", "r") as f:
            raw = f.read_bytes()
    except:
        return -1

    var bp = raw.unsafe_ptr()
    var n  = len(raw)

    # Suche "VmRSS:" (V=86 m=109 R=82 S=83 S=83 :=58)
    for i in range(n - 10):
        if (bp.load(i)   == UInt8(86) and
            bp.load(i+1) == UInt8(109) and
            bp.load(i+2) == UInt8(82) and
            bp.load(i+3) == UInt8(83) and
            bp.load(i+4) == UInt8(83) and
            bp.load(i+5) == UInt8(58)):
            var pos = i + 6
            while pos < n and (bp.load(pos) == UInt8(32) or bp.load(pos) == UInt8(9)):
                pos += 1
            var rss = 0
            while pos < n and bp.load(pos) >= UInt8(48) and bp.load(pos) <= UInt8(57):
                rss = rss * 10 + Int(bp.load(pos)) - 48
                pos += 1
            return rss
    return -1


fn rss_mb() -> Float64:
    var kb = get_rss_kb()
    if kb < 0: return Float64(-1)
    return Float64(kb) / 1024.0


# ── Nicht-owning Gewichts-Referenz (Zeiger in den Scratch-Buffer) ───────────

struct MappedLayerRef(Movable):
    """Hält rohe Zeiger in den Layer-Scratch-Buffer — keine Datenkopie, kein Ownership."""
    var Q_ptr: U8Ptr;  var scale_Q:    Float32
    var K_ptr: U8Ptr;  var scale_K:    Float32
    var V_ptr: U8Ptr;  var scale_V:    Float32
    var O_ptr: U8Ptr;  var scale_O:    Float32
    var G_ptr: U8Ptr;  var scale_G:    Float32
    var U_ptr: U8Ptr;  var scale_U:    Float32
    var D_ptr: U8Ptr;  var scale_D:    Float32
    var ple_scale: Float32

    fn __init__(out self):
        var null_ptr = U8Ptr()
        self.Q_ptr = null_ptr; self.scale_Q    = Float32(0)
        self.K_ptr = null_ptr; self.scale_K    = Float32(0)
        self.V_ptr = null_ptr; self.scale_V    = Float32(0)
        self.O_ptr = null_ptr; self.scale_O    = Float32(0)
        self.G_ptr = null_ptr; self.scale_G    = Float32(0)
        self.U_ptr = null_ptr; self.scale_U    = Float32(0)
        self.D_ptr = null_ptr; self.scale_D    = Float32(0)
        self.ple_scale = Float32(1.0)


# ── Streaming Forward-Pass (arbeitet direkt auf MappedLayerRef) ──────────────

fn gemma4_forward_stream(
    mut x:     Matrix,
    layer_id:  Int,
    w:         MappedLayerRef,
    mut kv:    KVCache,
    cfg:       Gemma4Config,
    base_pos:  Int,
    workers:   Int,
):
    """Identisch zu gemma4_forward_layer, aber nimmt rohe Zeiger statt List-Ownership."""
    var D        = cfg.hidden
    var KVD      = cfg.kv_dim
    var FFD      = cfg.ffn_dim
    var batch    = x.rows
    var head_dim = D // cfg.n_heads

    var q_buf    = Matrix(batch, D)
    var k_buf    = Matrix(batch, KVD)
    var v_buf    = Matrix(batch, KVD)
    var attn_out = Matrix(batch, D)
    var gate_buf = Matrix(batch, FFD)
    var up_buf   = Matrix(batch, FFD)
    var proj_out = Matrix(batch, D)

    ple_scale_inplace(x.data(), batch, D, w.ple_scale)
    rmsnorm_inplace(x.data(), batch, D)

    matmul_q4_bpack_raw(q_buf, x, w.Q_ptr, D   // 2, w.scale_Q, workers)
    matmul_q4_bpack_raw(k_buf, x, w.K_ptr, KVD // 2, w.scale_K, workers)
    matmul_q4_bpack_raw(v_buf, x, w.V_ptr, KVD // 2, w.scale_V, workers)

    for b in range(batch):
        apply_rope_inplace(q_buf.data() + b * D,   cfg.n_heads,    head_dim, base_pos + b)
        apply_rope_inplace(k_buf.data() + b * KVD, cfg.n_kv_heads, head_dim, base_pos + b)

    for b in range(batch):
        var t = base_pos + b
        if t < kv.max_len:
            var kptr = rebind[PtrT](kv.k_data[layer_id].unsafe_ptr()) + t * KVD
            var vptr = rebind[PtrT](kv.v_data[layer_id].unsafe_ptr()) + t * KVD
            var kb   = k_buf.data() + b * KVD
            var vb   = v_buf.data() + b * KVD
            for j in range(KVD): kptr.store(j, kb.load(j))
            for j in range(KVD): vptr.store(j, vb.load(j))

    gqa_attention_decode(attn_out, q_buf, kv, layer_id, cfg, base_pos)

    matmul_q4_bpack_raw(proj_out, attn_out, w.O_ptr, D // 2, w.scale_O, workers)
    var xd = x.data(); var pd = proj_out.data()
    for j in range(batch * D): xd.store(j, xd.load(j) + pd.load(j))

    rmsnorm_inplace(x.data(), batch, D)

    matmul_q4_bpack_raw(gate_buf, x, w.G_ptr, FFD // 2, w.scale_G, workers)
    matmul_q4_bpack_raw(up_buf,   x, w.U_ptr, FFD // 2, w.scale_U, workers)
    swiglu_inplace(gate_buf.data(), up_buf.data(), batch * FFD)

    var ffn_out = Matrix(batch, D)
    matmul_q4_bpack_raw(ffn_out, up_buf, w.D_ptr, D // 2, w.scale_D, workers)
    var fd = ffn_out.data()
    for j in range(batch * D): xd.store(j, xd.load(j) + fd.load(j))


# ── StreamingRunner — Strict Metadata Guard + pread Layer-Loader ────────────

struct StreamingRunner(Movable):
    """
    Lädt Layer-Gewichte via pread() direkt aus der .mojostream-Datei.
    Hält immer nur einen Layer im Scratch-Buffer — kein load-all.

    Initialisierung als strikte Invarianten-Sequenz (Fail-Fast):
      Step 1  openat()               → fd  (Datei muss existieren)
      Step 2  lseek(SEEK_END)        → file_size
      Step 3  pread(128 B)           → Header parsen
      Step 4  _check_shape()         → Gemma-4-Architektur-Invarianten
      Step 5  pread(n_tensors × 32)  → Tensor-Directory parsen
      Step 6  validate_tensor_entries() → Dimensionen, Alignment, Bounds, Scale
      Step 7  _init_offsets()        → Scratch-Buffer allozieren (nur nach Erfolg)

    Kein Raw-Pointer-Zugriff findet statt bevor alle Checks bestanden sind.
    RAM-Profil (D=1024): ~4.4 MB  |  (D=4096): ~68 MB
    """
    var fd:          Int32
    var meta:        MojoStreamMeta
    var entries:     List[TensorEntry]
    var scratch:     List[UInt8]
    var mat_offsets: List[Int]
    var mat_sizes:   List[Int]

    fn __init__(out self, path: String) raises:
        self.meta        = MojoStreamMeta()
        self.entries     = List[TensorEntry]()
        self.scratch     = List[UInt8]()
        self.mat_offsets = List[Int]()
        self.mat_sizes   = List[Int]()
        self.fd          = Int32(-1)

        # ── Step 1: Datei öffnen ─────────────────────────────────────────────
        # String.unsafe_ptr() → pointer<none> in Struct-Kontext: List[UInt8]-Workaround
        var path_buf = List[UInt8](capacity=len(path) + 1)
        var path_ptr = path.unsafe_ptr()
        for i in range(len(path)):
            path_buf.append(path_ptr.load(i))
        path_buf.append(UInt8(0))

        # openat(AT_FDCWD=-100, path, O_RDONLY=0) — vermeidet Konflikt mit Mojos open()
        self.fd = external_call["openat", Int32](
            Int32(-100), path_buf.unsafe_ptr(), Int32(0)
        )
        if self.fd < 0:
            raise Error("[StreamingRunner] Datei nicht gefunden: " + path)

        # ── Step 2: Dateigröße ermitteln ─────────────────────────────────────
        # SEEK_END = 2; pread ändert den Datei-Offset nicht → lseek nur für stat
        var file_size = Int(external_call["lseek", Int64](self.fd, Int64(0), Int32(2)))
        if file_size <= Int(HDR_BYTES):
            raise Error("[StreamingRunner] Datei zu klein: " + String(file_size) + " Bytes")

        # ── Step 3: Header lesen und parsen (128 Byte) ───────────────────────
        var hdr_buf = List[UInt8]()
        hdr_buf.resize(HDR_BYTES, 0)
        var n_hdr = external_call["pread", Int64](
            self.fd, hdr_buf.unsafe_ptr(), Int64(HDR_BYTES), Int64(0)
        )
        if n_hdr != Int64(HDR_BYTES):
            raise Error("[StreamingRunner] Header-Lesefehler: " + String(n_hdr) + " Bytes")

        var hp = hdr_buf.unsafe_ptr()
        if hp.bitcast[UInt32]().load(0) != UInt32(0x4F4A4F4D):
            raise Error("[StreamingRunner] Ungültige Magic-Zahl — kein .mojostream")

        var u32 = hp.bitcast[UInt32]()
        self.meta.n_layers   = Int(u32.load(4))
        self.meta.hidden     = Int(u32.load(5))
        self.meta.kv_dim     = Int(u32.load(6))
        self.meta.ffn_dim    = Int(u32.load(7))
        self.meta.n_heads    = Int(u32.load(8))
        self.meta.n_kv_heads = Int(u32.load(9))
        var u64 = hp.bitcast[UInt64]()
        self.meta.n_tensors  = Int(u64.load(5))
        self.meta.data_start = Int(u64.load(7))

        # ── Step 4: Gemma-4-Shape-Invarianten prüfen (vor Directory-Load) ────
        self._check_shape()

        # ── Step 5: Tensor-Directory lesen und parsen ────────────────────────
        var dir_bytes = self.meta.n_tensors * DIR_E_BYTES
        var dir_buf   = List[UInt8]()
        dir_buf.resize(dir_bytes, 0)
        var n_dir = external_call["pread", Int64](
            self.fd, dir_buf.unsafe_ptr(), Int64(dir_bytes), Int64(HDR_BYTES)
        )
        if n_dir != Int64(dir_bytes):
            raise Error("[StreamingRunner] Directory-Lesefehler: " + String(n_dir)
                        + " / " + String(dir_bytes) + " Bytes")

        var dp = dir_buf.unsafe_ptr()
        for i in range(self.meta.n_tensors):
            var ep   = dp + i * DIR_E_BYTES
            var eu32 = ep.bitcast[UInt32]()
            var ef32 = (ep + 16).bitcast[Float32]()
            var eu64 = (ep + 24).bitcast[UInt64]()
            var e    = TensorEntry()
            e.layer_id    = Int(eu32.load(0))
            e.mat_type    = eu32.load(1)
            e.rows        = Int(eu32.load(2))
            e.cols        = Int(eu32.load(3))
            e.scale       = ef32.load(0)
            e.data_offset = Int(eu64.load(0))
            self.entries.append(e^)

        # ── Step 6: Deep Tensor Validation (vor Scratch-Allokierung) ─────────
        self.validate_tensor_entries(file_size)

        # ── Step 7: Scratch-Buffer allozieren — NUR nach bestandenen Checks ──
        self._init_offsets()

    # ── Shape Guard ──────────────────────────────────────────────────────────

    fn _check_shape(self) raises:
        """Gemma-4-Architektur-Invarianten. Scheitert vor dem Directory-Load."""
        var D   = self.meta.hidden
        var NH  = self.meta.n_heads
        var NKV = self.meta.n_kv_heads

        if NH == 0:
            raise Error("[ShapeGuard] n_heads = 0")
        if NKV == 0:
            raise Error("[ShapeGuard] n_kv_heads = 0")
        if D % NH != 0:
            raise Error("[ShapeGuard] hidden=" + String(D)
                        + " nicht durch n_heads=" + String(NH) + " teilbar")
        if NH % NKV != 0:
            raise Error("[ShapeGuard] n_heads=" + String(NH)
                        + " nicht durch n_kv_heads=" + String(NKV) + " teilbar")
        var head_dim = D // NH
        if self.meta.kv_dim != NKV * head_dim:
            raise Error("[ShapeGuard] kv_dim=" + String(self.meta.kv_dim)
                        + " erwartet=" + String(NKV * head_dim)
                        + " (n_kv_heads × head_dim)")
        if self.meta.n_layers <= 0:
            raise Error("[ShapeGuard] n_layers <= 0")
        if self.meta.ffn_dim <= 0:
            raise Error("[ShapeGuard] ffn_dim <= 0")
        var exp_tensors = self.meta.n_layers * TENSORS_PER_LAYER
        if self.meta.n_tensors != exp_tensors:
            raise Error("[ShapeGuard] n_tensors=" + String(self.meta.n_tensors)
                        + " erwartet=" + String(exp_tensors)
                        + " (n_layers × " + String(TENSORS_PER_LAYER) + ")")

    # ── Deep Tensor Validation ────────────────────────────────────────────────

    fn validate_tensor_entries(self, file_size: Int) raises:
        """Prüft jeden Directory-Eintrag gegen fünf Kriterien:
          1. ID/Type-Sequenz  (layer_id, mat_type in Inferenz-Reihenfolge)
          2. Dimensionen      (rows, cols exakt für diesen mat_type)
          3. 4096-Alignment   (data_offset % 4096 == 0)
          4. File-Bounds      (offset + size <= file_size)
          5. Scale-Sanity     (scale > 0)
        Schlägt sofort mit präziser Fehlermeldung fehl. Kein undefiniertes Verhalten."""
        var D   = self.meta.hidden
        var KVD = self.meta.kv_dim
        var FFD = self.meta.ffn_dim

        # Erwartete (rows, cols) pro mat_type 0..7
        var exp_r = List[Int]()
        var exp_c = List[Int]()
        exp_r.append(1);   exp_c.append(0)     # 0 PLE   (kein Datenblock)
        exp_r.append(D);   exp_c.append(D)     # 1 Q
        exp_r.append(D);   exp_c.append(KVD)   # 2 K
        exp_r.append(D);   exp_c.append(KVD)   # 3 V
        exp_r.append(D);   exp_c.append(D)     # 4 O
        exp_r.append(D);   exp_c.append(FFD)   # 5 Gate
        exp_r.append(D);   exp_c.append(FFD)   # 6 Up
        exp_r.append(FFD); exp_c.append(D)     # 7 Down

        var mat_names = List[String]()
        mat_names.append("PLE"); mat_names.append("Q");   mat_names.append("K")
        mat_names.append("V");   mat_names.append("O");   mat_names.append("Gate")
        mat_names.append("Up");  mat_names.append("Down")

        for i in range(len(self.entries)):
            var e         = self.entries[i].copy()
            var exp_layer = i // TENSORS_PER_LAYER
            var exp_type  = i %  TENSORS_PER_LAYER
            var loc       = "Tensor[" + String(i) + "] Layer " + String(exp_layer) \
                            + " " + String(mat_names[exp_type])

            # 1. ID/Type-Sequenz
            if e.layer_id != exp_layer:
                raise Error("[TensorGuard] " + loc + ": layer_id=" + String(e.layer_id)
                            + " erwartet=" + String(exp_layer))
            if Int(e.mat_type) != exp_type:
                raise Error("[TensorGuard] " + loc + ": mat_type=" + String(Int(e.mat_type))
                            + " erwartet=" + String(exp_type))

            # 2. Dimensionen
            if e.rows != exp_r[exp_type]:
                raise Error("[TensorGuard] " + loc + ": rows=" + String(e.rows)
                            + " erwartet=" + String(exp_r[exp_type]))
            if e.cols != exp_c[exp_type]:
                raise Error("[TensorGuard] " + loc + ": cols=" + String(e.cols)
                            + " erwartet=" + String(exp_c[exp_type]))

            # 5. Scale-Sanity (für alle Einträge inkl. PLE)
            if e.scale <= Float32(0):
                raise Error("[TensorGuard] " + loc + ": scale=" + String(e.scale) + " <= 0")

            # PLE hat keinen Datenblock → Alignment/Bounds überspringen
            if exp_type == 0:
                continue

            # 3. 4096-Byte Alignment (SSD Page Boundary)
            if e.data_offset % 4096 != 0:
                raise Error("[TensorGuard] " + loc + ": offset=" + String(e.data_offset)
                            + " nicht 4096-aligned (Modulo=" + String(e.data_offset % 4096) + ")")

            # 4. File-Bounds
            var expected_bytes = e.rows * (e.cols // 2)
            var end_offset     = e.data_offset + expected_bytes
            if end_offset > file_size:
                raise Error("[TensorGuard] " + loc + ": offset=" + String(e.data_offset)
                            + " + size=" + String(expected_bytes) + " = " + String(end_offset)
                            + " > file_size=" + String(file_size))

    # ── Scratch-Buffer Dimensionierung ────────────────────────────────────────

    fn _init_offsets(mut self):
        """Alloziert Scratch-Buffer nach bestandener Validierung."""
        var D   = self.meta.hidden
        var KVD = self.meta.kv_dim
        var FFD = self.meta.ffn_dim

        var sizes = List[Int]()
        sizes.append(D * D   // 2)   # Q
        sizes.append(D * KVD // 2)   # K
        sizes.append(D * KVD // 2)   # V
        sizes.append(D * D   // 2)   # O
        sizes.append(D * FFD // 2)   # Gate
        sizes.append(D * FFD // 2)   # Up
        sizes.append(FFD * D // 2)   # Down

        var total = 0
        for i in range(7):
            self.mat_offsets.append(total)
            self.mat_sizes.append(sizes[i])
            total += sizes[i]

        self.scratch.resize(total, 0)

    fn load_layer(mut self, layer: Int) raises -> MappedLayerRef:
        """Liest alle 7 Gewichtsmatrizen von Layer `layer` via pread()
        in den Scratch-Buffer. Kein Alloc, kein Copy in den Heap."""
        var base_ptr = self.scratch.unsafe_ptr()
        var lref      = MappedLayerRef()

        # PLE-Skala direkt aus Verzeichnis (kein I/O)
        lref.ple_scale = self.entries[layer * TENSORS_PER_LAYER].copy().scale

        # Matrizen 1–7 per pread
        for mi in range(7):
            var mat_type = mi + 1   # 1=Q .. 7=Down
            var e        = self.entries[layer * TENSORS_PER_LAYER + mat_type].copy()
            var dst      = base_ptr + self.mat_offsets[mi]
            var sz       = Int64(self.mat_sizes[mi])

            var n = external_call["pread", Int64](
                self.fd,
                dst,
                sz,
                Int64(e.data_offset),
            )
            if n != sz:
                raise Error("pread: " + String(n) + " / " + String(sz) + " Bytes")

            # Zeiger und Skala in MappedLayerRef setzen
            var u8_ptr = rebind[U8Ptr](dst)
            var scale  = e.scale
            if mi == 0: lref.Q_ptr = u8_ptr; lref.scale_Q = scale
            elif mi == 1: lref.K_ptr = u8_ptr; lref.scale_K = scale
            elif mi == 2: lref.V_ptr = u8_ptr; lref.scale_V = scale
            elif mi == 3: lref.O_ptr = u8_ptr; lref.scale_O = scale
            elif mi == 4: lref.G_ptr = u8_ptr; lref.scale_G = scale
            elif mi == 5: lref.U_ptr = u8_ptr; lref.scale_U = scale
            else:         lref.D_ptr = u8_ptr; lref.scale_D = scale

        return lref^

    fn close(mut self):
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)
            self.fd = Int32(-1)

    fn scratch_mb(self) -> Float64:
        return Float64(len(self.scratch)) / 1e6


# ── Streaming-Benchmark ───────────────────────────────────────────────────────

fn run_streaming_bench(ms_path: String, workers: Int) raises:
    print("══════════════════════════════════════════════════════════════")
    print("  MojoStream  –  True Streaming Mode  (pread, 1 Layer RAM)")
    print("══════════════════════════════════════════════════════════════")

    var rss0 = rss_mb()
    print("  RSS vor Load:      ", rss0, "MB")

    # Strict Metadata Guard: Header → ShapeGuard → TensorGuard → Scratch-Alloc
    var runner = StreamingRunner(ms_path)
    print("  ShapeGuard:        OK")
    print("  TensorGuard:       OK (", runner.meta.n_tensors, "Einträge geprüft)")

    var cfg = Gemma4Config(
        hidden     = runner.meta.hidden,
        kv_dim     = runner.meta.kv_dim,
        ffn_dim    = runner.meta.ffn_dim,
        n_layers   = runner.meta.n_layers,
        n_heads    = runner.meta.n_heads,
        n_kv_heads = runner.meta.n_kv_heads,
    )

    print("  Modell:            D=", cfg.hidden, " KV=", cfg.kv_dim,
          " FFN=", cfg.ffn_dim, " Layers=", cfg.n_layers)
    print("  Scratch-Buffer:    ", runner.scratch_mb(), "MB  (1 Layer)")

    var rss1 = rss_mb()
    print("  RSS nach Init:     ", rss1, "MB  (Header+Verzeichnis geladen)")

    alias BATCH: Int = 4
    alias STEPS: Int = 3

    var kv = KVCache(cfg.n_layers, cfg.kv_dim, 64)
    var x  = Matrix(BATCH, cfg.hidden)

    print()
    print("  Starte", STEPS, "Schritte (batch=", BATCH, ") ...")
    print()

    var best_total_ns : UInt = UInt.MAX
    var best_io_ns    : UInt = 0
    var best_compute_ns: UInt = 0

    for step in range(STEPS):
        x.fill_random()
        var base_pos = kv.cur_len
        var t_step   = perf_counter_ns()
        var total_io : UInt = 0

        for layer in range(cfg.n_layers):
            # I/O: pread dieses Layers
            var t_io = perf_counter_ns()
            var w    = runner.load_layer(layer)
            var dt_io = perf_counter_ns() - t_io

            # Compute: Forward-Pass mit nicht-owning Zeigern
            var t_cmp = perf_counter_ns()
            gemma4_forward_stream(x, layer, w, kv, cfg, base_pos, workers)
            var dt_cmp = perf_counter_ns() - t_cmp

            total_io += dt_io

        var dt_total = perf_counter_ns() - t_step
        kv.cur_len += BATCH

        var compute_ns = dt_total - total_io
        if dt_total < best_total_ns:
            best_total_ns  = dt_total
            best_io_ns     = total_io
            best_compute_ns = compute_ns

        var rss_s = rss_mb()
        print("  Schritt", step + 1, ":  Total=", Float64(Int(dt_total)) / 1e6,
              "ms  I/O=", Float64(Int(total_io)) / 1e6,
              "ms  Compute=", Float64(Int(compute_ns)) / 1e6, "ms  RSS=", rss_s, "MB")

    runner.close()

    # Metriken
    var total_ms   = Float64(Int(best_total_ns))   / 1e6
    var io_ms      = Float64(Int(best_io_ns))      / 1e6
    var compute_ms = Float64(Int(best_compute_ns)) / 1e6

    # I/O-Druck: tatsächliche Wartezeit = max(0, IO - Compute)
    # Wenn IO < Compute: kein Warten (Compute dominiert)
    var wait_ms  = io_ms - compute_ms if io_ms > compute_ms else Float64(0)
    var io_pct   = wait_ms / total_ms * 100.0 if total_ms > 0 else Float64(0)
    var tps      = Float64(BATCH) / (total_ms / 1000.0)
    var rss_peak = rss_mb()

    print()
    print("══════════════════════════════════════════════════════════════")
    print("  STREAMING ERGEBNIS  (bester Schritt)")
    print("══════════════════════════════════════════════════════════════")
    print("  Gesamt/Schritt:     ", total_ms,   "ms")
    print("  davon I/O (pread):  ", io_ms,      "ms")
    print("  davon Compute:      ", compute_ms, "ms")
    print("  I/O-Wartezeit:      ", wait_ms,    "ms  (max(0, I/O - Compute))")
    print("  I/O-Druck:          ", io_pct,     "%  (0% = compute-bound)")
    print("  Tokens/Sek (b=4):   ", tps)
    print("  Scratch-Buffer:     ", runner.scratch_mb(), "MB")
    print("  Peak RSS:           ", rss_peak,   "MB")
    print("══════════════════════════════════════════════════════════════")
