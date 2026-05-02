# src/streaming/mojostream.mojo
#
# Binärer Reader für das .mojostream-Format (Phase 3).
#
# Dateiformat (little-endian):
#   HEADER  128 Byte  – Magic + Modell-Metadaten
#   TENSOR-VERZEICHNIS  n_tensors × 32 Byte  – Offsets, Skalen, Dimensionen
#   PADDING  Nullbytes bis zur nächsten 4096-Byte-Grenze
#   TENSOR-DATEN  page-aligned Q4-Gewichte (rows × cols//2 Byte pro Tensor)
#
# Strategie: Datei einmalig vollständig in RAM laden (eine sequenzielle I/O-
# Operation statt 40 Einzelöffnungen). Tensoren werden danach als Zeiger
# in den Puffer zurückgegeben – kein zweites Kopieren.
#
from std.memory import UnsafePointer
from std.time import perf_counter_ns

from src.linalg.kernels import U8Ptr, PtrT, DT

# ── Typ-Tags (müssen mit create_fake_model.py übereinstimmen) ───────────────
alias MS_PLE  : UInt32 = 0
alias MS_Q    : UInt32 = 1
alias MS_K    : UInt32 = 2
alias MS_V    : UInt32 = 3
alias MS_O    : UInt32 = 4
alias MS_GATE : UInt32 = 5
alias MS_UP   : UInt32 = 6
alias MS_DOWN : UInt32 = 7

alias TENSORS_PER_LAYER : Int = 8   # 1 PLE + 7 Matrizen (feste Reihenfolge)
alias HDR_BYTES         : Int = 128
alias DIR_E_BYTES       : Int = 32
alias MS_PAGE           : Int = 4096

# Rückgabe-Typ für tensor_ptr (Mojo 0.26 unterstützt keine Tuple-Returns)
struct TensorRef(Copyable, Movable):
    var ptr:   U8Ptr
    var scale: Float32

    fn __init__(out self, ptr: U8Ptr, scale: Float32):
        self.ptr   = ptr
        self.scale = scale

    fn copy(self) -> Self:
        return Self(self.ptr, self.scale)^

# ── Modell-Metadaten aus dem Header ─────────────────────────────────────────
struct MojoStreamMeta(Copyable, Movable):
    var n_layers:   Int
    var hidden:     Int
    var kv_dim:     Int
    var ffn_dim:    Int
    var n_heads:    Int
    var n_kv_heads: Int
    var n_tensors:  Int
    var data_start: Int

    fn __init__(out self):
        self.n_layers   = 0; self.hidden     = 0; self.kv_dim     = 0
        self.ffn_dim    = 0; self.n_heads    = 0; self.n_kv_heads = 0
        self.n_tensors  = 0; self.data_start = 0

    fn copy(self) -> Self:
        var out = Self()
        out.n_layers   = self.n_layers;   out.hidden     = self.hidden
        out.kv_dim     = self.kv_dim;     out.ffn_dim    = self.ffn_dim
        out.n_heads    = self.n_heads;    out.n_kv_heads = self.n_kv_heads
        out.n_tensors  = self.n_tensors;  out.data_start = self.data_start
        return out^


# ── Tensor-Verzeichnis-Eintrag (32 Byte in der Datei) ───────────────────────
struct TensorEntry(Copyable, Movable):
    var layer_id:    Int
    var mat_type:    UInt32
    var rows:        Int
    var cols:        Int
    var scale:       Float32
    var data_offset: Int     # 0 für PLE-Einträge (kein Datenblock)

    fn __init__(out self):
        self.layer_id = 0; self.mat_type = 0
        self.rows = 0;     self.cols = 0
        self.scale = 0.0;  self.data_offset = 0

    fn copy(self) -> Self:
        var out = Self()
        out.layer_id    = self.layer_id;    out.mat_type    = self.mat_type
        out.rows        = self.rows;        out.cols        = self.cols
        out.scale       = self.scale;       out.data_offset = self.data_offset
        return out^


# ── Haupt-Struct: geöffnete .mojostream-Datei ───────────────────────────────
struct MojoStreamFile(Movable):
    var raw:     List[UInt8]        # gesamter Dateiinhalt im RAM
    var meta:    MojoStreamMeta
    var entries: List[TensorEntry]  # Tensor-Verzeichnis
    var load_ns: UInt               # gemessene I/O-Zeit in Nanosekunden

    fn __init__(out self, path: String) raises:
        self.meta    = MojoStreamMeta()
        self.entries = List[TensorEntry]()
        self.load_ns = 0

        # Eine einzige sequenzielle Leseoperation (SSD-optimal)
        var t0 = perf_counter_ns()
        with open(path, "r") as f:
            self.raw = f.read_bytes()
        self.load_ns = perf_counter_ns() - t0

        self._parse_header()
        self._parse_directory()

    fn _parse_header(mut self) raises:
        var bp = self.raw.unsafe_ptr()

        # Magic prüfen: erste 4 Byte müssen "MOJO" (0x4F4A4F4D) sein
        var magic = bp.bitcast[UInt32]().load(0)
        if magic != UInt32(0x4F4A4F4D):
            raise Error(".mojostream: ungültige Magic-Zahl (erwartet 'MOJO...')")

        # uint32-Felder ab Byte 16
        var u32 = bp.bitcast[UInt32]()
        self.meta.n_layers   = Int(u32.load(4))   # Byte 16
        self.meta.hidden     = Int(u32.load(5))   # Byte 20
        self.meta.kv_dim     = Int(u32.load(6))   # Byte 24
        self.meta.ffn_dim    = Int(u32.load(7))   # Byte 28
        self.meta.n_heads    = Int(u32.load(8))   # Byte 32
        self.meta.n_kv_heads = Int(u32.load(9))   # Byte 36

        # uint64-Felder ab Byte 40
        var u64 = bp.bitcast[UInt64]()
        self.meta.n_tensors  = Int(u64.load(5))   # Byte 40
        # u64.load(6) = dir_offset (immer 128, nicht gespeichert)
        self.meta.data_start = Int(u64.load(7))   # Byte 56

    fn _parse_directory(mut self):
        var bp = self.raw.unsafe_ptr()
        for i in range(self.meta.n_tensors):
            var ep   = bp + HDR_BYTES + i * DIR_E_BYTES
            var eu32 = ep.bitcast[UInt32]()
            var ef32 = (ep + 16).bitcast[Float32]()
            var eu64 = (ep + 24).bitcast[UInt64]()

            var e = TensorEntry()
            e.layer_id    = Int(eu32.load(0))   # Byte 0 im Eintrag
            e.mat_type    = eu32.load(1)         # Byte 4
            e.rows        = Int(eu32.load(2))    # Byte 8
            e.cols        = Int(eu32.load(3))    # Byte 12
            e.scale       = ef32.load(0)         # Byte 16
            # eu32.load(5) = flags (Byte 20, aktuell ungenutzt)
            e.data_offset = Int(eu64.load(0))    # Byte 24
            self.entries.append(e^)

    fn tensor_ptr(self, layer: Int, mat_type: UInt32) -> TensorRef:
        """Gibt TensorRef(ptr, scale) für den angegebenen Tensor zurück.
        Direkter Index (O(1)) mit linearem Fallback falls nötig."""
        var idx = layer * TENSORS_PER_LAYER + Int(mat_type)
        if idx < len(self.entries):
            var e = self.entries[idx].copy()
            if Int(e.layer_id) == layer and e.mat_type == mat_type:
                return TensorRef(
                    rebind[U8Ptr](self.raw.unsafe_ptr() + e.data_offset),
                    e.scale,
                )
        # Lineares Fallback (sollte nie aufgerufen werden)
        for i in range(len(self.entries)):
            var e = self.entries[i].copy()
            if Int(e.layer_id) == layer and e.mat_type == mat_type:
                return TensorRef(
                    rebind[U8Ptr](self.raw.unsafe_ptr() + e.data_offset),
                    e.scale,
                )
        return TensorRef(rebind[U8Ptr](self.raw.unsafe_ptr()), Float32(0.0))

    fn ple_scale(self, layer: Int) -> Float32:
        """Gibt die PLE-Skalierung für den Layer zurück (aus Verzeichnis, kein I/O)."""
        var idx = layer * TENSORS_PER_LAYER   # PLE ist immer der erste Eintrag
        if idx < len(self.entries):
            return self.entries[idx].copy().scale
        return Float32(1.0)

    fn file_size_mb(self) -> Float64:
        return Float64(len(self.raw)) / 1e6

    fn load_time_ms(self) -> Float64:
        return Float64(Int(self.load_ns)) / 1e6

    fn validate_gemma4_shape(self) raises:
        """ShapeGuard: prüft Gemma-4-Architektur-Invarianten.
        Wirft Error falls Dimensionen inkonsistent oder unrealistisch sind."""
        var D   = self.meta.hidden
        var NH  = self.meta.n_heads
        var NKV = self.meta.n_kv_heads

        if NH == 0:
            raise Error("ShapeGuard: n_heads = 0")
        if NKV == 0:
            raise Error("ShapeGuard: n_kv_heads = 0")
        if D % NH != 0:
            raise Error("ShapeGuard: hidden (" + String(D) +
                        ") nicht durch n_heads (" + String(NH) + ") teilbar")
        if NH % NKV != 0:
            raise Error("ShapeGuard: n_heads (" + String(NH) +
                        ") nicht durch n_kv_heads (" + String(NKV) + ") teilbar")

        var head_dim = D // NH
        var kv_dim_expected = NKV * head_dim
        if self.meta.kv_dim != kv_dim_expected:
            raise Error("ShapeGuard: kv_dim=" + String(self.meta.kv_dim) +
                        " ≠ n_kv_heads*head_dim=" + String(kv_dim_expected))
        if self.meta.n_layers <= 0:
            raise Error("ShapeGuard: n_layers <= 0")
        if self.meta.ffn_dim <= 0:
            raise Error("ShapeGuard: ffn_dim <= 0")
        var expected_tensors = self.meta.n_layers * TENSORS_PER_LAYER
        if self.meta.n_tensors != expected_tensors:
            raise Error("ShapeGuard: n_tensors=" + String(self.meta.n_tensors) +
                        " ≠ n_layers*8=" + String(expected_tensors))
