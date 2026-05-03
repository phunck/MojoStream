# src/streaming/sparse_pin.mojo
#
# Tactical Sparse Pinning + Smart Skipping Prefetcher
#
# Architektur:
#   24 von 42 Layern werden beim Start einmalig in den RAM gepinnt
#   ("LIVE"); die verbleibenden 18 Layer werden via pread() on-demand
#   aus der .mojostream-Datei geladen ("STREAMED").
#
# Pinning-Strategie (3 Gruppen):
#   Anker  : 7 Full-Attention-Layer (5,11,17,23,29,35,41) —
#             wegen 4× größerer Q/O-Matrizen + kausale Abhängigkeit.
#   Brücke : First-4 (0-3) + Last-2-before-final (39,40).
#   Filler : Jeden zweiten verbleibenden Sliding-Layer → total 24.
#
# Smart Skipping Prefetcher:
#   Während ein gepinnter Layer N berechnet wird, sucht der Prefetcher
#   den nächsten NON-PINNED Layer M>N und lädt ihn in den inaktiven
#   Puffer (überspringt gepinnte Layer → kein nutzloser I/O).
#   Doppel-Buffer via Index-Flip (kein 52 MB-Copy).
#
from std.ffi import external_call

from src.streaming.mojostream import (
    MojoStreamMeta, TensorEntry,
    TENSORS_PER_LAYER, HDR_BYTES, DIR_E_BYTES,
)
from src.linalg.kernels import U8Ptr
from src.inference.gemma4_e4b import (
    E4BLayerRef, is_full_attn,
    E4B_D, E4B_N_LAYERS, E4B_FFD,
    E4B_SLIDE_Q_DIM, E4B_SLIDE_KV_DIM, E4B_SLIDE_HD,
    E4B_FULL_Q_DIM,  E4B_FULL_KV_DIM,  E4B_FULL_HD,
)

# ── Layer-Block-Größen (exakt 4096-aligned, kein Padding-Overhead) ────────────

comptime SLIDE_BYTES : Int = 45875200   # 43.8 MB pro Sliding-Layer
comptime FULL_BYTES  : Int = 52428800   # 50.0 MB pro Full-Attention-Layer
comptime STREAM_BYTES: Int = FULL_BYTES # Scratch-Buffer = max. Layer-Größe


fn layer_block_bytes(layer: Int) -> Int:
    return FULL_BYTES if is_full_attn(layer) else SLIDE_BYTES


# ── Pinning-Maske ────────────────────────────────────────────────────────────

fn build_pinned_mask() -> List[Bool]:
    """
    24 Pinning-Indices in drei Gruppen:
      1. Full-Attention Anker: 5, 11, 17, 23, 29, 35, 41   (7 Layer)
      2. Latenz-Brücke:        0, 1, 2, 3, 39, 40           (6 Layer)
      3. Filler Sliding:       4, 7, 9, 13, 15, 19, 21,
                               25, 27, 31, 33               (11 Layer)
    Total: 24 gepinnte Layer = 57 % I/O-Einsparung pro Token.
    """
    var mask = List[Bool]()
    for _ in range(E4B_N_LAYERS): mask.append(False)

    # Gruppe 1: Full-Attention-Anker
    mask[5]  = True; mask[11] = True; mask[17] = True; mask[23] = True
    mask[29] = True; mask[35] = True; mask[41] = True

    # Gruppe 2: Latenz-Brücke
    mask[0] = True; mask[1] = True; mask[2] = True; mask[3] = True
    mask[39] = True; mask[40] = True

    # Gruppe 3: Filler — jeden zweiten verbleibenden Sliding-Layer
    var candidates = List[Int]()
    for l in range(E4B_N_LAYERS):
        if not mask[l] and not is_full_attn(l):
            candidates.append(l)
    # candidates = [4, 6, 7, 8, 9, 10, 12, 13, ...]
    # Index 0,2,4,6,... → 4, 7, 9, 13, 15, 19, 21, 25, 27, 31, 33
    var count  = 0
    var pick   = True
    for idx in range(len(candidates)):
        if count >= 11: break
        if pick:
            mask[candidates[idx]] = True
            count += 1
        pick = not pick

    return mask^


fn next_non_pinned(from_layer: Int, mask: List[Bool]) -> Int:
    for l in range(from_layer + 1, E4B_N_LAYERS):
        if not mask[l]:
            return l
    return -1


fn count_pinned(mask: List[Bool]) -> Int:
    var n = 0
    for i in range(len(mask)):
        if mask[i]: n += 1
    return n


# ── Tensor-Offsets innerhalb eines Layer-Blocks ───────────────────────────────

struct LayerOffsets(Copyable, Movable):
    """Byte-Offsets der 7 Matrizen relativ zum Layer-Block-Start."""
    var q: Int; var k: Int; var v: Int; var o: Int
    var gate: Int; var up: Int; var down: Int; var total: Int

    fn __init__(out self, is_full: Bool):
        var D   = E4B_D
        var FFD = E4B_FFD
        var q_d  = E4B_FULL_Q_DIM  if is_full else E4B_SLIDE_Q_DIM
        var kv_d = E4B_FULL_KV_DIM if is_full else E4B_SLIDE_KV_DIM
        var q_sz  = D   * q_d  // 2
        var kv_sz = D   * kv_d // 2
        var f_sz  = D   * FFD  // 2
        var d_sz  = FFD * D    // 2
        self.q    = 0
        self.k    = self.q    + q_sz
        self.v    = self.k    + kv_sz
        self.o    = self.v    + kv_sz
        self.gate = self.o    + q_sz   # O und Q gleich groß
        self.up   = self.gate + f_sz
        self.down = self.up   + f_sz
        self.total = self.down + d_sz

    fn copy(self) -> Self:
        var o = Self(False)
        o.q=self.q; o.k=self.k; o.v=self.v; o.o=self.o
        o.gate=self.gate; o.up=self.up; o.down=self.down; o.total=self.total
        return o^


# ── SparsePinEngine ───────────────────────────────────────────────────────────

struct SparsePinEngine(Movable):
    """
    Speicher-Orchestrierung: gepinnte Layer aus RAM, gestreamte per pread().
    Doppel-Buffer via Index-Flip (kein Puffer-Copy).
    """
    var fd:           Int32
    var meta:         MojoStreamMeta
    var entries:      List[TensorEntry]
    var pinned_mask:  List[Bool]

    # Pinned-Layer-Puffer: pinned_bufs[l] hat Daten wenn pinned_mask[l]
    var pinned_bufs:  List[List[UInt8]]

    # Doppel-Buffer (Index-Flip statt Copy):
    #   slot 0 und slot 1 — aktiver Slot enthält den aktuellen Stream-Layer.
    var stream_buf_0: List[UInt8]
    var stream_buf_1: List[UInt8]
    var layer_in_0:   Int    # Layer-Index in slot 0 (-1 = leer)
    var layer_in_1:   Int    # Layer-Index in slot 1 (-1 = leer)
    var active_slot:  Int    # 0 oder 1: welcher Slot ist aktiv

    var io_hits:  Int        # Layer aus RAM (kein pread)
    var io_reads: Int        # Layer via pread von SSD

    fn __init__(out self, path: String) raises:
        self.meta        = MojoStreamMeta()
        self.entries     = List[TensorEntry]()
        self.pinned_mask = List[Bool]()
        self.pinned_bufs = List[List[UInt8]]()
        self.stream_buf_0 = List[UInt8]()
        self.stream_buf_1 = List[UInt8]()
        self.layer_in_0  = -1
        self.layer_in_1  = -1
        self.active_slot = 0
        self.fd          = Int32(-1)
        self.io_hits     = 0
        self.io_reads    = 0

        # ── Datei öffnen (openat(AT_FDCWD=-100, path, O_RDONLY=0)) ───────
        var path_buf = List[UInt8](capacity=len(path) + 1)
        var path_ptr = path.unsafe_ptr()
        for i in range(len(path)): path_buf.append(path_ptr.load(i))
        path_buf.append(UInt8(0))

        self.fd = external_call["openat", Int32](
            Int32(-100), path_buf.unsafe_ptr(), Int32(0)
        )
        if self.fd < Int32(0):
            raise Error("[SparsePinEngine] Nicht gefunden: " + path)

        # ── Header (128 Byte) ─────────────────────────────────────────────
        var hdr = List[UInt8]()
        hdr.resize(HDR_BYTES, 0)
        _ = external_call["pread", Int64](
            self.fd, hdr.unsafe_ptr(), Int64(HDR_BYTES), Int64(0)
        )
        var hp  = hdr.unsafe_ptr()
        if hp.bitcast[UInt32]().load(0) != UInt32(0x4F4A4F4D):
            raise Error("[SparsePinEngine] Ungültige Magic-Zahl")
        var u32 = hp.bitcast[UInt32]()
        self.meta.n_layers   = Int(u32.load(4))
        self.meta.hidden     = Int(u32.load(5))
        self.meta.kv_dim     = Int(u32.load(6))
        self.meta.ffn_dim    = Int(u32.load(7))
        self.meta.n_heads    = Int(u32.load(8))
        self.meta.n_kv_heads = Int(u32.load(9))
        var u64              = hp.bitcast[UInt64]()
        self.meta.n_tensors  = Int(u64.load(5))
        self.meta.data_start = Int(u64.load(7))

        # ── Tensor-Directory ──────────────────────────────────────────────
        var dir_sz  = self.meta.n_tensors * DIR_E_BYTES
        var dir_buf = List[UInt8]()
        dir_buf.resize(dir_sz, 0)
        _ = external_call["pread", Int64](
            self.fd, dir_buf.unsafe_ptr(), Int64(dir_sz), Int64(HDR_BYTES)
        )
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

        # ── Pinning-Maske + gepinnte Layer laden ──────────────────────────
        self.pinned_mask = build_pinned_mask()
        for layer in range(E4B_N_LAYERS):
            var lbytes = layer_block_bytes(layer)
            var buf    = List[UInt8]()
            if self.pinned_mask[layer]:
                buf.resize(lbytes, 0)
                var q_off = self.entries[layer * TENSORS_PER_LAYER + 1].data_offset
                _ = external_call["pread", Int64](
                    self.fd, buf.unsafe_ptr(), Int64(lbytes), Int64(q_off)
                )
            self.pinned_bufs.append(buf^)

        # ── Doppel-Buffer allozieren ──────────────────────────────────────
        self.stream_buf_0.resize(STREAM_BYTES, 0)
        self.stream_buf_1.resize(STREAM_BYTES, 0)

    fn _pread_into_slot(mut self, slot: Int, layer: Int) raises:
        """Liest Layer-Block via pread in den angegebenen Puffer-Slot."""
        var lbytes = layer_block_bytes(layer)
        var q_off  = self.entries[layer * TENSORS_PER_LAYER + 1].data_offset
        var n: Int64
        if slot == 0:
            n = external_call["pread", Int64](
                self.fd, self.stream_buf_0.unsafe_ptr(), Int64(lbytes), Int64(q_off)
            )
        else:
            n = external_call["pread", Int64](
                self.fd, self.stream_buf_1.unsafe_ptr(), Int64(lbytes), Int64(q_off)
            )
        if Int(n) != lbytes:
            raise Error("[SparsePinEngine] pread Layer " + String(layer)
                        + ": " + String(Int(n)) + "/" + String(lbytes))
        if slot == 0: self.layer_in_0 = layer
        else:         self.layer_in_1 = layer

    @always_inline
    fn _slot_layer(self, slot: Int) -> Int:
        return self.layer_in_0 if slot == 0 else self.layer_in_1

    fn advance(mut self, layer: Int) raises:
        """
        Smart Skipping Prefetcher:
          - Pinned → RAM-Hit; lädt nächsten non-pinned in inaktiven Slot.
          - Streamed → prüft ob Prefetch trifft (Slot-Flip O(1));
                       sonst blockierendes pread.
        """
        if self.pinned_mask[layer]:
            self.io_hits += 1
            # Prefetch: nächster non-pinned Layer M > layer
            var m = next_non_pinned(layer, self.pinned_mask)
            if m < 0: return
            var inactive = 1 - self.active_slot
            if self._slot_layer(inactive) != m:
                self._pread_into_slot(inactive, m)
        else:
            self.io_reads += 1
            if self._slot_layer(self.active_slot) == layer:
                return   # aktiver Slot hat bereits diesen Layer
            var inactive = 1 - self.active_slot
            if self._slot_layer(inactive) == layer:
                # Prefetch trifft: Slot-Flip O(1), kein Copy
                self.active_slot = inactive
            else:
                # Fallback: lade in aktiven Slot (Prefetch hat nicht gereicht)
                self._pread_into_slot(self.active_slot, layer)

    fn get_layer_ref(self, layer: Int) -> E4BLayerRef:
        """Gibt E4BLayerRef mit Zeigern in den korrekten Puffer zurück."""
        var full  = is_full_attn(layer)
        var offs  = LayerOffsets(full)
        var D     = E4B_D
        var FFD   = E4B_FFD
        var q_d   = E4B_FULL_Q_DIM  if full else E4B_SLIDE_Q_DIM
        var kv_d  = E4B_FULL_KV_DIM if full else E4B_SLIDE_KV_DIM
        var hd    = E4B_FULL_HD     if full else E4B_SLIDE_HD

        var base: UnsafePointer[UInt8, MutAnyOrigin]
        if self.pinned_mask[layer]:
            base = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
                self.pinned_bufs[layer].unsafe_ptr()
            )
        elif self.active_slot == 0:
            base = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
                self.stream_buf_0.unsafe_ptr()
            )
        else:
            base = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
                self.stream_buf_1.unsafe_ptr()
            )

        var eb = layer * TENSORS_PER_LAYER
        return E4BLayerRef(
            base + offs.q,    q_d  // 2, self.entries[eb+1].copy().scale,
            base + offs.k,    kv_d // 2, self.entries[eb+2].copy().scale,
            base + offs.v,    kv_d // 2, self.entries[eb+3].copy().scale,
            base + offs.o,    D    // 2, self.entries[eb+4].copy().scale,
            base + offs.gate, FFD  // 2, self.entries[eb+5].copy().scale,
            base + offs.up,   FFD  // 2, self.entries[eb+6].copy().scale,
            base + offs.down, D    // 2, self.entries[eb+7].copy().scale,
            self.entries[eb].copy().scale,
            full, q_d, kv_d, hd,
        )^

    fn reset_stats(mut self):
        self.io_hits = 0; self.io_reads = 0

    # ── Task 3: Memory-Orchestrierung & Logging ───────────────────────────────

    fn print_memory_map(self):
        """Tabellarische Übersicht: LIVE (RAM) vs STREAMED (SSD)."""
        var n_pin   = count_pinned(self.pinned_mask)
        var n_str   = E4B_N_LAYERS - n_pin
        var pin_mb  = Float64(0.0)
        var str_mb  = Float64(0.0)
        for i in range(E4B_N_LAYERS):
            var mb = Float64(layer_block_bytes(i)) / 1e6
            if self.pinned_mask[i]: pin_mb += mb
            else:                   str_mb  += mb

        print("══════════════════════════════════════════════════════════════")
        print("  Hybrid Memory Map  –  Gemma-4 E4B  (42 Layer)")
        print("══════════════════════════════════════════════════════════════")
        print("  #   | Typ      | Status   | Gruppe")
        print("  ----+----------+----------+----------------------------------")

        for layer in range(E4B_N_LAYERS):
            var ltype  = "FULL-ATT" if is_full_attn(layer) else "SLIDING "
            var status = "LIVE    " if self.pinned_mask[layer] else "STREAMED"
            var group  = ""
            if   layer <= 3:              group = "Brücke (First-4)"
            elif is_full_attn(layer):     group = "Anker (Full-Attention)"
            elif layer == 39 or layer == 40: group = "Brücke (Last-2)"
            elif self.pinned_mask[layer]: group = "Filler (Sliding)"
            else:                         group = "-"
            print("  ", layer, " |", ltype, "|", status, "|", group)

        print()
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │ LIVE   :", n_pin, "Layer  |", pin_mb, "MB  (RAM, vorab geladen)   │")
        print("  │ STREAM :", n_str, "Layer  |", str_mb, "MB  (SSD, on-demand pread) │")
        var buf_mb = Float64(STREAM_BYTES * 2) / 1e6
        print("  │ Buffer : 2 ×", Float64(STREAM_BYTES)/1e6, "MB =",
              buf_mb, "MB  (Doppel-Buffer)          │")
        var total_ram = pin_mb + buf_mb
        print("  ├─────────────────────────────────────────────────────────┤")
        print("  │ RAM-Gesamt:     ", total_ram, "MB =", total_ram/1e3, "GB │")
        print("  │ Referenz:        1972.6 MB  (full load)                  │")
        var ram_pct = (1972.6 - total_ram) / 1972.6 * 100.0
        print("  │ RAM-Einsparung: ", ram_pct, "%                           │")
        var io_pct = Float64(n_str) / Float64(E4B_N_LAYERS) * 100.0
        print("  │ SSD-Last/Token: ", io_pct, "% der Layer via pread         │")
        print("  │ SSD-Einsparung: ", 100.0 - io_pct, "% reduzierte SSD-Last │")
        print("  └─────────────────────────────────────────────────────────┘")
        print("══════════════════════════════════════════════════════════════")

    fn print_io_stats(self):
        var total   = self.io_hits + self.io_reads
        var hit_pct = Float64(self.io_hits) / Float64(total) * 100.0 if total > 0 else 0.0
        print("  I/O-Stats: RAM-Hits=", self.io_hits,
              "  SSD-Reads=", self.io_reads,
              "  Hit-Rate=", hit_pct, "%")
