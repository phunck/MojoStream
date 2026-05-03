# src/tests/validate_real.mojo
#
# TensorGuard Validierung für echte Gemma-4-Gewichte im .mojostream-Format.
#
# Checks:
#   1. Header-Magic + Format-Version
#   2. Architektur-Felder plausibel (n_layers, hidden, kv_dim, ffn_dim)
#   3. Engram-Extension vorhanden (Magic "ENGR" an Byte 68)
#   4. n_tensors == n_layers * 8
#   5. Alle Directory-Einträge: data_offset innerhalb Datei-Bounds
#   6. Tensor-Spot-Check (Layer 0, 20, 41): erste 64 Bytes nicht all-null
#   7. Directory-Checksum (XOR der Scale-Felder)
#
from src.streaming.mojostream import (
    MojoStreamFile, MS_Q, MS_K, MS_V, MS_O, MS_GATE, MS_UP, MS_DOWN,
    TENSORS_PER_LAYER, HDR_BYTES, DIR_E_BYTES,
)
from src.linalg.kernels import U8Ptr

# Engram-Extension: Bytes 64–127 im Haupt-Header
# (main_hdr struct = 8s + 8×I + 3×Q = 64 Byte)
comptime ENGRAM_MAGIC_OFFSET : Int = 64
comptime ENGRAM_VER_OFFSET   : Int = 68
comptime ENGRAM_KV_SH_OFFSET : Int = 72
comptime ENGRAM_SLIDE_OFFSET : Int = 76
comptime ENGRAM_FULL_OFFSET  : Int = 80
comptime ENGRAM_PLE_OFFSET   : Int = 84
comptime ENGRAM_WIN_OFFSET   : Int = 88
comptime ENGR_MAGIC : UInt32 = 0x52474E45  # "ENGR" little-endian


# ── Ergebnis-Struct ──────────────────────────────────────────────────────────

struct ValidationResult(Copyable, Movable):
    var passed:        Bool
    var n_checks:      Int
    var n_failed:      Int
    var checksum_hex:  String
    var file_size_mb:  Float64

    fn __init__(out self):
        self.passed       = False
        self.n_checks     = 0
        self.n_failed     = 0
        self.checksum_hex = String("")
        self.file_size_mb = 0.0

    fn copy(self) -> Self:
        var o = Self()
        o.passed       = self.passed
        o.n_checks     = self.n_checks
        o.n_failed     = self.n_failed
        o.checksum_hex = self.checksum_hex
        o.file_size_mb = self.file_size_mb
        return o^


# ── Hex-Hilfsfunktion (kein String-Indexing nötig) ───────────────────────────

fn nibble_to_hex(n: Int) -> String:
    if n < 10:
        return chr(48 + n)   # '0'..'9'
    return chr(55 + n)       # 'A'..'F'


fn uint32_to_hex(v: UInt32) -> String:
    var result = String("0x")
    var shift  = 28
    for _ in range(8):
        var nibble = Int((v >> UInt32(shift)) & UInt32(0xF))
        result = result + nibble_to_hex(nibble)
        shift -= 4
    return result


# ── Directory-Checksum ───────────────────────────────────────────────────────

fn compute_dir_checksum(bp: U8Ptr, n_tensors: Int) -> UInt32:
    var acc = UInt32(0)
    for i in range(n_tensors):
        var ep = bp + HDR_BYTES + i * DIR_E_BYTES + 16
        acc ^= ep.bitcast[UInt32]().load(0)
    return acc


# ── Spot-Tensor-Check ─────────────────────────────────────────────────────────

fn spot_check_tensor(ms: MojoStreamFile, layer: Int,
                     mat_type: UInt32, label: String) -> Bool:
    var tref     = ms.tensor_ptr(layer, mat_type)
    var ptr      = tref.ptr
    var all_zero = True
    var any_ff   = True
    for i in range(64):
        var b = ptr.load(i)
        if b != UInt8(0):   all_zero = False
        if b != UInt8(255): any_ff   = False
    if all_zero:
        print("  [WARN]  ", label, ": erste 64 Byte sind Null (kein Signal)")
        return False
    if any_ff:
        print("  [WARN]  ", label, ": erste 64 Byte = 0xFF (mögliche Korruption)")
        return False
    print("  [OK]    ", label, ": Tensor nicht leer  scale=", tref.scale)
    return True


# ── Hauptroutine ──────────────────────────────────────────────────────────────

fn run_validation(ms_path: String) raises -> ValidationResult:
    var res = ValidationResult()

    print("══════════════════════════════════════════════════════════════")
    print("  MojoStream TensorGuard – Real-Weight Validation")
    print("  Datei:", ms_path)
    print("══════════════════════════════════════════════════════════════")

    # ── 1. Datei laden ────────────────────────────────────────────────────
    print("\n[1] Lade .mojostream ...")
    var ms = MojoStreamFile(ms_path)
    res.file_size_mb = ms.file_size_mb()
    print("  Dateigröße:", ms.file_size_mb(), "MB  (",
          ms.file_size_mb() / 1e3, "GB)")
    print("  I/O-Zeit:  ", ms.load_time_ms(), "ms")

    var bp = ms.raw.unsafe_ptr()

    # ── 2. Magic ──────────────────────────────────────────────────────────
    print("\n[2] Header-Checks ...")
    res.n_checks += 1
    var magic_lo = bp.bitcast[UInt32]().load(0)
    if magic_lo != UInt32(0x4F4A4F4D):
        print("  [FAIL] Magic ungültig (erwartet 'MOJO...')")
        res.n_failed += 1
    else:
        print("  [OK]   Magic MOJOSTRM")

    var u32        = bp.bitcast[UInt32]()
    var n_layers   = Int(u32.load(4))
    var hidden     = Int(u32.load(5))
    var kv_dim     = Int(u32.load(6))
    var ffn_dim    = Int(u32.load(7))
    var n_heads    = Int(u32.load(8))
    var n_kv_heads = Int(u32.load(9))
    print("  n_layers=", n_layers, "  hidden=", hidden,
          "  kv_dim=", kv_dim, "  ffn_dim=", ffn_dim)
    print("  n_heads=", n_heads, "  n_kv_heads=", n_kv_heads)

    res.n_checks += 1
    if n_layers <= 0 or hidden <= 0 or ffn_dim <= 0 or n_heads <= 0:
        print("  [FAIL] Architektur-Felder ungültig")
        res.n_failed += 1
    elif n_kv_heads > 0 and kv_dim % n_kv_heads != 0:
        print("  [FAIL] kv_dim=", kv_dim,
              "nicht durch n_kv_heads=", n_kv_heads, "teilbar")
        res.n_failed += 1
    else:
        print("  [OK]   Architektur-Felder plausibel")

    # ── 3. Engram-Extension ───────────────────────────────────────────────
    print("\n[3] Engram-Header-Check (Bytes 68-127) ...")
    res.n_checks += 1
    var engram_magic = (bp + ENGRAM_MAGIC_OFFSET).bitcast[UInt32]().load(0)
    if engram_magic != ENGR_MAGIC:
        print("  [FAIL] Engram-Magic nicht gefunden (erwartet 'ENGR')")
        res.n_failed += 1
    else:
        var eu       = (bp + ENGRAM_VER_OFFSET).bitcast[UInt32]()
        var eng_ver  = Int(eu.load(0))
        var kv_sh    = Int(eu.load(1))
        var slide_kv = Int(eu.load(2))
        var full_kv  = Int(eu.load(3))
        var ple_dim  = Int(eu.load(4))
        var slide_w  = Int(eu.load(5))
        print("  [OK]   Engram v", eng_ver,
              ": kv_shared=", kv_sh,
              " kv_slide=", slide_kv,
              " kv_full=", full_kv)
        print("         ple_dim=", ple_dim,
              "  sliding_window=", slide_w)

    # ── 4. n_tensors == n_layers * 8 ─────────────────────────────────────
    print("\n[4] Tensor-Directory-Integrität ...")
    res.n_checks += 1
    var u64        = bp.bitcast[UInt64]()
    var n_tensors  = Int(u64.load(5))
    var data_start = Int(u64.load(7))
    var expected_t = n_layers * TENSORS_PER_LAYER
    if n_tensors != expected_t:
        print("  [FAIL] n_tensors=", n_tensors,
              "≠ n_layers*8=", expected_t)
        res.n_failed += 1
    else:
        print("  [OK]   n_tensors=", n_tensors, "=", n_layers, "× 8")

    # Alle Offsets innerhalb Datei-Bounds
    res.n_checks += 1
    var file_size  = len(ms.raw)
    var bad_offset = False
    for i in range(n_tensors):
        var ep     = bp + HDR_BYTES + i * DIR_E_BYTES
        var offset = Int((ep + 24).bitcast[UInt64]().load(0))
        if offset > 0 and offset >= file_size:
            print("  [FAIL] Eintrag", i, ": offset=", offset,
                  "außerhalb Datei (", file_size, "B)")
            bad_offset = True
            break
    if not bad_offset:
        print("  [OK]   Alle", n_tensors,
              "Directory-Offsets innerhalb Bounds")

    # ── 5. Directory-Checksum ─────────────────────────────────────────────
    print("\n[5] Directory-Checksum (XOR aller Scale-Felder) ...")
    res.n_checks += 1
    var chk = compute_dir_checksum(bp, n_tensors)
    res.checksum_hex = uint32_to_hex(chk)
    if chk == UInt32(0):
        print("  [WARN] Checksum=0 (alle Skalen Null?)")
        res.n_failed += 1
    else:
        print("  [OK]   Checksum=", res.checksum_hex)

    # ── 6. Tensor-Spot-Checks ─────────────────────────────────────────────
    print("\n[6] Tensor-Spot-Checks (Layer 0, 20, 41) ...")
    var spot_layers = List[Int]()
    spot_layers.append(0)
    if n_layers > 20: spot_layers.append(20)
    if n_layers > 41: spot_layers.append(41)

    for idx in range(len(spot_layers)):
        var sl = spot_layers[idx]
        res.n_checks += 1
        if not spot_check_tensor(ms, sl, MS_Q,
                "Layer " + String(sl) + " Q"):
            res.n_failed += 1
        res.n_checks += 1
        if not spot_check_tensor(ms, sl, MS_GATE,
                "Layer " + String(sl) + " Gate"):
            res.n_failed += 1

    # ── Zusammenfassung ───────────────────────────────────────────────────
    res.passed = res.n_failed == 0

    print()
    print("══════════════════════════════════════════════════════════════")
    print("  TENSORGUARD ERGEBNIS")
    print("══════════════════════════════════════════════════════════════")
    print("  Dateigröße:  ", ms.file_size_mb(), "MB  (",
          ms.file_size_mb() / 1e3, "GB)")
    print("  Checksum:    ", res.checksum_hex)
    print("  Checks:      ", res.n_checks - res.n_failed,
          "/", res.n_checks, "bestanden")
    print()
    if res.passed:
        print("  [PASS] TensorGuard: Alle Checks bestanden")
    else:
        print("  [FAIL] TensorGuard:", res.n_failed, "Check(s) fehlgeschlagen")
    print("══════════════════════════════════════════════════════════════")

    return res^


fn main() raises:
    var path = String("models/gemma4_e4b_q4.mojostream")
    _ = run_validation(path)
