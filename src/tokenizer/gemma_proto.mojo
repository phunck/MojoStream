# src/tokenizer/gemma_proto.mojo
#
# Tokenizer-Bridge für Gemma-4 E4B.
# Lädt vocab_proto.bin (erstellt von scripts/export_lm_head.py):
#   [4B n_tokens] [4B text_len, text_len×ASCII-Bytes] × n_tokens
#
# Token-ID 0 = <pad>, 1 = <eos>, 2 = <bos>, 3 = <unk>
# Token-IDs 238-493 = Byte-Fallback <0xXX> (bereits zu ASCII dekodiert)
# Token-IDs 494+ = BPE-Subwörter (▁ bereits zu Leerzeichen konvertiert)
#

fn _read_u32_le(ptr: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int:
    var b0 = Int(ptr.load(pos))
    var b1 = Int(ptr.load(pos + 1)) << 8
    var b2 = Int(ptr.load(pos + 2)) << 16
    var b3 = Int(ptr.load(pos + 3)) << 24
    return b0 | b1 | b2 | b3


struct TokenMap(Movable):
    """Compact ID→String Lookup-Tabelle, Index-adressierbar in O(1)."""
    var _raw:     List[UInt8]        # kompletter Dateiinhalt (Owner)
    var _offsets: List[Int]          # Byte-Offset des Texts für jede Token-ID
    var _lengths: List[Int]          # Textlänge für jede Token-ID
    var n_tokens: Int

    fn __init__(out self):
        self._raw     = List[UInt8]()
        self._offsets = List[Int]()
        self._lengths = List[Int]()
        self.n_tokens = 0

    fn decode(self, token_id: Int) -> String:
        """Gibt den dekodieren ASCII-Text für token_id zurück."""
        if token_id < 0 or token_id >= self.n_tokens:
            return "[?:" + String(token_id) + "]"
        var off = self._offsets[token_id]
        var ln  = self._lengths[token_id]
        var ptr = self._raw.unsafe_ptr()
        var result = String("")
        for i in range(ln):
            result = result + chr(Int(ptr.load(off + i)))
        return result

    fn decode_sequence(self, ids: List[Int]) -> String:
        var result = String("")
        for i in range(len(ids)):
            result = result + self.decode(ids[i])
        return result

    fn is_eos(self, token_id: Int) -> Bool:
        return token_id == 1   # <eos>


fn load_token_map(path: String) raises -> TokenMap:
    """Lädt vocab_proto.bin in den RAM und baut die Offset-Tabelle auf."""
    var tm = TokenMap()

    with open(path, "r") as f:
        tm._raw = f.read_bytes()

    var bp = rebind[UnsafePointer[UInt8, MutAnyOrigin]](tm._raw.unsafe_ptr())
    var n  = _read_u32_le(bp, 0)
    tm.n_tokens = n

    var pos = 4
    for _ in range(n):
        var text_len = _read_u32_le(bp, pos)
        pos += 4
        tm._offsets.append(pos)
        tm._lengths.append(text_len)
        pos += text_len

    return tm^
