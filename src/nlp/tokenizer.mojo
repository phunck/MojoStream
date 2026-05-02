# src/nlp/tokenizer.mojo
#
# BPE-Tokenizer für MojoStream / Gemma 4
#
# Lädt ein binäres .msvocab-Vokabular (erzeugt von scripts/create_vocab.py)
# und implementiert Byte-Pair-Encoding-Inferenz sowie Dekodierung.
#
# BPE-Algorithmus (greedy, nach Priorität):
#   1. Eingabe-String → Liste einzelner Zeichen
#   2. Wiederhole: Finde das Zeichenpaar mit dem niedrigsten Merge-Rang
#      und ersetze das erste Auftreten durch das zusammengesetzte Token.
#   3. Wandle Token-Strings in IDs um (via Dict-Lookup).
#
# Mojo-0.26-Besonderheiten:
#   - Dict:  from std.collections import Dict
#   - Zeichen-Konversion: chr(Int(byte)) → String
#   - List[String]-Zugriff gibt StringSlice zurück → String(slice) nötig
#   - Dict-Lookup: try/except statt .get()
#
from std.collections import Dict

alias VOCAB_MAGIC: UInt32 = 0x4256534D  # "MSVB" little-endian

alias BOS_ID: Int = 1
alias EOS_ID: Int = 2
alias UNK_ID: Int = 0
alias PAD_ID: Int = 3


# ── BPE-Tokenizer ────────────────────────────────────────────────────────────
struct BPETokenizer(Movable):
    var id_to_tok:   List[String]        # id → Token-String
    var tok_to_id:   Dict[String, Int]   # Token-String → id
    var merge_rank:  Dict[String, Int]   # "left|right" → Rang (Priorität)
    var vocab_size:  Int
    var bos_id:      Int
    var eos_id:      Int
    var unk_id:      Int
    var pad_id:      Int
    var loaded:      Bool

    fn __init__(out self):
        self.id_to_tok  = List[String]()
        self.tok_to_id  = Dict[String, Int]()
        self.merge_rank = Dict[String, Int]()
        self.vocab_size = 0
        self.bos_id     = BOS_ID
        self.eos_id     = EOS_ID
        self.unk_id     = UNK_ID
        self.pad_id     = PAD_ID
        self.loaded     = False

    fn _set_token(mut self, id: Int, token: String):
        """Fügt ein Token ins Vokabular ein (beide Richtungen)."""
        while len(self.id_to_tok) <= id:
            self.id_to_tok.append(String("<unk>"))
        self.id_to_tok[id] = token.copy()
        self.tok_to_id[token.copy()] = id
        if id + 1 > self.vocab_size:
            self.vocab_size = id + 1

    fn load(mut self, path: String) raises:
        """Lädt ein .msvocab-Vokabular aus einer Binär-Datei."""
        var raw = List[UInt8]()
        with open(path, "r") as f:
            raw = f.read_bytes()
        var bp = raw.unsafe_ptr()

        # ── Header ──────────────────────────────────────────────────────────
        var magic    = bp.bitcast[UInt32]().load(0)
        var n_tokens = Int(bp.bitcast[UInt32]().load(1))
        var n_merges = Int(bp.bitcast[UInt32]().load(2))
        # load(3) = reserved
        if magic != VOCAB_MAGIC:
            raise Error("vocab.msvocab: ungültige Magic-Zahl")

        var pos = 16  # nach Header

        # ── Token-Einträge ───────────────────────────────────────────────────
        for _ in range(n_tokens):
            var tok_id  = Int((bp + pos).bitcast[UInt32]().load(0))
            var tok_len = Int((bp + pos + 4).bitcast[UInt32]().load(0))
            pos += 8
            var tok_str = String("")
            for k in range(tok_len):
                tok_str = tok_str + chr(Int(bp.load(pos + k)))
            pos += tok_len
            self._set_token(tok_id, tok_str)

        # ── Merge-Einträge (Reihenfolge = Priorität) ────────────────────────
        for rank in range(n_merges):
            var left_len = Int((bp + pos).bitcast[UInt32]().load(0))
            pos += 4
            var left_str = String("")
            for k in range(left_len):
                left_str = left_str + chr(Int(bp.load(pos + k)))
            pos += left_len

            var right_len = Int((bp + pos).bitcast[UInt32]().load(0))
            pos += 4
            var right_str = String("")
            for k in range(right_len):
                right_str = right_str + chr(Int(bp.load(pos + k)))
            pos += right_len

            # Merge-Schlüssel: "left|right"  (| ist kein Vokabular-Zeichen)
            self.merge_rank[left_str + "|" + right_str] = rank

        self.loaded = True

    fn encode(self, text: String) -> List[Int]:
        """Kodiert einen String in eine Liste von Token-IDs (BPE).
        Eingabe: beliebiger UTF-8-String
        Ausgabe: Token-IDs aus dem geladenen Vokabular"""
        # ── Schritt 1: Zeichen-Tokenisierung ────────────────────────────────
        var toks = List[String]()
        var ptr  = text.unsafe_ptr()
        for i in range(len(text)):
            toks.append(chr(Int(ptr.load(i))))

        # ── Schritt 2: BPE-Merges (greedy, niedrigster Rang zuerst) ─────────
        var changed = True
        while changed:
            changed = False
            var best_rank = 999_999_999
            var best_idx  = -1

            for k in range(len(toks) - 1):
                var l = String(toks[k])
                var r = String(toks[k + 1])
                var pair_key = l + "|" + r
                try:
                    var rank = self.merge_rank[pair_key]
                    if rank < best_rank:
                        best_rank = rank
                        best_idx  = k
                except:
                    pass

            if best_idx >= 0:
                var merged    = String(toks[best_idx]) + String(toks[best_idx + 1])
                var new_toks  = List[String]()
                for k in range(len(toks)):
                    if k == best_idx:
                        new_toks.append(merged)
                    elif k != best_idx + 1:
                        new_toks.append(String(toks[k]))
                toks    = new_toks^
                changed = True

        # ── Schritt 3: String-IDs nachschlagen ──────────────────────────────
        var ids = List[Int]()
        for k in range(len(toks)):
            var tok = String(toks[k])
            try:
                ids.append(self.tok_to_id[tok])
            except:
                ids.append(self.unk_id)
        return ids^

    fn decode(self, ids: List[Int]) -> String:
        """Dekodiert eine Liste von Token-IDs zurück in einen String."""
        var result = String("")
        for k in range(len(ids)):
            var id = ids[k]
            # Sondertokens überspringen
            if id == self.bos_id or id == self.eos_id or id == self.pad_id:
                continue
            if id < len(self.id_to_tok):
                result = result + String(self.id_to_tok[id])
        return result

    fn n_tokens(self) -> Int:
        return self.vocab_size

    fn token_str(self, id: Int) -> String:
        """Gibt den Token-String für eine ID zurück."""
        if id < len(self.id_to_tok):
            return String(self.id_to_tok[id])
        return String("<unk>")
