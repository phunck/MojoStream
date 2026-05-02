#!/usr/bin/env python3
"""
MojoStream Vocabulary Generator

Erzeugt vocab.msvocab – ein binäres BPE-Vokabular für den MojoStream-Tokenizer.

Binärformat (little-endian):
  Header  16 Byte:
    [4]  Magic  0x4256534D  ("MSVB")
    [4]  n_tokens
    [4]  n_merges
    [4]  reserved = 0
  Token-Einträge (n_tokens Stück):
    [4]  token_id  (uint32)
    [4]  token_len (uint32, UTF-8-Bytes)
    [?]  token-Bytes
  Merge-Einträge (n_merges Stück, in Prioritätsreihenfolge):
    [4]  left_len  (uint32)
    [?]  left-Bytes
    [4]  right_len (uint32)
    [?]  right-Bytes

Demo-Vokabular deckt alle druckbaren ASCII-Zeichen (32–126) ab
und enthält BPE-Merges für "Hallo Mojo" als Integrations-Test.
"""
import os, struct, sys

MAGIC = 0x4256534D  # "MSVB"

# ── Demo-Vokabular ──────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    (0, "<unk>"),
    (1, "<s>"),      # BOS
    (2, "</s>"),     # EOS
    (3, "<pad>"),
]

# Druckbare ASCII-Zeichen (Byte-Wert = Token-ID, Zeichen = Token-String)
# Bereich 32 (' ') bis 126 ('~')
ASCII_TOKENS = [(b, chr(b)) for b in range(32, 127)]

# BPE-Merge-Ergebnisse (Gemma-4-Stil: Wörter werden zu einzelnen Tokens)
# Folge: M+o → Mo, Mo+j → Moj, Moj+o → Mojo, etc.
MERGED_TOKENS = [
    (256, "Ha"),
    (257, "Hal"),
    (258, "Hall"),
    (259, "Hallo"),
    (260, "Mo"),
    (261, "Moj"),
    (262, "Mojo"),
    (263, " M"),
    (264, " Mo"),
    (265, " Moj"),
    (266, " Mojo"),
    (267, "ll"),     # häufig in anderen Wörtern
    (268, "al"),
    (269, "llo"),
    (270, "oj"),
]

VOCAB_TOKENS = SPECIAL_TOKENS + ASCII_TOKENS + MERGED_TOKENS

# BPE-Merge-Regeln (Index = Priorität, niedriger Index = höhere Priorität)
BPE_MERGES = [
    ("H",    "a"),     # rank 0  →  Ha
    ("Ha",   "l"),     # rank 1  →  Hal
    ("Hal",  "l"),     # rank 2  →  Hall
    ("Hall", "o"),     # rank 3  →  Hallo
    ("M",    "o"),     # rank 4  →  Mo
    ("Mo",   "j"),     # rank 5  →  Moj
    ("Moj",  "o"),     # rank 6  →  Mojo
    (" ",    "M"),     # rank 7  →  " M"  (Leerzeichen vor Majuskel)
    (" M",   "o"),     # rank 8  →  " Mo"
    (" Mo",  "j"),     # rank 9  →  " Moj"
    (" Moj", "o"),     # rank 10 →  " Mojo"
    ("l",    "l"),     # rank 11 →  ll
    ("a",    "l"),     # rank 12 →  al
    ("ll",   "o"),     # rank 13 →  llo
    ("o",    "j"),     # rank 14 →  oj
]


# ── BPE-Simulation (Python, zur Verifikation) ───────────────────────────────

def bpe_encode(text: str, merges: list) -> list:
    merge_rank = {(l, r): i for i, (l, r) in enumerate(merges)}
    chars = list(text)
    while True:
        best_rank, best_i = len(merges), -1
        for i in range(len(chars) - 1):
            key = (chars[i], chars[i + 1])
            if key in merge_rank and merge_rank[key] < best_rank:
                best_rank, best_i = merge_rank[key], i
        if best_i == -1:
            break
        chars = chars[:best_i] + [chars[best_i] + chars[best_i + 1]] + chars[best_i + 2:]
    return chars


def bpe_decode(token_strings: list) -> str:
    return "".join(token_strings)


# ── Schreib-Funktion ─────────────────────────────────────────────────────────

def write_vocab(tokens, merges, path: str):
    n_tokens = len(tokens)
    n_merges = len(merges)

    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<IIII", MAGIC, n_tokens, n_merges, 0))

        # Token-Einträge
        for token_id, token_str in tokens:
            data = token_str.encode("utf-8")
            f.write(struct.pack("<II", token_id, len(data)))
            f.write(data)

        # Merge-Einträge (in Prioritätsreihenfolge)
        for left, right in merges:
            l_bytes = left.encode("utf-8")
            r_bytes = right.encode("utf-8")
            f.write(struct.pack("<I", len(l_bytes)))
            f.write(l_bytes)
            f.write(struct.pack("<I", len(r_bytes)))
            f.write(r_bytes)

    size_kb = os.path.getsize(path) / 1024
    print(f"vocab.msvocab:  {n_tokens} Token  {n_merges} Merges  {size_kb:.1f} KB")


# ── Verifikation ─────────────────────────────────────────────────────────────

def verify(tokens, merges):
    tok_dict  = {t: tid for tid, t in tokens}
    id_to_tok = {tid: t  for tid, t in tokens}

    test_cases = ["Hallo Mojo", "Hello World", "Mojo"]
    all_ok = True
    for text in test_cases:
        token_strs = bpe_encode(text, merges)
        ids        = [tok_dict.get(t, 0) for t in token_strs]
        decoded    = bpe_decode([id_to_tok.get(i, "<unk>") for i in ids])
        ok = decoded == text
        if not ok:
            all_ok = False
        print(f'  "{text}" → {token_strs} → {ids}')
        print(f'  IDs {ids} → "{decoded}"  {"✓" if ok else "✗ FEHLGESCHLAGEN"}')
    return all_ok


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "vocab.msvocab")

    print("Erzeuge Demo-BPE-Vokabular ...")
    write_vocab(VOCAB_TOKENS, BPE_MERGES, out)

    print("\nVerifikation:")
    ok = verify(VOCAB_TOKENS, BPE_MERGES)
    if ok:
        print("\nAlle Round-Trip-Tests bestanden ✓")
    else:
        print("\nEINIGE TESTS FEHLGESCHLAGEN ✗")
        sys.exit(1)
