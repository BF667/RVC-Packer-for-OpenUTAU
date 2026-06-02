"""Generate OpenUtau-compatible phoneme files from S2H IPA vocabulary.

Outputs:
  phoneme_map/phonemes.txt   — one phoneme per line, line number = token ID
  phoneme_map/dsdict-zh.yaml — Chinese lyrics → IPA phoneme dictionary
  phoneme_map/dsdict-ja.yaml — Japanese lyrics → IPA phoneme dictionary
"""

import os
import sys
from pathlib import Path

# import S2H phoneme vocabulary directly (bypass data/__init__.py which needs torch)
# stub yaml since phoneme_v1.py imports it but we only need the constants
import types as _types
if "yaml" not in sys.modules:
    _ym = _types.ModuleType("yaml")
    _ym.dump = lambda *a, **k: ""
    _ym.safe_load = lambda *a, **k: {}
    sys.modules["yaml"] = _ym

import importlib.util
_PV1_PATH = Path(os.environ.get("S2H_ROOT", "score2hubert_v2")) / "src" / "data" / "phoneme_v1.py"
_spec = importlib.util.spec_from_file_location("phoneme_v1", _PV1_PATH)
_pv1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pv1)

SPECIAL_TOKENS = _pv1.SPECIAL_TOKENS
IPA_PHONEMES = _pv1.IPA_PHONEMES
PINYIN_INITIAL_TO_IPA = _pv1.PINYIN_INITIAL_TO_IPA
PINYIN_FINAL_TO_IPA = _pv1.PINYIN_FINAL_TO_IPA
OPENJTALK_TO_IPA = _pv1.OPENJTALK_TO_IPA
IPA_VOWELS = _pv1.IPA_VOWELS

OUT_DIR = Path(__file__).resolve().parents[1] / "phoneme_map"


def generate_phonemes_txt():
    """Generate phonemes.txt: one phoneme per line, index = line number."""
    all_phones = list(SPECIAL_TOKENS) + list(IPA_PHONEMES)
    # pad to 80 (same as model vocab)
    while len(all_phones) % 8 != 0:
        all_phones.append(f"<unused_{len(all_phones)}>")

    out = OUT_DIR / "phonemes.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ph in all_phones:
            f.write(ph + "\n")
    print(f"  phonemes.txt: {len(all_phones)} entries → {out}")
    return all_phones


def _ipa_list_to_str(ipa_list):
    """Join IPA segments with space."""
    return " ".join(ipa_list)


def generate_dsdict_zh():
    """Generate dsdict-zh.yaml for Chinese (pinyin → IPA).

    DiffSinger dsdict format:
      entries:
        - {grapheme: "a", phonemes: "a"}
        ...
      symbols:
        - {symbol: "a", type: vowel}
        ...
    """
    entries = []

    # initials
    for py, ipa_list in PINYIN_INITIAL_TO_IPA.items():
        entries.append({"grapheme": py, "phonemes": _ipa_list_to_str(ipa_list)})

    # finals
    for py, ipa_list in PINYIN_FINAL_TO_IPA.items():
        entries.append({"grapheme": py, "phonemes": _ipa_list_to_str(ipa_list)})

    # common full pinyin syllables (initial + final combos)
    common_initials = ["b","p","m","f","d","t","n","l","g","k","h",
                       "j","q","x","zh","ch","sh","r","z","c","s","y","w"]
    common_finals = ["a","o","e","i","u","v","ai","ei","ao","ou",
                     "an","en","ang","eng","ong","er",
                     "ia","ie","iao","iou","iu","ian","in","iang","ing","iong",
                     "ua","uo","uai","uei","ui","uan","uen","un","uang",
                     "ve","van","vn"]

    seen = set()
    for ini in common_initials:
        for fin in common_finals:
            syl = ini + fin
            if syl in seen:
                continue
            seen.add(syl)
            ini_ipa = PINYIN_INITIAL_TO_IPA.get(ini, [])
            fin_ipa = PINYIN_FINAL_TO_IPA.get(fin, [])
            if ini_ipa and fin_ipa:
                full = ini_ipa + fin_ipa
                entries.append({"grapheme": syl, "phonemes": _ipa_list_to_str(full)})

    # special tokens
    entries.append({"grapheme": "SP", "phonemes": "SP"})
    entries.append({"grapheme": "AP", "phonemes": "AP"})
    entries.append({"grapheme": "br", "phonemes": "br"})

    # symbols classification
    vowel_set = set(IPA_VOWELS)
    symbols = []
    all_ipa = set()
    for e in entries:
        for ph in e["phonemes"].split():
            all_ipa.add(ph)
    for ph in sorted(all_ipa):
        if ph in ("SP", "AP", "br"):
            symbols.append({"symbol": ph, "type": "vowel"})
        elif ph in vowel_set:
            symbols.append({"symbol": ph, "type": "vowel"})
        else:
            symbols.append({"symbol": ph, "type": "consonant"})

    # write YAML manually (avoid dependency on pyyaml)
    out = OUT_DIR / "dsdict-zh.yaml"
    with open(out, "w", encoding="utf-8") as f:
        f.write("entries:\n")
        for e in entries:
            f.write(f'  - {{grapheme: "{e["grapheme"]}", '
                    f'phonemes: [{", ".join(e["phonemes"].split())}]}}\n')
        f.write("\nsymbols:\n")
        for s in symbols:
            f.write(f'  - {{symbol: "{s["symbol"]}", type: {s["type"]}}}\n')

    print(f"  dsdict-zh.yaml: {len(entries)} entries → {out}")


def generate_dsdict_ja():
    """Generate dsdict-ja.yaml for Japanese.

    CRITICAL: DiffSingerJapanesePhonemizer calls KanaToRomaji() BEFORE
    dsdict lookup. So graphemes must be ROMAJI (what WanaKana produces),
    NOT kana. e.g. "tsu" not "つ", "shi" not "し".
    """
    entries = []

    # romaji → openjtalk phonemes → IPA
    # These romaji strings are what OpenUtau's KanaToRomaji() produces
    _ROMAJI_TO_OJT = {
        # vowels
        "a": "a", "i": "i", "u": "u", "e": "e", "o": "o",
        # k-row
        "ka": "k a", "ki": "k i", "ku": "k u", "ke": "k e", "ko": "k o",
        # s-row
        "sa": "s a", "shi": "sh i", "su": "s u", "se": "s e", "so": "s o",
        "si": "sh i",  # alternate romanization
        # t-row
        "ta": "t a", "chi": "ch i", "tsu": "ts u", "te": "t e", "to": "t o",
        "ti": "ch i", "tu": "ts u",  # alternates
        # n-row
        "na": "n a", "ni": "ny i", "nu": "n u", "ne": "n e", "no": "n o",
        # h-row
        "ha": "h a", "hi": "hy i", "fu": "f u", "he": "h e", "ho": "h o",
        "hu": "f u",  # alternate
        # m-row
        "ma": "m a", "mi": "m i", "mu": "m u", "me": "m e", "mo": "m o",
        # y-row
        "ya": "y a", "yu": "y u", "yo": "y o",
        # r-row
        "ra": "r a", "ri": "r i", "ru": "r u", "re": "r e", "ro": "r o",
        # w-row
        "wa": "w a", "wo": "o",
        # n
        "n": "N", "nn": "N",
        # voiced
        "ga": "g a", "gi": "g i", "gu": "g u", "ge": "g e", "go": "g o",
        "za": "z a", "ji": "j i", "zu": "z u", "ze": "z e", "zo": "z o",
        "zi": "j i",
        "da": "d a", "di": "j i", "du": "z u", "de": "d e", "do": "d o",
        "ba": "b a", "bi": "b i", "bu": "b u", "be": "b e", "bo": "b o",
        "pa": "p a", "pi": "p i", "pu": "p u", "pe": "p e", "po": "p o",
        # compound (拗音) — WanaKana produces these
        "kya": "ky a", "kyu": "ky u", "kyo": "ky o",
        "sha": "sh a", "shu": "sh u", "sho": "sh o",
        "cha": "ch a", "chu": "ch u", "cho": "ch o",
        "nya": "ny a", "nyu": "ny u", "nyo": "ny o",
        "hya": "hy a", "hyu": "hy u", "hyo": "hy o",
        "mya": "my a", "myu": "my u", "myo": "my o",
        "rya": "ry a", "ryu": "ry u", "ryo": "ry o",
        "gya": "gy a", "gyu": "gy u", "gyo": "gy o",
        "ja": "j a", "ju": "j u", "jo": "j o",
        "bya": "by a", "byu": "by u", "byo": "by o",
        "pya": "py a", "pyu": "py u", "pyo": "py o",
        # geminate
        "cl": "cl", "q": "cl",
    }

    for romaji, ojt_str in _ROMAJI_TO_OJT.items():
        ojt_phones = ojt_str.split()
        ipa_phones = []
        for op in ojt_phones:
            if op in OPENJTALK_TO_IPA:
                ipa_phones.extend(OPENJTALK_TO_IPA[op])
            else:
                ipa_phones.append(op)
        entries.append({"grapheme": romaji,
                        "phonemes": _ipa_list_to_str(ipa_phones)})

    # foreign-loan / edge-case graphemes (WanaKana or OpenUtau may produce these)
    _FOREIGN = {
        "fa": "f a", "fi": "f i", "fe": "f e", "fo": "f o",
        "va": "b a", "vi": "b i", "ve": "b e", "vo": "b o", "vu": "b u",
        "li": "r i", "la": "r a", "lu": "r u", "le": "r e", "lo": "r o",
        "wi": "w i", "we": "w e",
        "ye": "y e",
        "sh": "sh",
        "a_hh": "a", "e_hh": "e", "u_hh": "u", "i_hh": "i", "o_hh": "o",
        "u -": "u",
        "え'": "e",
    }
    for romaji, ojt_str in _FOREIGN.items():
        ojt_phones = ojt_str.split()
        ipa_phones = []
        for op in ojt_phones:
            if op in OPENJTALK_TO_IPA:
                ipa_phones.extend(OPENJTALK_TO_IPA[op])
            else:
                ipa_phones.append(op)
        entries.append({"grapheme": romaji,
                        "phonemes": _ipa_list_to_str(ipa_phones)})

    entries.append({"grapheme": "SP", "phonemes": "SP"})
    entries.append({"grapheme": "AP", "phonemes": "AP"})
    entries.append({"grapheme": "br", "phonemes": "br"})

    # symbols
    vowel_set = set(IPA_VOWELS)
    all_ipa = set()
    for e in entries:
        for ph in e["phonemes"].split():
            all_ipa.add(ph)
    symbols = []
    for ph in sorted(all_ipa):
        if ph in ("SP", "AP", "br"):
            symbols.append({"symbol": ph, "type": "vowel"})
        elif ph in vowel_set:
            symbols.append({"symbol": ph, "type": "vowel"})
        else:
            symbols.append({"symbol": ph, "type": "consonant"})

    out = OUT_DIR / "dsdict-ja.yaml"
    with open(out, "w", encoding="utf-8") as f:
        f.write("entries:\n")
        for e in entries:
            f.write(f'  - {{grapheme: "{e["grapheme"]}", '
                    f'phonemes: [{", ".join(e["phonemes"].split())}]}}\n')
        f.write("\nsymbols:\n")
        for s in symbols:
            f.write(f'  - {{symbol: "{s["symbol"]}", type: {s["type"]}}}\n')

    print(f"  dsdict-ja.yaml: {len(entries)} entries → {out}")


def main():
    print("Generating OpenUtau phoneme maps...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_phonemes_txt()
    generate_dsdict_zh()
    generate_dsdict_ja()
    print("Done.")


if __name__ == "__main__":
    main()
