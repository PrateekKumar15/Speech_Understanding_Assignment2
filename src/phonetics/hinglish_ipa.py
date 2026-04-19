from __future__ import annotations

import re
from typing import Dict, List

DEVANAGARI_TO_IPA: Dict[str, str] = {
    "अ": "ə",
    "आ": "aː",
    "इ": "ɪ",
    "ई": "iː",
    "उ": "ʊ",
    "ऊ": "uː",
    "ए": "eː",
    "ऐ": "ɛː",
    "ओ": "oː",
    "औ": "ɔː",
    "क": "k",
    "ख": "kʰ",
    "ग": "ɡ",
    "घ": "ɡʱ",
    "च": "t͡ʃ",
    "छ": "t͡ʃʰ",
    "ज": "d͡ʒ",
    "झ": "d͡ʒʱ",
    "ट": "ʈ",
    "ठ": "ʈʰ",
    "ड": "ɖ",
    "ढ": "ɖʱ",
    "त": "t̪",
    "थ": "t̪ʰ",
    "द": "d̪",
    "ध": "d̪ʱ",
    "न": "n",
    "प": "p",
    "फ": "pʰ",
    "ब": "b",
    "भ": "bʱ",
    "म": "m",
    "य": "j",
    "र": "r",
    "ल": "l",
    "व": "ʋ",
    "स": "s",
    "श": "ʃ",
    "ह": "ɦ",
}

LATIN_TO_IPA: Dict[str, str] = {
    "a": "ə",
    "aa": "aː",
    "i": "ɪ",
    "ee": "iː",
    "u": "ʊ",
    "oo": "uː",
    "e": "e",
    "o": "o",
    "k": "k",
    "kh": "kʰ",
    "g": "ɡ",
    "gh": "ɡʱ",
    "ch": "t͡ʃ",
    "j": "d͡ʒ",
    "t": "t̪",
    "d": "d̪",
    "th": "t̪ʰ",
    "dh": "d̪ʱ",
    "n": "n",
    "p": "p",
    "ph": "pʰ",
    "b": "b",
    "bh": "bʱ",
    "m": "m",
    "r": "r",
    "l": "l",
    "v": "ʋ",
    "s": "s",
    "sh": "ʃ",
    "h": "ɦ",
}

HINGLISH_OVERRIDES: Dict[str, str] = {
    "stochastic": "stoːkəstɪk",
    "cepstrum": "sɛpstrəm",
    "phoneme": "foʊniːm",
    "spectrum": "spɛktrəm",
    "acoustic": "əkuːstɪk",
}


def _is_devanagari(token: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in token)


def _latin_token_to_ipa(token: str) -> str:
    token = token.lower()
    i = 0
    out: List[str] = []
    while i < len(token):
        if i + 1 < len(token) and token[i : i + 2] in LATIN_TO_IPA:
            out.append(LATIN_TO_IPA[token[i : i + 2]])
            i += 2
            continue
        out.append(LATIN_TO_IPA.get(token[i], token[i]))
        i += 1
    return "".join(out)


def _devanagari_token_to_ipa(token: str) -> str:
    return "".join(DEVANAGARI_TO_IPA.get(ch, ch) for ch in token)


def text_to_ipa(text: str) -> str:
    tokens = re.findall(r"[\w\u0900-\u097f]+|[^\w\s]", text, flags=re.UNICODE)
    out: List[str] = []

    for tok in tokens:
        low = tok.lower()
        if low in HINGLISH_OVERRIDES:
            out.append(HINGLISH_OVERRIDES[low])
        elif _is_devanagari(tok):
            out.append(_devanagari_token_to_ipa(tok))
        elif re.match(r"^[A-Za-z]+$", tok):
            out.append(_latin_token_to_ipa(tok))
        else:
            out.append(tok)

    return " ".join(out).replace("  ", " ").strip()


if __name__ == "__main__":
    sample = "Hindi English code switched stochastic cepstrum lecture"
    print(text_to_ipa(sample))
