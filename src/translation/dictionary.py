from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def load_dictionary(path: str | Path) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError("Dictionary CSV must have columns: source,target")
    return {str(r.source).strip().lower(): str(r.target).strip() for r in df.itertuples()}


def save_dictionary(path: str | Path, dictionary: Dict[str, str]) -> None:
    items = [{"source": k, "target": v} for k, v in sorted(dictionary.items())]
    pd.DataFrame(items).to_csv(path, index=False)


def bootstrap_technical_dictionary(path: str | Path, lrl_tag: str = "lrl") -> None:
    seed_terms = [
        "speech",
        "signal",
        "stochastic",
        "cepstrum",
        "feature",
        "phoneme",
        "allophone",
        "decoder",
        "beam",
        "language",
        "energy",
        "prosody",
        "frequency",
        "spectrum",
        "classifier",
    ]

    dictionary = {term: f"{term}_{lrl_tag}" for term in seed_terms}

    # Expand to 500 placeholders so you can replace with real LRL entries.
    for i in range(len(seed_terms), 500):
        dictionary[f"tech_term_{i+1}"] = f"tech_term_{i+1}_{lrl_tag}"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_dictionary(path, dictionary)


def translate_text(text: str, dictionary: Dict[str, str]) -> str:
    out = []
    for token in text.split():
        key = token.lower().strip(".,!?;:'\"()[]{}")
        translated = dictionary.get(key, token)
        out.append(translated)
    return " ".join(out)
