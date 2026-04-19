from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LID CSV manifest from TSV/CSV segment annotations.")
    parser.add_argument("--segments", required=True, help="Input CSV/TSV with columns: audio_path,start_sec,end_sec,lang")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.segments)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","

    df = pd.read_csv(path, sep=sep)
    required = {"audio_path", "start_sec", "end_sec", "lang"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["audio_path", "start_sec", "end_sec", "lang"]].copy()
    df["lang"] = df["lang"].str.lower().replace({"english": "en", "hindi": "hi"})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
