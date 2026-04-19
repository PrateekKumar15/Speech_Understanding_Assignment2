from __future__ import annotations

import argparse
from pathlib import Path

from src.phonetics.hinglish_ipa import text_to_ipa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert code-switched transcript text into IPA.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/transcript_ipa.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    ipa = text_to_ipa(text)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ipa + "\n", encoding="utf-8")
    print(ipa)


if __name__ == "__main__":
    main()
