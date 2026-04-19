from __future__ import annotations

import argparse
from pathlib import Path

from src.translation.dictionary import bootstrap_technical_dictionary, load_dictionary, translate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate transcript to target LRL with custom dictionary.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--output", default="outputs/translated_lrl.txt")
    parser.add_argument("--bootstrap-if-missing", action="store_true")
    parser.add_argument("--lrl-tag", default="lrl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dict_path = Path(args.dictionary)

    if args.bootstrap_if_missing and not dict_path.exists():
        bootstrap_technical_dictionary(dict_path, lrl_tag=args.lrl_tag)

    dictionary = load_dictionary(dict_path)
    src_text = Path(args.input).read_text(encoding="utf-8")
    out_text = translate_text(src_text, dictionary)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text + "\n", encoding="utf-8")
    print(out_text)


if __name__ == "__main__":
    main()
