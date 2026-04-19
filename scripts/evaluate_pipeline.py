from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import mcd_db, switching_precision_with_tolerance, word_error_rate, SwitchPoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WER, MCD and switch timestamp precision.")
    parser.add_argument("--ref-en", required=True)
    parser.add_argument("--hyp-en", required=True)
    parser.add_argument("--ref-hi", required=True)
    parser.add_argument("--hyp-hi", required=True)
    parser.add_argument("--voice-ref", required=True)
    parser.add_argument("--voice-syn", required=True)
    parser.add_argument("--switch-ref", required=True, help="CSV with timestamp_sec")
    parser.add_argument("--switch-pred", required=True, help="CSV with timestamp_sec")
    parser.add_argument("--output", default="outputs/evaluation_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ref_en = Path(args.ref_en).read_text(encoding="utf-8")
    hyp_en = Path(args.hyp_en).read_text(encoding="utf-8")
    ref_hi = Path(args.ref_hi).read_text(encoding="utf-8")
    hyp_hi = Path(args.hyp_hi).read_text(encoding="utf-8")

    wer_en = word_error_rate(ref_en, hyp_en)
    wer_hi = word_error_rate(ref_hi, hyp_hi)
    mcd = mcd_db(args.voice_ref, args.voice_syn)

    ref_sw = [SwitchPoint(float(x)) for x in pd.read_csv(args.switch_ref)["timestamp_sec"].tolist()]
    pred_sw = [SwitchPoint(float(x)) for x in pd.read_csv(args.switch_pred)["timestamp_sec"].tolist()]
    sw_precision = switching_precision_with_tolerance(pred_sw, ref_sw, tolerance_ms=200.0)

    result = {
        "wer_en": wer_en,
        "wer_hi": wer_hi,
        "mcd": mcd,
        "switch_precision_200ms": sw_precision,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pd.Series(result).to_json(indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
