from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from src.stt.constrained_decoder import constrained_ctc_beam_search
from src.stt.ngram_lm import NGramLM


def _load_wav2vec2_model(device: torch.device):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    return model, labels, bundle.sample_rate


@torch.no_grad()
def transcribe_with_constraints(
    audio_path: str,
    lm: NGramLM,
    bias_terms: list[str],
    beam_size: int,
    lm_weight: float,
    bias_weight: float,
    device: torch.device,
) -> str:
    model, labels, expected_sr = _load_wav2vec2_model(device)

    waveform, sr = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != expected_sr:
        waveform = torchaudio.functional.resample(waveform, sr, expected_sr)

    emissions, _ = model(waveform.to(device))
    log_probs = emissions.log_softmax(dim=-1).squeeze(0).cpu()
    blank_id = 0

    transcript = constrained_ctc_beam_search(
        log_probs=log_probs,
        vocab=list(labels),
        blank_id=blank_id,
        lm=lm,
        beam_size=beam_size,
        lm_weight=lm_weight,
        bias_terms=bias_terms,
        bias_weight=bias_weight,
    )
    return transcript


def build_lm_from_file(text_file: str, order: int, out_path: str | None = None) -> NGramLM:
    text = Path(text_file).read_text(encoding="utf-8")
    lm = NGramLM(order=order)
    lm.fit_text(text)
    if out_path:
        lm.save(out_path)
    return lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained STT with N-gram logit biasing.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--syllabus", required=True)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--lm-weight", type=float, default=1.2)
    parser.add_argument("--bias-weight", type=float, default=2.0)
    parser.add_argument(
        "--bias-terms",
        nargs="*",
        default=["stochastic", "cepstrum", "phoneme", "allophone"],
    )
    parser.add_argument("--output", default="outputs/transcript.txt")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    lm = build_lm_from_file(args.syllabus, args.order)
    transcript = transcribe_with_constraints(
        audio_path=args.audio,
        lm=lm,
        bias_terms=args.bias_terms,
        beam_size=args.beam_size,
        lm_weight=args.lm_weight,
        bias_weight=args.bias_weight,
        device=device,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(transcript + "\n", encoding="utf-8")
    print(transcript)


if __name__ == "__main__":
    main()
