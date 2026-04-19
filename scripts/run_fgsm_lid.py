from __future__ import annotations

import argparse

import torch
import torchaudio

from src.adversarial.fgsm import find_min_epsilon_for_flip
from src.lid.model import FrameLIDNet
from src.utils import LANG_TO_ID


class LIDClipWrapper(torch.nn.Module):
    def __init__(self, lid_model: FrameLIDNet):
        super().__init__()
        self.model = lid_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        clip_logits = logits.mean(dim=1)
        return clip_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FGSM robustness test on trained LID model.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--lid-ckpt", required=True)
    parser.add_argument("--label", required=True, choices=["en", "hi"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eps-start", type=float, default=0.0001)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-steps", type=int, default=25)
    parser.add_argument("--min-snr-db", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    wav, sr = torchaudio.load(args.audio)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    wav = wav[:, : 5 * 16000]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
    )(wav)
    feats = torch.log(mel.clamp(min=1e-5)).transpose(1, 2).to(device)

    lid = FrameLIDNet(input_dim=80)
    lid.load_state_dict(torch.load(args.lid_ckpt, map_location="cpu"))
    wrapper = LIDClipWrapper(lid).to(device)

    y = torch.tensor([LANG_TO_ID[args.label]], dtype=torch.long, device=device)
    eps_values = torch.linspace(args.eps_start, args.eps_end, steps=args.eps_steps).tolist()

    eps, snr = find_min_epsilon_for_flip(
        model=wrapper,
        x=feats,
        y_true=y,
        eps_values=eps_values,
        min_snr_db=args.min_snr_db,
    )

    print({"epsilon": eps, "snr_db": snr})


if __name__ == "__main__":
    main()
