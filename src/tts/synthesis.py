from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

from src.prosody.dtw import extract_prosody, warp_contour
from src.tts.speaker import extract_speaker_embedding


class SimpleConditionedTTS(nn.Module):
    def __init__(self, vocab_size: int = 256, emb_dim: int = 256, n_mels: int = 80):
        super().__init__()
        self.text_emb = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)
        self.mel_proj = nn.Linear(emb_dim * 2 + emb_dim + 2, n_mels)

    def forward(
        self,
        token_ids: torch.Tensor,
        speaker_emb: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        x = self.text_emb(token_ids)
        x, _ = self.encoder(x)

        t = x.size(1)
        spk = speaker_emb.unsqueeze(1).expand(-1, t, -1)

        f0 = f0[:t].view(1, t, 1)
        energy = energy[:t].view(1, t, 1)
        cond = torch.cat([x, spk, f0, energy], dim=-1)
        mel = self.mel_proj(cond).transpose(1, 2)
        return mel


def text_to_ids(text: str, max_len: int = 1000) -> torch.Tensor:
    ids = [ord(ch) % 256 for ch in text][:max_len]
    if not ids:
        ids = [32]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def synthesize_with_prosody(
    text: str,
    speaker_ref_wav: str,
    prosody_ref_wav: str,
    output_wav: str,
    output_sample_rate: int = 22050,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    speaker_emb = extract_speaker_embedding(speaker_ref_wav, device=device).to(device).unsqueeze(0)

    prosody_wav, sr = torchaudio.load(prosody_ref_wav)
    if prosody_wav.size(0) > 1:
        prosody_wav = prosody_wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        prosody_wav = torchaudio.functional.resample(prosody_wav, sr, 16000)

    prosody = extract_prosody(prosody_wav.squeeze(0), sample_rate=16000)

    token_ids = text_to_ids(text).to(device)
    src_curve = torch.linspace(0.0, 1.0, steps=token_ids.size(1))
    f0_warped = warp_contour(src_curve, prosody.f0.to(src_curve.device).float())
    en_warped = warp_contour(src_curve, prosody.energy.to(src_curve.device).float())

    if f0_warped.numel() < token_ids.size(1):
        f0_warped = torch.nn.functional.pad(f0_warped, (0, token_ids.size(1) - f0_warped.numel()))
    if en_warped.numel() < token_ids.size(1):
        en_warped = torch.nn.functional.pad(en_warped, (0, token_ids.size(1) - en_warped.numel()))

    model = SimpleConditionedTTS().to(device)
    mel = model(token_ids, speaker_emb, f0_warped.to(device), en_warped.to(device))

    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256).to(device)
    mel_to_lin = torch.exp(mel[:, :513, :] if mel.size(1) >= 513 else torch.nn.functional.pad(mel, (0, 0, 0, 513 - mel.size(1))))
    wav = griffin_lim(mel_to_lin)

    if output_sample_rate != 22050:
        wav = torchaudio.functional.resample(wav, 22050, output_sample_rate)

    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav.cpu(), output_sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize LRL speech with speaker and prosody conditioning.")
    parser.add_argument("--text", required=True, help="Input text file path.")
    parser.add_argument("--speaker-ref", required=True)
    parser.add_argument("--prosody-ref", required=True)
    parser.add_argument("--output", default="outputs/output_LRL_cloned.wav")
    parser.add_argument("--sample-rate", type=int, default=22050)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.text).read_text(encoding="utf-8")
    synthesize_with_prosody(
        text=text,
        speaker_ref_wav=args.speaker_ref,
        prosody_ref_wav=args.prosody_ref,
        output_wav=args.output,
        output_sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
