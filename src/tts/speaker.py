from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class SpeakerEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, emb_dim: int = 256):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels)
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(256, emb_dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 2:
            wav = wav.mean(dim=0, keepdim=True)
        mel = torch.log(self.mel(wav).clamp(min=1e-5))
        x = self.net(mel).squeeze(-1)
        x = self.proj(x)
        return torch.nn.functional.normalize(x, dim=-1)


@torch.no_grad()
def extract_speaker_embedding(audio_path: str, device: torch.device, emb_dim: int = 256) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    model = SpeakerEncoder(emb_dim=emb_dim).to(device)
    emb = model(wav.to(device))
    return emb.squeeze(0).cpu()
