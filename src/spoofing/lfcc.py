from __future__ import annotations

import math

import torch
import torchaudio


def _dct_matrix(n_mfcc: int, n_mels: int, device: torch.device) -> torch.Tensor:
    k = torch.arange(n_mfcc, device=device).float().unsqueeze(1)
    n = torch.arange(n_mels, device=device).float().unsqueeze(0)
    mat = torch.cos(math.pi / n_mels * (n + 0.5) * k)
    mat[0] *= 1.0 / math.sqrt(2.0)
    mat *= math.sqrt(2.0 / n_mels)
    return mat


def extract_lfcc(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_filter: int = 70,
    n_lfcc: int = 20,
    hop_length: int = 160,
) -> torch.Tensor:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )(waveform)

    fb = torchaudio.functional.linear_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=20.0,
        f_max=sample_rate / 2,
        n_filter=n_filter,
        sample_rate=sample_rate,
    ).to(spec.device)

    lin_energy = torch.matmul(spec.squeeze(0).transpose(0, 1), fb).clamp(min=1e-8)
    log_energy = lin_energy.log()

    dct = _dct_matrix(n_lfcc, n_filter, spec.device)
    lfcc = torch.matmul(log_energy, dct.transpose(0, 1))
    return lfcc
