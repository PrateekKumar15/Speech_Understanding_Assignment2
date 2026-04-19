from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


@dataclass
class DenoiseConfig:
    sample_rate: int = 16000
    noise_seconds: float = 0.5
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    alpha: float = 2.0
    floor_db: float = -35.0


def _to_mono_resampled(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def spectral_subtraction_denoise(waveform: torch.Tensor, cfg: DenoiseConfig) -> torch.Tensor:
    waveform = waveform.squeeze(0)
    window = torch.hann_window(cfg.win_length, device=waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        return_complex=True,
    )

    noise_frames = max(1, int((cfg.noise_seconds * cfg.sample_rate) / cfg.hop_length))
    noise_mag = spec[:, :noise_frames].abs().mean(dim=1, keepdim=True)

    mag = spec.abs()
    phase = torch.angle(spec)
    clean_mag = torch.clamp(mag - cfg.alpha * noise_mag, min=10 ** (cfg.floor_db / 20.0))
    clean_spec = torch.polar(clean_mag, phase)

    clean = torch.istft(
        clean_spec,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        length=waveform.numel(),
    )

    peak = clean.abs().max().clamp(min=1e-8)
    clean = 0.98 * clean / peak
    return clean.unsqueeze(0)


def preprocess_audio(input_path: str | Path, output_path: str | Path, cfg: DenoiseConfig) -> None:
    waveform, sr = torchaudio.load(str(input_path))
    waveform = _to_mono_resampled(waveform, sr, cfg.sample_rate)
    denoised = spectral_subtraction_denoise(waveform, cfg)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), denoised.cpu(), cfg.sample_rate)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Denoise and normalize lecture audio.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    cfg = DenoiseConfig(sample_rate=args.sample_rate)
    preprocess_audio(args.input, args.output, cfg)
