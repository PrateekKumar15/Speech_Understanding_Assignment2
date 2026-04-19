from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.audio.preprocess import DenoiseConfig, preprocess_audio
from src.phonetics.hinglish_ipa import text_to_ipa
from src.stt.transcribe import build_lm_from_file, transcribe_with_constraints
from src.translation.dictionary import bootstrap_technical_dictionary, load_dictionary, translate_text
from src.tts.synthesis import synthesize_with_prosody
from src.utils import load_yaml, resolve_device, set_seed


def run_preprocess(cfg: dict) -> None:
    denoise_cfg = DenoiseConfig(
        sample_rate=cfg["sample_rate"],
        noise_seconds=cfg["preprocessing"]["noise_seconds"],
        n_fft=cfg["preprocessing"]["n_fft"],
        hop_length=cfg["preprocessing"]["hop_length"],
        win_length=cfg["preprocessing"]["win_length"],
        alpha=cfg["preprocessing"]["alpha"],
        floor_db=cfg["preprocessing"]["floor_db"],
    )
    preprocess_audio(
        input_path=cfg["paths"]["lecture_audio"],
        output_path=cfg["paths"]["clean_audio"],
        cfg=denoise_cfg,
    )
    print(f"[ok] preprocessed audio -> {cfg['paths']['clean_audio']}")


def run_stt(cfg: dict, device: torch.device) -> Path:
    lm = build_lm_from_file(cfg["paths"]["syllabus_text"], cfg["stt"]["lm_order"])
    text = transcribe_with_constraints(
        audio_path=cfg["paths"]["clean_audio"],
        lm=lm,
        bias_terms=cfg["stt"]["bias_terms"],
        beam_size=cfg["stt"]["beam_size"],
        lm_weight=cfg["stt"]["lm_weight"],
        bias_weight=cfg["stt"]["bias_weight"],
        device=device,
    )

    out = Path("outputs/transcript.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"[ok] transcript -> {out}")
    return out


def run_ipa(transcript_path: Path) -> Path:
    ipa = text_to_ipa(transcript_path.read_text(encoding="utf-8"))
    out = Path("outputs/transcript_ipa.txt")
    out.write_text(ipa + "\n", encoding="utf-8")
    print(f"[ok] ipa -> {out}")
    return out


def run_translation(cfg: dict, transcript_path: Path) -> Path:
    dict_path = Path(cfg["paths"]["custom_dictionary"])
    if not dict_path.exists():
        bootstrap_technical_dictionary(dict_path, lrl_tag="replace_me")

    dictionary = load_dictionary(dict_path)
    translated = translate_text(transcript_path.read_text(encoding="utf-8"), dictionary)

    out = Path("outputs/translated_lrl.txt")
    out.write_text(translated + "\n", encoding="utf-8")
    print(f"[ok] translation -> {out}")
    return out


def run_tts(cfg: dict, translated_path: Path) -> None:
    synthesize_with_prosody(
        text=translated_path.read_text(encoding="utf-8"),
        speaker_ref_wav=cfg["paths"]["student_voice_ref"],
        prosody_ref_wav=cfg["paths"]["lecture_audio"],
        output_wav=cfg["paths"]["output_cloned"],
        output_sample_rate=cfg["tts"]["output_sample_rate"],
    )
    print(f"[ok] synthesized -> {cfg['paths']['output_cloned']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech Understanding Assignment 2 pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["preprocess", "stt", "ipa", "translate", "tts", "all"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda"))

    transcript = Path("outputs/transcript.txt")
    translated = Path("outputs/translated_lrl.txt")

    if args.stage in {"preprocess", "all"}:
        run_preprocess(cfg)

    if args.stage in {"stt", "all"}:
        transcript = run_stt(cfg, device)

    if args.stage in {"ipa", "all"}:
        if not transcript.exists():
            raise FileNotFoundError("transcript not found. run stt first.")
        run_ipa(transcript)

    if args.stage in {"translate", "all"}:
        if not transcript.exists():
            raise FileNotFoundError("transcript not found. run stt first.")
        translated = run_translation(cfg, transcript)

    if args.stage in {"tts", "all"}:
        if not translated.exists():
            raise FileNotFoundError("translation not found. run translate first.")
        run_tts(cfg, translated)


if __name__ == "__main__":
    main()
