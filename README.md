# Speech Understanding Programming Assignment 2

End-to-end code-switched STT to low-resource-language (LRL) voice synthesis pipeline with adversarial robustness and anti-spoofing evaluation.

## 1. Repository Contents

- `pipeline.py`: Main stage runner.
- `src/audio/preprocess.py`: Spectral subtraction denoising and normalization.
- `src/lid/*`: Frame-level Hinglish LID model and training loop.
- `src/stt/*`: Constrained CTC decoding using N-gram LM bias.
- `src/phonetics/*`: Hinglish to IPA conversion layer.
- `src/translation/*`: LRL dictionary bootstrap and translation.
- `src/prosody/dtw.py`: F0 and energy extraction with DTW warping.
- `src/tts/*`: Speaker embedding extraction and conditioned synthesis scaffold.
- `src/spoofing/*`: LFCC feature extractor and anti-spoofing classifier.
- `src/adversarial/fgsm.py`: FGSM attack and SNR-constrained epsilon search.
- `scripts/*`: Evaluation and helper scripts.
- `report/report_template.md`: 10-page report skeleton (convert to PDF in IEEE/CVPR format).
- `report/implementation_note_template.md`: 1-page implementation note skeleton.

## 2. Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Data Placement

Place your files here:

- `data/samples/original_segment.wav` (10-minute lecture segment)
- `data/samples/student_voice_ref.wav` (exactly 60-second student voice)

Populate manifests:

- `data/manifests/lid_train.csv`
- `data/manifests/lid_val.csv`
- `data/manifests/cm_train.csv`
- `data/manifests/cm_val.csv`

Expected columns:

- LID CSV: `audio_path,start_sec,end_sec,lang`
- CM CSV: `audio_path,label` where label is `bona_fide` or `spoof`

## 4. Run Stages

### 4.1 Full Pipeline

```powershell
python pipeline.py --config configs/default.yaml --stage all
```

### 4.2 Individual Stages

```powershell
python pipeline.py --stage preprocess
python pipeline.py --stage stt
python pipeline.py --stage ipa
python pipeline.py --stage translate
python pipeline.py --stage tts
```

### 4.3 Train LID

```powershell
python -m src.lid.train \
  --train-manifest data/manifests/lid_train.csv \
  --val-manifest data/manifests/lid_val.csv \
  --output models/lid_best.pt
```

### 4.4 Train Anti-Spoofing Classifier

```powershell
python -m src.spoofing.cm_model \
  --train-manifest data/manifests/cm_train.csv \
  --val-manifest data/manifests/cm_val.csv \
  --output models/cm_best.pt
```

### 4.5 FGSM Robustness Search

```powershell
python scripts/run_fgsm_lid.py \
  --audio data/samples/original_segment.wav \
  --lid-ckpt models/lid_best.pt \
  --label hi
```

### 4.6 Metrics

```powershell
python scripts/evaluate_pipeline.py \
  --ref-en <path_to_ref_english_text> \
  --hyp-en <path_to_hyp_english_text> \
  --ref-hi <path_to_ref_hindi_text> \
  --hyp-hi <path_to_hyp_hindi_text> \
  --voice-ref data/samples/student_voice_ref.wav \
  --voice-syn outputs/output_LRL_cloned.wav \
  --switch-ref <path_to_ref_switch_csv> \
  --switch-pred <path_to_pred_switch_csv>
```

## 5. Deliverables Checklist

- Code: this repository.
- Audio files:
  - `original_segment.wav`
  - `student_voice_ref.wav`
  - `output_LRL_cloned.wav`
- Report: 10-page IEEE/CVPR two-column PDF.
- 1-page implementation note with one non-obvious design choice for each major task.
- Mention GitHub repository link in report and submission comment.

## 6. Notes

- This project keeps all custom logic for LID, N-gram constrained decoding, DTW prosody warping, LFCC CM, and FGSM attack in Python/PyTorch.
- Replace placeholder LRL dictionary entries in `data/lexicons/lrl_dictionary.csv` with your real 500-word parallel corpus.
- For final grading quality, you should train and tune each model on your actual dataset; this codebase is the integration-ready scaffold.

## 7. Citation Template

Please cite:

- PyTorch
- Torchaudio
- Any pretrained checkpoints/models you use
- Any external corpora/lexicons
