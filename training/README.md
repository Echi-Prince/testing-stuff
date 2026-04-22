# Model Training Scaffold

This directory is the first trainable path away from the rule-based backend classifier.

What it contains
1. `manifest.example.jsonl`
   - Example dataset manifest format.
2. `dataset.py`
   - Manifest loader plus WAV preprocessing that reuses the backend audio pipeline.
3. `model.py`
   - Small 1D CNN for fixed-length waveform classification.
4. `train.py`
   - Offline training and TorchScript export entry point.
5. `generate_synthetic_dataset.py`
   - Local generator for a starter synthetic WAV dataset and manifest.
6. `build_real_manifest.py`
   - Builds a manifest from real class-labeled WAV recordings.
7. `convert_real_recordings_to_pcm.py`
   - Rewrites non-PCM or non-16-bit WAV files into PCM 16-bit WAV so the manifest builder and backend loader can read them.
8. `real_recordings/README.md`
   - Expected folder layout for real recordings.
9. `requirements.txt`
   - Training-only dependencies.
10. Browser-assisted collection flow
   - The frontend can now record microphone audio and save labeled WAV clips into `training/real_recordings/` through the backend `POST /recordings` endpoint.
11. Artifact version history
   - Completed training runs are archived into `training/artifacts/versions/<run_id>/`, and the active manifest is tracked in `training/artifacts/active-model.json`.

Dataset manifest format
Each line is one JSON object:

```json
{"audio_path":"data/train/speech/example.wav","label":"speech","split":"train"}
```

Required fields:
1. `audio_path`
   - Path to a PCM WAV file, relative to the manifest file.
2. `label`
   - One supported class name.

Optional fields:
1. `split`
   - `train`, `val`, or `test`

Training flow
1. Install the training dependencies.
2. Build a JSONL manifest of labeled WAV clips, or generate a starter synthetic dataset:

```powershell
python -m training.generate_synthetic_dataset --output-dir training\synthetic_data
```

3. Run:

```powershell
python -m training.train --manifest training\synthetic_data\manifest.jsonl --output-dir training\artifacts\latest
```

Real data flow
1. Place WAV recordings under `training/real_recordings/`.
   - You can now do this either manually or by recording/labelling clips from the dashboard UI.
2. Build a manifest:

```powershell
training\.venv\Scripts\python -m training.convert_real_recordings_to_pcm --source-dir training\real_recordings
training\.venv\Scripts\python -m training.build_real_manifest --source-dir training\real_recordings --output-manifest training\real_recordings\manifest.jsonl
```

3. Train on the real manifest:

```powershell
training\.venv\Scripts\python -m training.train --manifest training\real_recordings\manifest.jsonl --output-dir training\artifacts\real-v1
```

Dashboard-managed training flow
1. Save labeled WAV clips from the site into `training/real_recordings/`.
2. Use the Dataset Manager to build the manifest and start a training run.
3. The backend archives the exported `model.ts` and `manifest.json` into `training/artifacts/versions/<run_id>/`.
4. The new version is promoted to active status automatically, and older versions remain selectable as backups.

Exported files
1. `model.ts`
   - TorchScript model artifact for backend inference.
2. `manifest.json`
   - Metadata the backend loader uses to validate and load the model artifact.

Notes
1. The training scaffold currently expects fixed-length clips based on the backend chunk duration.
2. The live backend still falls back to the rule-based classifier until a valid TorchScript artifact is configured in `backend/app/config.py`.
3. The synthetic generator is only for pipeline bring-up and local experimentation. Real detections will need real labeled recordings.
4. `build_real_manifest.py` accepts either `real_recordings/<label>/*.wav` or `real_recordings/<split>/<label>/*.wav` layouts.
5. The recording ingest route writes browser-captured WAV files into that same layout so manifest generation does not need a separate conversion step.
6. The inference path now checks the active manifest first, then older archived manifests, before finally degrading to the baseline heuristic classifier.
7. `train.py` now uses inverse-frequency class weights plus weighted sampling so minority classes are not drowned out during training.
