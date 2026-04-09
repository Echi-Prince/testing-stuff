# Sound Dashboard Prototype

This repository now includes planning documents plus an MVP scaffold for a sound analysis dashboard.

Current focus:
1. Start with uploaded-audio analysis.
2. Build a backend API for detection and suppression workflows.
3. Add a frontend dashboard after the backend contracts stabilize.
4. Replace the rule-based classifier with a trained model artifact.

Repository Structure
1. `backend/`
   - FastAPI backend scaffold for health checks, configuration, and audio analysis requests.
2. `frontend/`
   - Static upload dashboard for analysis results and event timeline rendering.
3. `docs/`
   - Planning and architecture notes.
4. Root text files
   - High-level and technical build plans created earlier.

Backend MVP
The current backend scaffold provides:
1. `GET /health`
   - Verifies the service is running.
2. `GET /config`
   - Returns the configured sample rate, chunk duration, overlap, and supported classes.
3. `POST /analyze`
   - Accepts a PCM WAV file, decodes it, normalizes and resamples it to the configured target sample rate, computes basic audio features, generates log-mel spectral summaries, and returns baseline classifier detections.
4. `POST /process`
   - Accepts a PCM WAV file plus suppression settings, applies prototype class-based attenuation over matching detected spans, and returns a processed WAV preview as base64.

The current analysis route is still an MVP, but it now performs real WAV decoding, preprocessing, model-ready spectral feature generation, and baseline classifier scoring over the configured sound classes.
The spectral extraction path is `numpy`-accelerated when available, with a pure-Python fallback kept in place for minimal environments.

Suggested Next Implementation Steps
1. Collect labeled WAV clips and train the first `training/` artifact.
2. Point `trained_model_manifest_path` at the exported `manifest.json`.
3. Persist session results for later dashboard playback.
4. Improve suppression quality beyond span attenuation.
5. Add broader tests for model inference and API integration behavior.

Running the Backend
1. Fastest option from the repo root:

```powershell
.\start-backend.cmd
```

2. Or run the PowerShell script directly:

```powershell
.\start-backend.ps1
```

3. Manual fallback:

```powershell
training\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The API will be available locally at `http://127.0.0.1:8000`.

Running Tests
1. Install backend dependencies.
2. From the repository root, run:

```powershell
$env:PYTHONPATH='backend\.deps;.'
python -m unittest discover -s backend\tests -v
```

Running the Frontend
1. Fastest option from the repo root:

```powershell
.\start-frontend.cmd
```

2. Or run the PowerShell script directly:

```powershell
.\start-frontend.ps1
```

3. Manual fallback:

```powershell
cd frontend
python -m http.server 3000
```

4. Open `http://127.0.0.1:3000`.

Run Both
1. From the repo root, start both windows at once:

```powershell
.\start-dev.cmd
```

Notes
1. This repo now contains a starter trained TorchScript artifact produced from synthetic audio, but it is only for pipeline validation and not meaningful detection quality.
2. `POST /analyze` currently supports PCM WAV uploads only.
3. The backend is now configured to load `training/artifacts/latest/manifest.json` by default, and will fall back to the baseline classifier only if that artifact cannot be loaded.
4. Analysis metadata now includes preprocessing details such as target sample rate, sample count after resampling, and normalization gain.
5. The analysis response also includes compact log-mel spectrogram summary statistics for model-ready feature inspection.
6. `backend/requirements.txt` now includes `numpy` so realistic audio lengths remain fast enough for local analysis.
7. Local CORS is enabled for `http://127.0.0.1:3000` and `http://localhost:3000` so the static frontend can call the backend during development.
8. `POST /process` is a prototype suppression path that currently attenuates detected class spans rather than doing true source separation.
9. `training/` now contains an offline waveform-classifier training scaffold that exports a TorchScript model and JSON manifest for backend loading.
10. `training/generate_synthetic_dataset.py` can create a local starter WAV dataset and manifest so the training/export path can be exercised before real labeled data is available.
11. The current trained artifact was built from the synthetic starter dataset and should be treated as infrastructure validation rather than a production-quality model.
12. `training/build_real_manifest.py` and `training/real_recordings/README.md` now provide the handoff path for real labeled WAV collection and retraining.
13. Root-level launch scripts are included: `start-backend`, `start-frontend`, and `start-dev`.
