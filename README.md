# Sound Dashboard Prototype

This repository now includes planning documents plus an MVP scaffold for a sound analysis dashboard.

Current focus:
1. Start with uploaded-audio analysis.
2. Build a backend API for detection and suppression workflows.
3. Add a frontend dashboard after the backend contracts stabilize.

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

The current analysis route is still an MVP, but it now performs real WAV decoding, preprocessing, model-ready spectral feature generation, and baseline classifier scoring over the configured sound classes.
The spectral extraction path is `numpy`-accelerated when available, with a pure-Python fallback kept in place for minimal environments.

Suggested Next Implementation Steps
1. Replace the rule-based baseline classifier with a trained model.
2. Persist session results for later dashboard playback.
3. Add spectrogram visualization or exported waveform thumbnails to deepen the frontend review view.
4. Add selective suppression processing for chosen classes.
5. Add broader tests for edge cases and API integration behavior.

Running the Backend
1. Create a Python virtual environment.
2. Install dependencies from `backend/requirements.txt`.
3. Start the server with:

```powershell
uvicorn app.main:app --reload
```

From the `backend/` directory, the API will be available locally at `http://127.0.0.1:8000`.

Running Tests
1. Install backend dependencies.
2. From the repository root, run:

```powershell
$env:PYTHONPATH='backend\.deps;.'
python -m unittest discover -s backend\tests -v
```

Running the Frontend
1. Start the backend API from `backend/`.
2. In a second terminal, serve the static frontend from `frontend/`:

```powershell
python -m http.server 3000
```

3. Open `http://127.0.0.1:3000/frontend/` if serving from the repo root, or `http://127.0.0.1:3000/` if serving from the `frontend/` directory.

Notes
1. This repo does not yet contain a trained model.
2. `POST /analyze` currently supports PCM WAV uploads only.
3. The backend currently uses a configurable baseline classifier name and threshold rather than a trained model artifact.
4. Analysis metadata now includes preprocessing details such as target sample rate, sample count after resampling, and normalization gain.
5. The analysis response also includes compact log-mel spectrogram summary statistics for model-ready feature inspection.
6. `backend/requirements.txt` now includes `numpy` so realistic audio lengths remain fast enough for local analysis.
7. Local CORS is enabled for `http://127.0.0.1:3000` and `http://localhost:3000` so the static frontend can call the backend during development.
