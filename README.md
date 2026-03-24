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
   - Placeholder directory for the dashboard UI.
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
   - Accepts an uploaded audio file and returns a stub analysis payload.

The analysis route is intentionally a placeholder. It establishes the request and response contract that the real model pipeline will use later.

Suggested Next Implementation Steps
1. Add audio decoding and waveform normalization.
2. Add feature extraction with log-mel spectrograms.
3. Plug in a baseline sound event classifier.
4. Persist session results for later dashboard playback.
5. Add a frontend upload flow and event timeline.

Running the Backend
1. Create a Python virtual environment.
2. Install dependencies from `backend/requirements.txt`.
3. Start the server with:

```powershell
uvicorn app.main:app --reload
```

From the `backend/` directory, the API will be available locally at `http://127.0.0.1:8000`.

Notes
1. This repo does not yet contain a trained model.
2. The backend routes are designed so the model can be added without rewriting the API shape.
