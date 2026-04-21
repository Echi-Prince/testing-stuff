# Frontend MVP

This directory now contains a static dashboard prototype for uploaded-audio analysis and browser microphone capture.

Files
1. `index.html`
   - Dashboard markup, recording controls, training-save form, recent-session list, playback panel, suppression preview controls, waveform and spectrogram previews, and result containers.
2. `styles.css`
   - Visual layout, recording panel styling, per-class suppression controls, comparison playback controls, waveform and spectrogram styling, and timeline styling.
3. `app.js`
   - Upload flow, browser recording and WAV conversion, saved-session loading, training-set save requests, per-class suppression profile requests, original-versus-processed playback comparison, waveform and spectrogram decoding, and interactive playback/timeline behavior.

Local Run
1. From the repo root, the easiest launcher is:

```powershell
.\start-frontend.cmd
```

2. Manual fallback from the `frontend/` directory:

```powershell
python -m http.server 3000
```

3. Open `http://127.0.0.1:3000`.

The page expects the backend API at `http://127.0.0.1:8000`.
Saved sessions appear in the Recent Sessions panel when the backend session store is available.
Recorded clips can be labeled and saved into `training/real_recordings/` through the backend `POST /recordings` route.
