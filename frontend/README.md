# Frontend MVP

This directory now contains a static dashboard prototype for uploaded-audio analysis.

Files
1. `index.html`
   - Dashboard markup, playback panel, waveform preview, and result containers.
2. `styles.css`
   - Visual layout, waveform styling, playback controls, and timeline styling.
3. `app.js`
   - Upload flow, API calls, result rendering, waveform decoding, and interactive playback/timeline behavior.

Local Run
1. Start the backend API.
2. From the `frontend/` directory, serve the files locally:

```powershell
python -m http.server 3000
```

3. Open `http://127.0.0.1:3000`.

The page expects the backend API at `http://127.0.0.1:8000`.
