const API_BASE_URL = "http://127.0.0.1:8000";

const uploadForm = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const submitButton = document.querySelector("#submit-button");
const statusBanner = document.querySelector("#status-banner");
const recordStartButton = document.querySelector("#record-start-button");
const recordStopButton = document.querySelector("#record-stop-button");
const recordAnalyzeButton = document.querySelector("#record-analyze-button");
const recordSaveButton = document.querySelector("#record-save-button");
const recordingStatus = document.querySelector("#recording-status");
const recordingPreview = document.querySelector("#recording-preview");
const recordingSaveForm = document.querySelector("#recording-save-form");
const recordingLabelInput = document.querySelector("#recording-label");
const recordingSplitInput = document.querySelector("#recording-split");
const recordingSourceNameInput = document.querySelector("#recording-source-name");
const recentSessions = document.querySelector("#recent-sessions");
const detectionsCaption = document.querySelector("#detections-caption");
const sessionSummary = document.querySelector("#session-summary");
const metadataSummary = document.querySelector("#metadata-summary");
const featureSummary = document.querySelector("#feature-summary");
const spectralSummary = document.querySelector("#spectral-summary");
const detectionsContainer = document.querySelector("#detections");
const playbackPanel = document.querySelector("#playback-panel");
const timelineContainer = document.querySelector("#timeline");
const selectionSummary = document.querySelector("#selection-summary");
const SUPPRESSION_PRESET_STORAGE_KEY = "sound_dashboard_suppression_profile_v1";

let currentFile = null;
let currentAudioUrl = "";
let currentProcessedAudioUrl = "";
let currentAudioElement = null;
let currentProcessedAudioElement = null;
let currentAnalysis = null;
let currentSessionId = "";
let activeDetectionIndex = -1;
let currentWaveformPeaks = [];
let currentSpectrogramFrames = [];
let currentWaveformDurationMs = 0;
let audioContext = null;
let lastSuppressionProfile = loadSuppressionPreset();
let suppressOriginalPlaybackSync = false;
let suppressProcessedPlaybackSync = false;
let mediaRecorder = null;
let recordingStream = null;
let recordingChunks = [];
let currentRecordingBlob = null;
let currentRecordedFile = null;
let currentRecordingUrl = "";
let isRecording = false;

void checkBackendHealth();
void refreshSessionList();

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await analyzeCurrentFile();
});

recordStartButton?.addEventListener("click", () => {
  void startBrowserRecording();
});

recordStopButton?.addEventListener("click", () => {
  stopBrowserRecording();
});

recordAnalyzeButton?.addEventListener("click", () => {
  if (!currentFile) {
    setStatus("Record audio before analysis.", true);
    return;
  }
  void analyzeCurrentFile();
});

recordingSaveForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  void saveCurrentRecordingToTrainingSet();
});

fileInput?.addEventListener("change", () => {
  if (fileInput.files?.[0]) {
    currentFile = fileInput.files[0];
    currentRecordedFile = null;
    currentRecordingBlob = null;
    if (recordingPreview) {
      setEmpty(recordingPreview, "processed-player", "Record a clip to preview it here.");
    }
    setRecordingStatus("Microphone idle.", false);
  }
  updateRecordingUiState();
});

window.addEventListener("resize", () => {
  if (currentAnalysis) {
    drawWaveform(getCurrentPlaybackMs());
    drawSpectrogram(getCurrentPlaybackMs());
  }
});

async function analyzeCurrentFile() {
  const file = currentFile || fileInput.files?.[0];
  if (!file) {
    setStatus("Choose a WAV file or record audio before running analysis.", true);
    return;
  }

  submitButton.disabled = true;
  recordAnalyzeButton.disabled = true;
  setStatus(`Uploading ${file.name}...`, false);
  try {
    currentFile = file;
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${API_BASE_URL}/analyze`, { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis request failed.");
    await prepareAudioPreview(file);
    currentAnalysis = payload;
    currentSessionId = payload.session_id;
    activeDetectionIndex = payload.detections.length ? 0 : -1;
    renderAnalysis(payload);
    await refreshSessionList();
    setStatus(`Analysis complete for ${payload.filename}.`, false);
  } catch (error) {
    clearResults();
    currentFile = file;
    setStatus(error.message || "Unexpected error during analysis.", true);
  } finally {
    submitButton.disabled = false;
    updateRecordingUiState();
  }
}

async function prepareAudioPreview(file) {
  if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
  currentAudioUrl = URL.createObjectURL(file);
  currentWaveformPeaks = [];
  currentSpectrogramFrames = [];
  currentWaveformDurationMs = 0;
  try {
    const context = getAudioContext();
    const audioBuffer = await context.decodeAudioData((await file.arrayBuffer()).slice(0));
    currentWaveformPeaks = buildWaveformPeaks(audioBuffer, 720);
    currentSpectrogramFrames = buildSpectrogramFrames(audioBuffer, 256, 96);
    currentWaveformDurationMs = Math.round(audioBuffer.duration * 1000);
  } catch {
    currentWaveformPeaks = [];
    currentSpectrogramFrames = [];
  }
}

async function startBrowserRecording() {
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    setRecordingStatus("This browser does not support microphone recording.", true);
    return;
  }

  try {
    if (currentRecordingUrl) {
      URL.revokeObjectURL(currentRecordingUrl);
      currentRecordingUrl = "";
    }
    currentRecordingBlob = null;
    recordingChunks = [];
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(recordingStream);
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) recordingChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", () => {
      void finalizeRecording();
    });
    mediaRecorder.start();
    isRecording = true;
    setRecordingStatus("Recording from microphone...", false);
    updateRecordingUiState();
  } catch (error) {
    stopRecordingStream();
    setRecordingStatus(error.message || "Microphone access failed.", true);
    updateRecordingUiState();
  }
}

function stopBrowserRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;
  mediaRecorder.stop();
  isRecording = false;
  setRecordingStatus("Finishing recording...", false);
  updateRecordingUiState();
}

async function finalizeRecording() {
  try {
    const recordedBlob = new Blob(recordingChunks, { type: mediaRecorder?.mimeType || "audio/webm" });
    const wavBlob = await convertRecordedBlobToWav(recordedBlob);
    currentRecordingBlob = wavBlob;
    currentRecordedFile = new File([wavBlob], buildRecordedFilename(), { type: "audio/wav" });
    currentFile = currentRecordedFile;
    currentRecordingUrl = URL.createObjectURL(wavBlob);
    renderRecordingPreview(currentRecordingUrl, currentRecordedFile.name, wavBlob.size);
    setRecordingStatus("Recording ready. Analyze it or save it to the training set.", false);
  } catch (error) {
    currentRecordingBlob = null;
    setRecordingStatus(error.message || "Failed to convert the recording into WAV.", true);
    if (recordingPreview) {
      setEmpty(recordingPreview, "processed-player", "Record a clip to preview it here.");
    }
  } finally {
    mediaRecorder = null;
    recordingChunks = [];
    stopRecordingStream();
    isRecording = false;
    updateRecordingUiState();
  }
}

async function saveCurrentRecordingToTrainingSet() {
  if (!currentRecordedFile || !currentRecordingBlob) {
    setRecordingStatus("Record audio before saving it to the training set.", true);
    return;
  }

  recordSaveButton.disabled = true;
  try {
    const formData = new FormData();
    formData.append("file", currentRecordedFile);
    formData.append("label", recordingLabelInput.value);
    formData.append("split", recordingSplitInput.value);
    formData.append("source_name", recordingSourceNameInput.value.trim() || "browser");
    const response = await fetch(`${API_BASE_URL}/recordings`, { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to save the recording.");
    setRecordingStatus(`Saved training clip to ${payload.relative_path}.`, false);
  } catch (error) {
    setRecordingStatus(error.message || "Failed to save the recording.", true);
  } finally {
    updateRecordingUiState();
  }
}

async function convertRecordedBlobToWav(blob) {
  const context = getAudioContext();
  if (context.state === "suspended") {
    await context.resume();
  }
  const audioBuffer = await context.decodeAudioData((await blob.arrayBuffer()).slice(0));
  return encodeAudioBufferToWav(audioBuffer);
}

function encodeAudioBufferToWav(audioBuffer) {
  const channelData = audioBuffer.getChannelData(0);
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = audioBuffer.sampleRate * blockAlign;
  const dataLength = channelData.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, audioBuffer.sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let index = 0; index < channelData.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, channelData[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += bytesPerSample;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function buildRecordedFilename() {
  const stamp = new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
  return `recording-${stamp}.wav`;
}

function renderRecordingPreview(url, filename, byteCount) {
  if (!recordingPreview) return;
  recordingPreview.className = "processed-player";
  recordingPreview.innerHTML = `<strong>${escapeHtml(filename)}</strong><p class="muted">${Math.round(byteCount / 1024)} KB WAV clip ready for analysis or training ingest.</p><audio controls preload="metadata" src="${escapeHtml(url)}"></audio>`;
}

function stopRecordingStream() {
  if (recordingStream) {
    recordingStream.getTracks().forEach((track) => track.stop());
    recordingStream = null;
  }
}

function setRecordingStatus(message, isError) {
  if (!recordingStatus) return;
  recordingStatus.textContent = message;
  recordingStatus.classList.toggle("error", Boolean(isError));
}

function updateRecordingUiState() {
  const hasAnalyzableFile = Boolean(fileInput?.files?.[0] || currentFile);
  if (recordStartButton) recordStartButton.disabled = isRecording;
  if (recordStopButton) recordStopButton.disabled = !isRecording;
  if (recordAnalyzeButton) recordAnalyzeButton.disabled = isRecording || !hasAnalyzableFile;
  if (recordSaveButton) recordSaveButton.disabled = isRecording || !currentRecordingBlob;
}

function renderAnalysis(payload) {
  renderMetricList(sessionSummary, [["Status", payload.status], ["Source", payload.classifier_source], ["Fallback", payload.used_fallback ? "Yes" : "No"], ["Filename", payload.filename], ["Session", payload.session_id.slice(0, 8)], ["Detections", String(payload.detections.length)]]);
  renderMetricList(metadataSummary, [["Input Rate", `${payload.metadata.sample_rate_hz} Hz`], ["Processed Rate", `${payload.metadata.processed_sample_rate_hz} Hz`], ["Duration", `${payload.metadata.duration_ms} ms`], ["Processed Samples", String(payload.metadata.processed_sample_count)], ["Normalization Gain", payload.metadata.normalization_gain.toFixed(3)], ["Resampled", payload.metadata.was_resampled ? "Yes" : "No"]]);
  renderMetricList(featureSummary, [["RMS", payload.features.rms.toFixed(6)], ["Peak", payload.features.peak_amplitude.toFixed(6)], ["Zero Crossings", payload.features.zero_crossing_rate.toFixed(6)], ["Activity Ratio", payload.features.dominant_activity_ratio.toFixed(6)]]);
  renderMetricList(spectralSummary, [["Frames", String(payload.spectral_features.frame_count)], ["Mel Bins", String(payload.spectral_features.mel_bin_count)], ["Mean dB", payload.spectral_features.mean_db.toFixed(3)], ["Dynamic Range", payload.spectral_features.dynamic_range_db.toFixed(3)]]);
  renderPlayback(payload);
  renderInteractiveViews();
  if (detectionsCaption) {
    detectionsCaption.textContent = payload.used_fallback
      ? `Highest-confidence labels returned by ${payload.classifier_source} after the trained model returned no classes.`
      : `Highest-confidence labels returned by ${payload.classifier_source}.`;
  }
}

function renderPlayback(payload) {
  playbackPanel.className = "playback-panel";
  playbackPanel.innerHTML = `
    <div class="player-shell">
      <div class="player-meta">
        <div><h3>${escapeHtml(payload.filename)}</h3><p class="muted">Tune suppression by detected class and compare original versus processed playback.</p></div>
        <div class="player-actions"><button id="play-toggle" type="button">Play</button><button id="jump-selection" type="button" class="subtle-button">Jump To Selection</button></div>
      </div>
      <div class="suppression-panel">
        <div class="section-head"><h3>Suppression Preview</h3><p class="muted">Per-class attenuation profile</p></div>
        <div class="suppression-grid">
          <div id="suppression-choices" class="suppression-choices"></div>
          <div class="suppression-actions">
            <button id="process-button" type="button">Render Suppressed Audio</button>
            <div id="processed-audio-summary" class="processed-player empty-state">No processed audio yet.</div>
          </div>
        </div>
      </div>
      <div class="waveform-panel">
        <div class="preview-grid">
          <div><div class="waveform-meta"><strong>Waveform Preview</strong><p class="muted">${formatMilliseconds(currentWaveformDurationMs || payload.metadata.duration_ms)} total</p></div><canvas id="waveform-canvas" class="waveform-canvas" width="960" height="180"></canvas></div>
          <div><div class="waveform-meta"><strong>Spectrogram Preview</strong><p class="muted">Browser-side STFT snapshot</p></div><canvas id="spectrogram-canvas" class="spectrogram-canvas" width="960" height="180"></canvas></div>
        </div>
      </div>
      <div class="preview-grid">
        <div><div class="waveform-meta"><strong>Original Audio</strong><p class="muted">Source upload</p></div><audio id="audio-player" preload="metadata" controls src="${escapeHtml(currentAudioUrl)}"></audio></div>
        <div><div class="waveform-meta"><strong>Processed Audio</strong><p class="muted">Suppression preview</p></div><div id="processed-audio-shell" class="processed-player empty-state">No processed audio yet.</div></div>
      </div>
      <input id="timeline-scrubber" class="scrubber" type="range" min="0" max="${payload.metadata.duration_ms}" value="0" step="1" />
      <div class="time-row"><span id="current-time">0:00.000</span><span>${formatMilliseconds(payload.metadata.duration_ms)}</span></div>
    </div>`;

  currentAudioElement = playbackPanel.querySelector("#audio-player");
  currentProcessedAudioElement = null;
  const playToggle = playbackPanel.querySelector("#play-toggle");
  const jumpSelection = playbackPanel.querySelector("#jump-selection");
  const scrubber = playbackPanel.querySelector("#timeline-scrubber");
  const currentTime = playbackPanel.querySelector("#current-time");
  const waveformCanvas = playbackPanel.querySelector("#waveform-canvas");
  const spectrogramCanvas = playbackPanel.querySelector("#spectrogram-canvas");
  renderSuppressionChoices(playbackPanel.querySelector("#suppression-choices"), payload.detections);

  currentAudioElement.addEventListener("timeupdate", () => {
    const currentMs = getCurrentPlaybackMs();
    scrubber.value = String(currentMs);
    currentTime.textContent = formatMilliseconds(currentMs);
    syncProcessedAudioPosition();
    updateTimelinePlayhead(currentMs, payload.metadata.duration_ms);
    drawWaveform(currentMs);
    drawSpectrogram(currentMs);
  });
  currentAudioElement.addEventListener("loadedmetadata", () => {
    scrubber.max = String(Math.round(currentAudioElement.duration * 1000) || payload.metadata.duration_ms);
    drawWaveform(getCurrentPlaybackMs());
    drawSpectrogram(getCurrentPlaybackMs());
  });
  currentAudioElement.addEventListener("play", () => {
    playToggle.textContent = "Pause";
    if (suppressProcessedPlaybackSync) {
      suppressProcessedPlaybackSync = false;
      return;
    }
    syncProcessedAudioPlaybackState(true);
  });
  currentAudioElement.addEventListener("pause", () => {
    playToggle.textContent = "Play";
    if (suppressProcessedPlaybackSync) {
      suppressProcessedPlaybackSync = false;
      return;
    }
    syncProcessedAudioPlaybackState(false);
  });
  currentAudioElement.addEventListener("ended", () => syncProcessedAudioPlaybackState(false));
  playToggle.addEventListener("click", () => currentAudioElement.paused ? currentAudioElement.play() : currentAudioElement.pause());
  jumpSelection.addEventListener("click", () => { if (activeDetectionIndex >= 0) focusDetection(activeDetectionIndex, false); });
  scrubber.addEventListener("input", () => handleSeek(Number(scrubber.value), payload.metadata.duration_ms));
  waveformCanvas.addEventListener("click", (event) => handlePreviewSeek(event, waveformCanvas, payload.metadata.duration_ms));
  spectrogramCanvas.addEventListener("click", (event) => handlePreviewSeek(event, spectrogramCanvas, payload.metadata.duration_ms));
  playbackPanel.querySelector("#process-button").addEventListener("click", async () => {
    try {
      setStatus("Rendering suppression preview...", false);
      const suppressionProfile = getSuppressionProfile();
      saveSuppressionPreset(suppressionProfile);
      renderProcessedAudio(await requestProcessedAudio({ file: currentFile, suppressionProfile }));
      await refreshSessionList();
      setStatus("Suppression preview ready.", false);
    } catch (error) {
      renderProcessedAudio(null);
      setStatus(error.message || "Suppression preview failed.", true);
    }
  });

  drawWaveform(0);
  drawSpectrogram(0);
}

function renderInteractiveViews() {
  renderDetections(currentAnalysis.detections);
  renderTimeline(currentAnalysis.detections, currentAnalysis.metadata.duration_ms);
  renderSelectionSummary(currentAnalysis.detections, currentAnalysis.metadata.duration_ms);
  updateTimelinePlayhead(getCurrentPlaybackMs(), currentAnalysis.metadata.duration_ms);
  drawWaveform(getCurrentPlaybackMs());
  drawSpectrogram(getCurrentPlaybackMs());
}

function renderMetricList(target, entries) {
  target.classList.remove("empty-state");
  target.innerHTML = entries.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
}

function renderDetections(detections) {
  if (!detections.length) return void setEmpty(detectionsContainer, "detections", "No detections returned.");
  detectionsContainer.className = "detections";
  detectionsContainer.innerHTML = detections.map((detection, index) => `<button type="button" class="chip is-button ${index === activeDetectionIndex ? "is-active" : ""}" data-detection-index="${index}"><h3>${escapeHtml(detection.label)}</h3><p>Confidence ${Number(detection.confidence).toFixed(3)}</p><p>${detection.start_ms} ms to ${detection.end_ms} ms</p></button>`).join("");
  detectionsContainer.querySelectorAll("[data-detection-index]").forEach((element) => element.addEventListener("click", () => focusDetection(Number(element.dataset.detectionIndex), true)));
}

function renderTimeline(detections, durationMs) {
  if (!detections.length || durationMs <= 0) return void setEmpty(timelineContainer, "timeline", "No timeline available.");
  timelineContainer.className = "timeline";
  timelineContainer.innerHTML = detections.map((detection, index) => {
    const left = (detection.start_ms / durationMs) * 100;
    const width = ((detection.end_ms - detection.start_ms) / durationMs) * 100;
    return `<div class="timeline-row"><div class="timeline-label">${escapeHtml(detection.label)}</div><button type="button" class="timeline-track ${index === activeDetectionIndex ? "is-active" : ""}" data-timeline-index="${index}"><div class="timeline-bar" style="left:${left}%; width:${Math.max(width, 2)}%;"></div><div class="timeline-playhead" style="left:0%;"></div></button></div>`;
  }).join("");
  timelineContainer.querySelectorAll("[data-timeline-index]").forEach((element) => element.addEventListener("click", () => focusDetection(Number(element.dataset.timelineIndex), true)));
}

function renderSelectionSummary(detections, durationMs) {
  if (!detections.length || activeDetectionIndex < 0) return void setEmpty(selectionSummary, "selection-summary", "No event selected.");
  const detection = detections[activeDetectionIndex];
  const share = (((detection.end_ms - detection.start_ms) / durationMs) * 100).toFixed(1);
  selectionSummary.className = "selection-summary";
  selectionSummary.innerHTML = `<strong>${escapeHtml(detection.label)}</strong><p>Confidence ${Number(detection.confidence).toFixed(3)}. Span ${detection.start_ms} ms to ${detection.end_ms} ms, covering ${share}% of the clip.</p>`;
}

function renderSuppressionChoices(container, detections) {
  const labels = [...new Set(detections.map((d) => d.label))];
  if (!labels.length) return void setEmpty(container, "suppression-choices", "No detected classes available for suppression.");
  container.innerHTML = labels.map((label, index) => {
    const presetValue = lastSuppressionProfile[label];
    const sliderValue = typeof presetValue === "number" ? presetValue.toFixed(2) : settingsDefaultAttenuation();
    const checked = typeof presetValue === "number" ? presetValue < 1.0 : index === 0;
    return `<label class="choice-chip"><div class="choice-head"><span class="choice-label"><input type="checkbox" value="${escapeHtml(label)}" ${checked ? "checked" : ""} /><span>${escapeHtml(label)}</span></span><span class="choice-value" data-choice-value="${escapeHtml(label)}">${sliderValue}</span></div><input type="range" min="0" max="1" step="0.05" value="${sliderValue}" data-choice-slider="${escapeHtml(label)}" /></label>`;
  }).join("");
  container.querySelectorAll("[data-choice-slider]").forEach((slider) => slider.addEventListener("input", () => {
    const valueNode = container.querySelector(`[data-choice-value="${escapeSelector(slider.dataset.choiceSlider)}"]`);
    if (valueNode) valueNode.textContent = Number(slider.value).toFixed(2);
  }));
}

function getSuppressionProfile() {
  const profile = {};
  playbackPanel.querySelectorAll("#suppression-choices .choice-chip").forEach((choice) => {
    const checkbox = choice.querySelector('input[type="checkbox"]');
    const slider = choice.querySelector('input[type="range"]');
    if (checkbox?.checked && slider) profile[checkbox.value] = Number(slider.value);
  });
  return profile;
}

async function requestProcessedAudio({ file, suppressionProfile }) {
  if (!file) throw new Error("Upload a file before requesting suppression.");
  if (!Object.keys(suppressionProfile).length) throw new Error("Select at least one detected class for suppression.");
  const formData = new FormData();
  formData.append("file", file);
  formData.append("suppression_profile", JSON.stringify(suppressionProfile));
  if (currentSessionId) formData.append("session_id", currentSessionId);
  const response = await fetch(`${API_BASE_URL}/process`, { method: "POST", body: formData });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Suppression processing failed.");
  return payload;
}

function renderProcessedAudio(payload) {
  const summary = playbackPanel.querySelector("#processed-audio-summary");
  const shell = playbackPanel.querySelector("#processed-audio-shell");
  if (currentProcessedAudioUrl) URL.revokeObjectURL(currentProcessedAudioUrl);
  currentProcessedAudioUrl = "";
  currentProcessedAudioElement = null;
  if (!payload) {
    setEmpty(summary, "processed-player", "No processed audio yet.");
    setEmpty(shell, "processed-player", "No processed audio yet.");
    return;
  }
  const blob = new Blob([decodeBase64(payload.processed_audio.wav_base64)], { type: "audio/wav" });
  currentProcessedAudioUrl = URL.createObjectURL(blob);
  currentSessionId = payload.session_id || currentSessionId;
  summary.className = "processed-player";
  summary.innerHTML = `<strong>Processed Audio Preview</strong><p class="muted">Detection source: ${escapeHtml(payload.classifier_source)}${payload.used_fallback ? " (baseline fallback)" : ""}</p><p class="muted">Suppressed classes: ${escapeHtml(payload.processed_audio.suppressed_classes.join(", ") || "none")}</p><p class="muted">Profile: ${escapeHtml(formatSuppressionProfile(payload.processed_audio.class_attenuation_factors))}</p>`;
  shell.className = "processed-player";
  shell.innerHTML = `<audio id="processed-audio-player" controls preload="metadata" src="${escapeHtml(currentProcessedAudioUrl)}"></audio><div class="compare-actions"><button type="button" id="sync-processed">Match Original Position</button><button type="button" id="play-both" class="subtle-button">Play Both</button></div>`;
  currentProcessedAudioElement = shell.querySelector("#processed-audio-player");
  shell.querySelector("#sync-processed").addEventListener("click", () => syncProcessedAudioPosition(true));
  shell.querySelector("#play-both").addEventListener("click", async () => {
    if (!currentProcessedAudioElement) return;
    syncProcessedAudioPosition(true);
    suppressProcessedPlaybackSync = true;
    if (currentAudioElement) await currentAudioElement.play();
    suppressOriginalPlaybackSync = true;
    await currentProcessedAudioElement.play();
  });
  currentProcessedAudioElement.addEventListener("play", () => {
    if (suppressOriginalPlaybackSync) {
      suppressOriginalPlaybackSync = false;
      return;
    }
    syncProcessedAudioPosition(true);
    if (currentAudioElement?.paused) {
      suppressProcessedPlaybackSync = true;
      void currentAudioElement.play().catch(() => {
        suppressProcessedPlaybackSync = false;
      });
    }
  });
  currentProcessedAudioElement.addEventListener("pause", () => {
    if (suppressOriginalPlaybackSync) {
      suppressOriginalPlaybackSync = false;
      return;
    }
    if (currentAudioElement && !currentAudioElement.paused) {
      suppressProcessedPlaybackSync = true;
      currentAudioElement.pause();
    }
  });
  currentProcessedAudioElement.addEventListener("ended", () => {
    if (currentAudioElement && !currentAudioElement.paused) {
      suppressProcessedPlaybackSync = true;
      currentAudioElement.pause();
    }
  });
}

function focusDetection(index, shouldPlay) {
  if (!currentAnalysis || index < 0 || index >= currentAnalysis.detections.length) return;
  activeDetectionIndex = index;
  handleSeek(currentAnalysis.detections[index].start_ms, currentAnalysis.metadata.duration_ms);
  renderInteractiveViews();
  if (shouldPlay && currentAudioElement) void currentAudioElement.play();
}

function handleSeek(targetMs, durationMs) {
  seekAudio(targetMs);
  updateTimelinePlayhead(targetMs, durationMs);
  drawWaveform(targetMs);
  drawSpectrogram(targetMs);
}

function handlePreviewSeek(event, canvas, durationMs) {
  const rect = canvas.getBoundingClientRect();
  handleSeek(Math.max(0, Math.min(durationMs, Math.round(durationMs * ((event.clientX - rect.left) / rect.width)))), durationMs);
}

function updateTimelinePlayhead(currentMs, durationMs) {
  if (durationMs <= 0) return;
  const percent = (currentMs / durationMs) * 100;
  timelineContainer.querySelectorAll(".timeline-playhead").forEach((playhead) => { playhead.style.left = `${Math.max(0, Math.min(percent, 100))}%`; });
}

function drawWaveform(currentMs) { drawPreview("#waveform-canvas", currentMs, false); }
function drawSpectrogram(currentMs) { drawPreview("#spectrogram-canvas", currentMs, true); }

function drawPreview(selector, currentMs, useSpectrogram) {
  const canvas = playbackPanel.querySelector(selector);
  if (!canvas) return;
  const context = canvas.getContext("2d");
  if (!context) return;
  const width = canvas.clientWidth || 960;
  const height = canvas.clientHeight || 180;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const durationMs = currentWaveformDurationMs || currentAnalysis?.metadata.duration_ms || 1;
  const selection = getActiveDetection();
  if (useSpectrogram) drawSpectrogramBody(context, width, height); else drawWaveformBody(context, width, height);
  if (selection) {
    const left = (selection.start_ms / durationMs) * width;
    const bandWidth = ((selection.end_ms - selection.start_ms) / durationMs) * width;
    context.fillStyle = useSpectrogram ? "rgba(255,255,255,0.16)" : "rgba(0,109,119,0.14)";
    context.fillRect(left, 0, Math.max(bandWidth, 2), height);
  }
  const x = (currentMs / durationMs) * width;
  context.strokeStyle = useSpectrogram ? "rgba(255,255,255,0.95)" : "rgba(27,28,29,0.9)";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(x, 0);
  context.lineTo(x, height);
  context.stroke();
}

function drawWaveformBody(context, width, height) {
  context.fillStyle = "rgba(255,255,255,0.4)";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(0,109,119,0.92)";
  context.lineWidth = 1.2;
  context.beginPath();
  if (currentWaveformPeaks.length) {
    const mid = height / 2;
    const stepX = width / currentWaveformPeaks.length;
    currentWaveformPeaks.forEach((peak, index) => {
      const x = index * stepX;
      const amplitude = peak * (height * 0.42);
      context.moveTo(x, mid - amplitude);
      context.lineTo(x, mid + amplitude);
    });
  } else {
    context.moveTo(0, height / 2);
    context.lineTo(width, height / 2);
  }
  context.stroke();
}

function drawSpectrogramBody(context, width, height) {
  if (!currentSpectrogramFrames.length) {
    context.fillStyle = "rgba(255,255,255,0.5)";
    context.fillRect(0, 0, width, height);
    return;
  }
  const columnWidth = width / currentSpectrogramFrames.length;
  currentSpectrogramFrames.forEach((frame, frameIndex) => {
    const x = frameIndex * columnWidth;
    const rowHeight = height / frame.length;
    frame.forEach((value, rowIndex) => {
      context.fillStyle = spectrogramColor(value);
      context.fillRect(x, height - ((rowIndex + 1) * rowHeight), Math.ceil(columnWidth + 1), Math.ceil(rowHeight + 1));
    });
  });
}

function seekAudio(targetMs) {
  if (!currentAudioElement) return;
  currentAudioElement.currentTime = targetMs / 1000;
  const scrubber = playbackPanel.querySelector("#timeline-scrubber");
  const currentTime = playbackPanel.querySelector("#current-time");
  if (scrubber) scrubber.value = String(targetMs);
  if (currentTime) currentTime.textContent = formatMilliseconds(targetMs);
}

function getCurrentPlaybackMs() { return currentAudioElement ? Math.round(currentAudioElement.currentTime * 1000) : 0; }
function getActiveDetection() { return currentAnalysis && activeDetectionIndex >= 0 ? currentAnalysis.detections[activeDetectionIndex] || null : null; }

function buildWaveformPeaks(audioBuffer, bucketCount) {
  const data = audioBuffer.getChannelData(0);
  const bucketSize = Math.max(1, Math.floor(data.length / bucketCount));
  const peaks = [];
  for (let bucketIndex = 0; bucketIndex < bucketCount; bucketIndex += 1) {
    const start = bucketIndex * bucketSize;
    const end = Math.min(start + bucketSize, data.length);
    let peak = 0;
    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) peak = Math.max(peak, Math.abs(data[sampleIndex]));
    peaks.push(peak);
  }
  return peaks;
}

function buildSpectrogramFrames(audioBuffer, fftSize, frameCount) {
  const data = audioBuffer.getChannelData(0);
  const hopSize = Math.max(1, Math.floor(data.length / frameCount));
  const window = buildHannWindow(fftSize);
  return Array.from({ length: frameCount }, (_, frameIndex) => {
    const real = new Array(fftSize).fill(0);
    const imag = new Array(fftSize).fill(0);
    const start = frameIndex * hopSize;
    for (let sampleIndex = 0; sampleIndex < fftSize; sampleIndex += 1) real[sampleIndex] = (data[start + sampleIndex] || 0) * window[sampleIndex];
    fftInPlace(real, imag);
    const bins = [];
    for (let binIndex = 0; binIndex < fftSize / 2; binIndex += 2) bins.push(Math.min(1, Math.sqrt((real[binIndex] ** 2) + (imag[binIndex] ** 2)) * 6));
    return bins;
  });
}

function buildHannWindow(size) { return Array.from({ length: size }, (_, index) => 0.5 - (0.5 * Math.cos((2 * Math.PI * index) / (size - 1)))); }

function fftInPlace(real, imag) {
  const size = real.length;
  let halfSize = 1;
  for (let index = 0, j = 0; index < size; index += 1) {
    if (j > index) { [real[index], real[j]] = [real[j], real[index]]; [imag[index], imag[j]] = [imag[j], imag[index]]; }
    let bit = size >> 1;
    while (bit >= 1 && j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
  }
  while (halfSize < size) {
    const phaseStepReal = Math.cos(-Math.PI / halfSize);
    const phaseStepImag = Math.sin(-Math.PI / halfSize);
    for (let fftStep = 0; fftStep < size; fftStep += halfSize * 2) {
      let phaseReal = 1;
      let phaseImag = 0;
      for (let index = 0; index < halfSize; index += 1) {
        const off = fftStep + index;
        const match = off + halfSize;
        const tr = (phaseReal * real[match]) - (phaseImag * imag[match]);
        const ti = (phaseReal * imag[match]) + (phaseImag * real[match]);
        real[match] = real[off] - tr;
        imag[match] = imag[off] - ti;
        real[off] += tr;
        imag[off] += ti;
        const nextPhaseReal = (phaseReal * phaseStepReal) - (phaseImag * phaseStepImag);
        phaseImag = (phaseReal * phaseStepImag) + (phaseImag * phaseStepReal);
        phaseReal = nextPhaseReal;
      }
    }
    halfSize *= 2;
  }
}

function spectrogramColor(value) {
  const bounded = Math.max(0, Math.min(1, value));
  return `hsl(${210 - (bounded * 190)} ${72 + (bounded * 18)}% ${12 + (bounded * 55)}%)`;
}

function getAudioContext() {
  if (!audioContext) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error("This browser does not support AudioContext for preview rendering.");
    audioContext = new AudioContextClass();
  }
  return audioContext;
}

function setEmpty(target, className, message) {
  target.className = `${className} empty-state`;
  target.textContent = message;
}

function setStatus(message, isError) {
  statusBanner.textContent = message;
  statusBanner.classList.toggle("error", Boolean(isError));
}

function formatMilliseconds(value) {
  const total = Math.max(0, Math.round(value));
  return `${Math.floor(total / 60000)}:${String(Math.floor((total % 60000) / 1000)).padStart(2, "0")}.${String(total % 1000).padStart(3, "0")}`;
}

function escapeHtml(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function escapeSelector(value) {
  return window.CSS?.escape ? window.CSS.escape(value) : String(value).replaceAll('"', '\\"');
}

function decodeBase64(value) {
  const binary = window.atob(value);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function loadSuppressionPreset() {
  try {
    const rawValue = window.localStorage.getItem(SUPPRESSION_PRESET_STORAGE_KEY);
    if (!rawValue) return {};
    const parsedValue = JSON.parse(rawValue);
    if (!parsedValue || typeof parsedValue !== "object" || Array.isArray(parsedValue)) return {};
    return Object.fromEntries(
      Object.entries(parsedValue)
        .filter(([label, factor]) => label && Number.isFinite(Number(factor)))
        .map(([label, factor]) => [label, Math.max(0, Math.min(1, Number(factor)))])
    );
  } catch {
    return {};
  }
}

function saveSuppressionPreset(profile) {
  lastSuppressionProfile = { ...profile };
  try {
    window.localStorage.setItem(SUPPRESSION_PRESET_STORAGE_KEY, JSON.stringify(lastSuppressionProfile));
  } catch {
    // Ignore storage failures and keep the current session working.
  }
}

function syncProcessedAudioPosition(forceSync = false) {
  if (!currentAudioElement || !currentProcessedAudioElement) return;
  const originalTime = Number(currentAudioElement.currentTime || 0);
  const processedTime = Number(currentProcessedAudioElement.currentTime || 0);
  if (forceSync || Math.abs(originalTime - processedTime) > 0.08) {
    currentProcessedAudioElement.currentTime = originalTime;
  }
}

function syncProcessedAudioPlaybackState(shouldPlay) {
  if (!currentProcessedAudioElement) return;
  if (shouldPlay) {
    syncProcessedAudioPosition(true);
    if (currentProcessedAudioElement.paused) {
      suppressOriginalPlaybackSync = true;
      void currentProcessedAudioElement.play().catch(() => {
        suppressOriginalPlaybackSync = false;
      });
    }
    return;
  }

  if (!currentProcessedAudioElement.paused) {
    suppressOriginalPlaybackSync = true;
    currentProcessedAudioElement.pause();
  }
}

function clearResults() {
  currentAnalysis = null;
  currentFile = null;
  currentSessionId = "";
  activeDetectionIndex = -1;
  currentWaveformPeaks = [];
  currentSpectrogramFrames = [];
  currentWaveformDurationMs = 0;
  if (!currentRecordingBlob) {
    currentRecordedFile = null;
  }

  if (currentAudioElement && !currentAudioElement.paused) currentAudioElement.pause();
  if (currentProcessedAudioElement && !currentProcessedAudioElement.paused) currentProcessedAudioElement.pause();
  currentAudioElement = null;
  currentProcessedAudioElement = null;
  suppressOriginalPlaybackSync = false;
  suppressProcessedPlaybackSync = false;

  if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
  if (currentProcessedAudioUrl) URL.revokeObjectURL(currentProcessedAudioUrl);
  currentAudioUrl = "";
  currentProcessedAudioUrl = "";

  setEmpty(sessionSummary, "metric-list", "No analysis yet.");
  setEmpty(metadataSummary, "metric-list", "Waiting for results.");
  setEmpty(featureSummary, "metric-list", "Waiting for results.");
  setEmpty(spectralSummary, "metric-list", "Waiting for results.");
  setEmpty(detectionsContainer, "detections", "No detections yet.");
  setEmpty(playbackPanel, "playback-panel", "Upload a WAV file to enable playback controls.");
  setEmpty(timelineContainer, "timeline", "No timeline yet.");
  setEmpty(selectionSummary, "selection-summary", "No event selected.");
  if (detectionsCaption) {
    detectionsCaption.textContent = "Highest-confidence labels returned by the active classifier backend.";
  }
  updateRecordingUiState();
}

async function refreshSessionList() {
  try {
    const response = await fetch(`${API_BASE_URL}/sessions`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to load saved sessions.");
    renderSessionList(payload.sessions || []);
  } catch {
    setEmpty(recentSessions, "session-list", "Saved sessions are unavailable.");
  }
}

async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const payload = await response.json();
    if (!response.ok || payload.status !== "ok") {
      throw new Error("Backend health check failed.");
    }
    setStatus("Frontend ready. Backend connection is healthy.", false);
  } catch {
    setStatus("Frontend loaded, but the backend is not reachable at http://127.0.0.1:8000. Start the backend or run start-dev.cmd.", true);
  }
}

function renderSessionList(sessions) {
  if (!sessions.length) return void setEmpty(recentSessions, "session-list", "No saved sessions yet.");
  recentSessions.className = "session-list";
  recentSessions.innerHTML = sessions.map((session) => `
    <article class="session-item">
      <div>
        <strong>${escapeHtml(session.filename)}</strong>
        <p>${escapeHtml(session.status)} - ${session.detection_count} detections - ${session.has_processed_audio ? "processed preview saved" : "analysis only"}</p>
      </div>
      <button type="button" class="subtle-button" data-session-id="${escapeHtml(session.session_id)}">Load Session</button>
    </article>`).join("");
  recentSessions.querySelectorAll("[data-session-id]").forEach((button) => {
    button.addEventListener("click", () => { void loadSavedSession(button.dataset.sessionId); });
  });
}

async function loadSavedSession(sessionId) {
  try {
    setStatus("Loading saved session...", false);
    const response = await fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to load session.");
    currentSessionId = payload.session_id;
    currentFile = buildFileFromBase64(payload.original_audio_base64, payload.filename);
    currentRecordedFile = null;
    currentRecordingBlob = null;
    if (recordingPreview) {
      setEmpty(recordingPreview, "processed-player", "Record a clip to preview it here.");
    }
    setRecordingStatus("Microphone idle.", false);
    await prepareAudioPreview(currentFile);
    currentAnalysis = payload.analysis;
    activeDetectionIndex = payload.analysis.detections.length ? 0 : -1;
    renderAnalysis(payload.analysis);
    renderProcessedAudio(payload.processed_response);
    setStatus(`Loaded saved session for ${payload.filename}.`, false);
    updateRecordingUiState();
  } catch (error) {
    setStatus(error.message || "Failed to load session.", true);
  }
}

function buildFileFromBase64(base64Value, filename) {
  const bytes = decodeBase64(base64Value);
  return new File([bytes], filename, { type: "audio/wav" });
}

function settingsDefaultAttenuation() { return "0.20"; }
function formatSuppressionProfile(profile) { return Object.entries(profile).map(([label, factor]) => `${label}: ${Number(factor).toFixed(2)}`).join(", "); }

updateRecordingUiState();
