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
const trainingUploadForm = document.querySelector("#training-upload-form");
const trainingUploadInput = document.querySelector("#training-upload-input");
const trainingUploadLabelInput = document.querySelector("#training-upload-label");
const trainingUploadSplitInput = document.querySelector("#training-upload-split");
const trainingUploadSourceNameInput = document.querySelector("#training-upload-source-name");
const trainingUploadSubmitButton = document.querySelector("#training-upload-submit");
const trainingUploadStatus = document.querySelector("#training-upload-status");
const trainingUploadList = document.querySelector("#training-upload-list");
const recentSessions = document.querySelector("#recent-sessions");
const datasetSummary = document.querySelector("#dataset-summary");
const datasetRecordings = document.querySelector("#dataset-recordings");
const datasetDetail = document.querySelector("#dataset-detail");
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
const DASHBOARD_SECTION_STORAGE_KEY = "sound_dashboard_sections_v1";

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
let currentUploadConversionPromise = null;
let currentDatasetRecordingId = "";
let currentDatasetRecording = null;
let trainingStatusRefreshHandle = null;
let artifactActivationInFlight = false;
let trainingUploadInFlight = false;
let currentPreviewPreparationPromise = null;
let currentPreviewPreparationKey = "";

const dashboardSectionElements = Array.from(document.querySelectorAll("[data-dashboard-section]"));
const workspaceTabButtons = Array.from(document.querySelectorAll("[data-toggle-section]"));
const openAllSectionsButton = document.querySelector("#open-all-sections-button");
const closeAllSectionsButton = document.querySelector("#close-all-sections-button");

void checkBackendHealth();
void refreshSessionList();
void refreshDatasetManager();
initializeWorkspaceTabs();
renderTrainingUploadSelection();

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

trainingUploadForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  void uploadTrainingFilesToDataset();
});

fileInput?.addEventListener("change", () => {
  if (fileInput.files?.[0]) {
    currentFile = null;
    currentRecordedFile = null;
    currentRecordingBlob = null;
    if (recordingPreview) {
      setEmpty(recordingPreview, "processed-player", "Record a clip to preview it here.");
    }
    setRecordingStatus("Microphone idle.", false);
    currentUploadConversionPromise = convertSelectedUploadToWav(fileInput.files[0]);
  }
  updateRecordingUiState();
});

trainingUploadInput?.addEventListener("change", () => {
  renderTrainingUploadSelection();
  updateTrainingUploadUiState();
});

workspaceTabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    toggleDashboardSection(button.dataset.toggleSection);
  });
});

openAllSectionsButton?.addEventListener("click", () => {
  setAllDashboardSectionsVisibility(true);
});

closeAllSectionsButton?.addEventListener("click", () => {
  setAllDashboardSectionsVisibility(false);
});

window.addEventListener("resize", () => {
  if (currentAnalysis) {
    drawWaveform(getCurrentPlaybackMs());
    drawSpectrogram(getCurrentPlaybackMs());
  }
});

async function analyzeCurrentFile() {
  const file = await resolveAnalyzableFile();
  if (!file) {
    setStatus("Choose an audio file or record audio before running analysis.", true);
    return;
  }

  submitButton.disabled = true;
  recordAnalyzeButton.disabled = true;
  setStatus(`Analyzing ${file.name} with the backend...`, false);
  try {
    currentFile = file;
    const previewPromise = ensureAudioPreviewPrepared(file);
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${API_BASE_URL}/analyze`, { method: "POST", body: formData });
    setStatus(`Backend response received for ${file.name}. Rendering the preview...`, false);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis request failed.");
    await previewPromise;
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

function ensureAudioPreviewPrepared(file) {
  const previewKey = buildPreviewPreparationKey(file);
  if (
    currentPreviewPreparationPromise &&
    currentPreviewPreparationKey === previewKey
  ) {
    return currentPreviewPreparationPromise;
  }

  currentPreviewPreparationKey = previewKey;
  currentPreviewPreparationPromise = prepareAudioPreview(file);
  return currentPreviewPreparationPromise;
}

function buildPreviewPreparationKey(file) {
  if (!file) return "";
  return [file.name || "", file.size || 0, file.lastModified || 0].join(":");
}

async function resolveAnalyzableFile() {
  if (currentRecordedFile) {
    currentFile = currentRecordedFile;
    return currentRecordedFile;
  }

  if (fileInput?.files?.[0]) {
    if (!currentUploadConversionPromise) {
      currentUploadConversionPromise = convertSelectedUploadToWav(fileInput.files[0]);
    }
    return currentUploadConversionPromise;
  }

  return currentFile;
}

async function convertSelectedUploadToWav(file) {
  setStatus(`Preparing ${file.name} for backend analysis...`, false);
  try {
    const wavFile = await convertAudioFileToWav(file);
    currentFile = wavFile;
    currentPreviewPreparationPromise = ensureAudioPreviewPrepared(wavFile);
    setStatus(
      wavFile.name === file.name
        ? `${file.name} is ready for analysis.`
        : `${file.name} converted to ${wavFile.name} for backend compatibility.`,
      false,
    );
    return wavFile;
  } catch (error) {
    currentFile = null;
    setStatus(error.message || "Failed to convert the selected audio file.", true);
    throw error;
  } finally {
    updateRecordingUiState();
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
    currentUploadConversionPromise = null;
    currentRecordingUrl = URL.createObjectURL(wavBlob);
    currentPreviewPreparationPromise = ensureAudioPreviewPrepared(currentRecordedFile);
    renderRecordingPreview(currentRecordingUrl, currentRecordedFile.name, wavBlob.size);
    setRecordingStatus("Recording ready. Analyze it or save it to the training set.", false);
  } catch (error) {
    currentRecordingBlob = null;
    setRecordingStatus(error.message || "Failed to convert the recording into a compatible WAV file.", true);
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
    await refreshDatasetManager();
  } catch (error) {
    setRecordingStatus(error.message || "Failed to save the recording.", true);
  } finally {
    updateRecordingUiState();
  }
}

async function uploadTrainingFilesToDataset() {
  const selectedFiles = Array.from(trainingUploadInput?.files || []);
  if (!selectedFiles.length) {
    setTrainingUploadStatus("Choose at least one audio file to upload into the training dataset.", true);
    return;
  }

  trainingUploadInFlight = true;
  updateTrainingUploadUiState();
  const uploadLabel = trainingUploadLabelInput?.value || "speech";
  const uploadSplit = trainingUploadSplitInput?.value || "";
  const uploadSourceName = trainingUploadSourceNameInput?.value.trim() || "manual-upload";
  let completedCount = 0;

  try {
    const uploadedPaths = [];
    for (const [index, sourceFile] of selectedFiles.entries()) {
      setTrainingUploadStatus(
        `Preparing ${sourceFile.name} (${index + 1}/${selectedFiles.length}) for training ingest...`,
        false,
      );
      const wavFile = await convertAudioFileToWav(sourceFile);
      const formData = new FormData();
      formData.append("file", wavFile);
      formData.append("label", uploadLabel);
      formData.append("split", uploadSplit);
      formData.append("source_name", uploadSourceName);

      setTrainingUploadStatus(
        `Uploading ${wavFile.name} (${index + 1}/${selectedFiles.length}) into the training dataset...`,
        false,
      );
      const response = await fetch(`${API_BASE_URL}/recordings`, { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Failed to upload ${sourceFile.name}.`);
      }

      uploadedPaths.push(payload.relative_path);
      completedCount += 1;
    }

    trainingUploadInput.value = "";
    renderTrainingUploadSelection();
    await refreshDatasetManager();
    const finalMessage = completedCount === 1
      ? `Uploaded 1 training clip to ${uploadedPaths[0]}.`
      : `Uploaded ${completedCount} training clips into the dataset.`;
    setTrainingUploadStatus(finalMessage, false);
  } catch (error) {
    setTrainingUploadStatus(
      error.message || `Training upload stopped after ${completedCount} completed file(s).`,
      true,
    );
  } finally {
    trainingUploadInFlight = false;
    updateTrainingUploadUiState();
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

async function convertAudioFileToWav(file) {
  const compatibleWavFile = await tryUseCompatibleWavFile(file);
  if (compatibleWavFile) {
    return compatibleWavFile;
  }

  const context = getAudioContext();
  if (context.state === "suspended") {
    await context.resume();
  }

  let audioBuffer;
  try {
    audioBuffer = await context.decodeAudioData((await file.arrayBuffer()).slice(0));
  } catch {
    throw new Error("The selected file could not be decoded in the browser.");
  }

  const wavBlob = encodeAudioBufferToWav(audioBuffer);
  const wavFilename = buildCompatibleWavFilename(file.name);
  return new File([wavBlob], wavFilename, {
    type: "audio/wav",
    lastModified: Date.now(),
  });
}

async function tryUseCompatibleWavFile(file) {
  const headerBytes = new Uint8Array(await file.slice(0, Math.min(file.size, 262144)).arrayBuffer());
  const wavMetadata = parseWavCompatibility(headerBytes);
  if (!wavMetadata.isCompatible) {
    return null;
  }

  if (file.type === "audio/wav" || file.name.toLowerCase().endsWith(".wav")) {
    return file;
  }

  return new File([file], buildCompatibleWavFilename(file.name), {
    type: "audio/wav",
    lastModified: file.lastModified || Date.now(),
  });
}

function parseWavCompatibility(bytes) {
  if (!bytes || bytes.length < 44) {
    return { isCompatible: false };
  }

  if (readAscii(bytes, 0, 4) !== "RIFF" || readAscii(bytes, 8, 4) !== "WAVE") {
    return { isCompatible: false };
  }

  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const chunkId = readAscii(bytes, offset, 4);
    const chunkSize = readUint32LittleEndian(bytes, offset + 4);
    const chunkDataOffset = offset + 8;

    if (chunkId === "fmt " && chunkDataOffset + 16 <= bytes.length) {
      const audioFormat = readUint16LittleEndian(bytes, chunkDataOffset);
      const bitsPerSample = readUint16LittleEndian(bytes, chunkDataOffset + 14);
      return {
        isCompatible: audioFormat === 1 && [8, 16, 32].includes(bitsPerSample),
      };
    }

    offset = chunkDataOffset + chunkSize + (chunkSize % 2);
  }

  return { isCompatible: false };
}

function readAscii(bytes, offset, length) {
  let value = "";
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(bytes[offset + index] || 0);
  }
  return value;
}

function readUint16LittleEndian(bytes, offset) {
  return (bytes[offset] || 0) | ((bytes[offset + 1] || 0) << 8);
}

function readUint32LittleEndian(bytes, offset) {
  return (
    (bytes[offset] || 0) |
    ((bytes[offset + 1] || 0) << 8) |
    ((bytes[offset + 2] || 0) << 16) |
    ((bytes[offset + 3] || 0) << 24)
  ) >>> 0;
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

function buildCompatibleWavFilename(filename) {
  const basename = String(filename || "audio")
    .replace(/\.[^./\\]+$/, "")
    .trim();
  return `${basename || "audio"}.wav`;
}

function buildRecordedFilename() {
  const stamp = new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
  return `recording-${stamp}.wav`;
}

function renderRecordingPreview(url, filename, byteCount) {
  if (!recordingPreview) return;
  recordingPreview.className = "processed-player";
  recordingPreview.innerHTML = `<strong>${escapeHtml(filename)}</strong><p class="muted">${Math.round(byteCount / 1024)} KB compatible WAV clip ready for analysis or training ingest.</p><audio controls preload="metadata" src="${escapeHtml(url)}"></audio>`;
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

function setTrainingUploadStatus(message, isError) {
  if (!trainingUploadStatus) return;
  trainingUploadStatus.textContent = message;
  trainingUploadStatus.classList.toggle("error", Boolean(isError));
}

function updateRecordingUiState() {
  const hasAnalyzableFile = Boolean(currentRecordedFile || fileInput?.files?.[0] || currentFile);
  if (recordStartButton) recordStartButton.disabled = isRecording;
  if (recordStopButton) recordStopButton.disabled = !isRecording;
  if (recordAnalyzeButton) recordAnalyzeButton.disabled = isRecording || !hasAnalyzableFile;
  if (recordSaveButton) recordSaveButton.disabled = isRecording || !currentRecordingBlob;
  updateTrainingUploadUiState();
}

function updateTrainingUploadUiState() {
  if (!trainingUploadSubmitButton) return;
  trainingUploadSubmitButton.disabled = trainingUploadInFlight || !(trainingUploadInput?.files?.length);
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
  if (!file) throw new Error("Select or record an audio file before requesting suppression.");
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
  currentUploadConversionPromise = null;
  currentPreviewPreparationPromise = null;
  currentPreviewPreparationKey = "";
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
  setEmpty(playbackPanel, "playback-panel", "Upload or record audio to enable playback controls.");
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

async function refreshDatasetManager() {
  try {
    const [recordingsResponse, summaryResponse, trainingResponse, artifactsResponse] = await Promise.all([
      fetch(`${API_BASE_URL}/recordings`),
      fetch(`${API_BASE_URL}/recordings/summary`),
      fetch(`${API_BASE_URL}/training/status`),
      fetch(`${API_BASE_URL}/artifacts`),
    ]);
    const recordingsPayload = await recordingsResponse.json();
    const summaryPayload = await summaryResponse.json();
    const trainingPayload = await trainingResponse.json();
    const artifactsPayload = await artifactsResponse.json();
    if (!recordingsResponse.ok) throw new Error(recordingsPayload.detail || "Failed to load saved recordings.");
    if (!summaryResponse.ok) throw new Error(summaryPayload.detail || "Failed to load dataset summary.");
    if (!trainingResponse.ok) throw new Error(trainingPayload.detail || "Failed to load training status.");
    if (!artifactsResponse.ok) throw new Error(artifactsPayload.detail || "Failed to load model artifacts.");
    renderDatasetSummary(summaryPayload, trainingPayload, artifactsPayload.artifacts || []);
    renderDatasetRecordings(recordingsPayload.recordings || []);
    syncTrainingStatusPolling(trainingPayload.status);
    if (currentDatasetRecordingId) {
      const stillExists = (recordingsPayload.recordings || []).some((recording) => recording.recording_id === currentDatasetRecordingId);
      if (stillExists) {
        await loadDatasetRecording(currentDatasetRecordingId, false);
      } else {
        currentDatasetRecordingId = "";
        currentDatasetRecording = null;
        setEmpty(datasetDetail, "dataset-detail", "Select a saved training clip to preview and edit it.");
      }
    }
  } catch {
    setEmpty(datasetSummary, "dataset-summary", "Dataset summary is unavailable.");
    setEmpty(datasetRecordings, "session-list", "Saved training clips are unavailable.");
    setEmpty(datasetDetail, "dataset-detail", "Saved training clips are unavailable.");
    syncTrainingStatusPolling("idle");
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

function renderTrainingUploadSelection() {
  if (!trainingUploadList) return;
  const selectedFiles = Array.from(trainingUploadInput?.files || []);
  if (!selectedFiles.length) {
    setEmpty(trainingUploadList, "upload-list", "No training files selected.");
    return;
  }

  trainingUploadList.className = "upload-list";
  trainingUploadList.innerHTML = selectedFiles.map((file) => `
    <article class="upload-list-item">
      <strong>${escapeHtml(file.name)}</strong>
      <p>${Math.round(file.size / 1024)} KB</p>
    </article>`).join("");
}

function initializeWorkspaceTabs() {
  const storedState = loadDashboardSectionState();
  for (const section of dashboardSectionElements) {
    const shouldShow = storedState[section.id] !== false;
    section.classList.toggle("is-hidden", !shouldShow);
  }
  syncWorkspaceTabs();
}

function toggleDashboardSection(sectionId) {
  const targetSection = dashboardSectionElements.find((section) => section.id === sectionId);
  if (!targetSection) return;
  targetSection.classList.toggle("is-hidden");
  syncWorkspaceTabs();
  persistDashboardSectionState();
}

function setAllDashboardSectionsVisibility(shouldShow) {
  for (const section of dashboardSectionElements) {
    section.classList.toggle("is-hidden", !shouldShow);
  }
  syncWorkspaceTabs();
  persistDashboardSectionState();
}

function syncWorkspaceTabs() {
  for (const button of workspaceTabButtons) {
    const targetSection = dashboardSectionElements.find((section) => section.id === button.dataset.toggleSection);
    const isVisible = Boolean(targetSection) && !targetSection.classList.contains("is-hidden");
    button.classList.toggle("is-active", isVisible);
    button.setAttribute("aria-pressed", isVisible ? "true" : "false");
  }
}

function persistDashboardSectionState() {
  try {
    const payload = Object.fromEntries(
      dashboardSectionElements.map((section) => [section.id, !section.classList.contains("is-hidden")])
    );
    window.localStorage.setItem(DASHBOARD_SECTION_STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Keep section toggles working even if storage is unavailable.
  }
}

function loadDashboardSectionState() {
  try {
    const rawValue = window.localStorage.getItem(DASHBOARD_SECTION_STORAGE_KEY);
    if (!rawValue) return {};
    const parsedValue = JSON.parse(rawValue);
    if (!parsedValue || typeof parsedValue !== "object" || Array.isArray(parsedValue)) {
      return {};
    }
    return parsedValue;
  } catch {
    return {};
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

function renderDatasetRecordings(recordings) {
  if (!recordings.length) {
    setEmpty(datasetRecordings, "session-list", "No saved training clips yet.");
    return;
  }
  datasetRecordings.className = "session-list";
  datasetRecordings.innerHTML = recordings.map((recording) => `
    <article class="dataset-card ${recording.recording_id === currentDatasetRecordingId ? "is-active" : ""}">
      <div>
        <strong>${escapeHtml(recording.filename)}</strong>
        <p>${escapeHtml(recording.relative_path)}</p>
      </div>
      <div class="dataset-card-meta">
        <span class="dataset-meta-chip">${escapeHtml(recording.label)}</span>
        <span class="dataset-meta-chip">${escapeHtml(recording.split || "unspecified")}</span>
        <span class="dataset-meta-chip">${formatMilliseconds(recording.duration_ms)}</span>
      </div>
      <button type="button" class="subtle-button" data-recording-id="${escapeHtml(recording.recording_id)}">Manage Clip</button>
    </article>`).join("");
  datasetRecordings.querySelectorAll("[data-recording-id]").forEach((button) => {
    button.addEventListener("click", () => {
      void loadDatasetRecording(button.dataset.recordingId, true);
    });
  });
}

function renderDatasetSummary(summary, trainingStatus, artifacts) {
  if (!datasetSummary) return;
  const artifactCards = artifacts.length
    ? artifacts.map((artifact) => `
      <article class="artifact-card ${artifact.is_active ? "is-active" : ""}">
        <div class="artifact-card-head">
          <div>
            <strong>${escapeHtml(artifact.model_name)}</strong>
            <p class="muted">${escapeHtml(artifact.relative_path)}</p>
          </div>
          <span class="dataset-meta-chip">${artifact.is_active ? "active" : "standby"}</span>
        </div>
        <div class="artifact-card-meta">
          <span class="dataset-meta-chip">${artifact.training_example_count} train</span>
          <span class="dataset-meta-chip">${artifact.validation_example_count} val</span>
          <span class="dataset-meta-chip">${artifact.class_names.length} classes</span>
          <span class="dataset-meta-chip">${artifact.source_run_id ? escapeHtml(artifact.source_run_id) : "manual/default"}</span>
        </div>
        <p class="muted">Weights: <code>${escapeHtml(artifact.weights_relative_path)}</code></p>
        <p class="muted">Updated ${formatTimestamp(artifact.updated_at)}</p>
        <div class="button-row">
          <button
            type="button"
            class="${artifact.is_active ? "subtle-button" : ""}"
            data-activate-artifact-id="${escapeHtml(artifact.artifact_id)}"
            ${artifact.is_active || artifactActivationInFlight ? "disabled" : ""}
          >
            ${artifact.is_active ? "Active Model" : "Use This Version"}
          </button>
        </div>
      </article>`)
      .join("")
    : `<div class="dataset-count-list"><p class="muted">No trained model artifacts are available yet. Start a training run to populate this list.</p></div>`;
  datasetSummary.className = "dataset-summary";
  datasetSummary.innerHTML = `
    <div class="section-head">
      <div>
        <strong>Collected Dataset</strong>
        <p class="muted">Current counts from <code>training/real_recordings/</code></p>
      </div>
      <div class="button-row">
        <button type="button" id="build-manifest-button">Build Manifest</button>
        <button type="button" id="start-training-button" ${trainingStatus.status === "running" ? "disabled" : ""}>Start Training</button>
      </div>
    </div>
    <div class="dataset-summary-grid">
      <div class="dataset-summary-card"><span class="muted">Total Clips</span><strong>${summary.total_recordings}</strong></div>
      <div class="dataset-summary-card"><span class="muted">Total Duration</span><strong>${formatMilliseconds(summary.total_duration_ms)}</strong></div>
      <div class="dataset-summary-card"><span class="muted">Manifest</span><strong>${summary.manifest_exists ? "Ready" : "Missing"}</strong></div>
      <div class="dataset-summary-card"><span class="muted">Manifest Path</span><strong>${escapeHtml(summary.manifest_relative_path)}</strong></div>
    </div>
    <div class="dataset-count-grid">
      <div class="dataset-count-list">
        <h3>By Label</h3>
        ${Object.entries(summary.by_label).map(([label, count]) => `<div><dt>${escapeHtml(label)}</dt><dd>${count}</dd></div>`).join("")}
      </div>
      <div class="dataset-count-list">
        <h3>By Split</h3>
        ${Object.entries(summary.by_split).map(([split, count]) => `<div><dt>${escapeHtml(split)}</dt><dd>${count}</dd></div>`).join("")}
      </div>
    </div>
    <div class="dataset-count-list">
      <h3>Training Status</h3>
      <div><dt>Status</dt><dd>${escapeHtml(trainingStatus.status)}</dd></div>
      <div><dt>Run ID</dt><dd>${escapeHtml(trainingStatus.run_id || "not started")}</dd></div>
      <div><dt>Epoch</dt><dd>${trainingStatus.current_epoch}/${trainingStatus.epochs}</dd></div>
      <div><dt>Last Val Acc</dt><dd>${Number(trainingStatus.last_val_accuracy || 0).toFixed(4)}</dd></div>
      <div><dt>Output</dt><dd>${escapeHtml(trainingStatus.output_relative_path || "n/a")}</dd></div>
    </div>
    <div class="artifact-list-section">
      <div class="section-head">
        <div>
          <h3>Model Artifacts</h3>
          <p class="muted">Switch the active trained model and keep older versions available as inference backups.</p>
        </div>
      </div>
      <div class="artifact-list">${artifactCards}</div>
    </div>
    <p class="muted">${summary.manifest_exists ? `Last manifest update: ${formatTimestamp(summary.manifest_updated_at)}.` : "No manifest has been built yet."}</p>`;
  datasetSummary.querySelector("#build-manifest-button")?.addEventListener("click", () => {
    void buildDatasetManifest();
  });
  datasetSummary.querySelector("#start-training-button")?.addEventListener("click", () => {
    void startTrainingRun();
  });
  datasetSummary.querySelectorAll("[data-activate-artifact-id]").forEach((button) => {
    button.addEventListener("click", () => {
      void activateModelArtifact(button.dataset.activateArtifactId);
    });
  });
}

async function loadDatasetRecording(recordingId, updateSelection = true) {
  try {
    if (updateSelection) currentDatasetRecordingId = recordingId;
    const response = await fetch(`${API_BASE_URL}/recordings/${encodeURIComponent(recordingId)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to load the recording.");
    currentDatasetRecording = payload;
    currentDatasetRecordingId = payload.recording_id;
    renderDatasetDetail(payload);
    await refreshDatasetManagerSelection();
  } catch (error) {
    setEmpty(datasetDetail, "dataset-detail", error.message || "Failed to load the recording.");
  }
}

async function refreshDatasetManagerSelection() {
  try {
    const response = await fetch(`${API_BASE_URL}/recordings`);
    const payload = await response.json();
    if (!response.ok) return;
    renderDatasetRecordings(payload.recordings || []);
  } catch {
    // Leave current dataset UI intact if the refresh fails.
  }
}

function renderDatasetDetail(recording) {
  const audioUrl = `data:audio/wav;base64,${recording.wav_base64}`;
  datasetDetail.className = "dataset-detail";
  datasetDetail.innerHTML = `
    <div>
      <h3>${escapeHtml(recording.filename)}</h3>
      <p class="muted">${escapeHtml(recording.relative_path)}</p>
    </div>
    <div class="dataset-card-meta">
      <span class="dataset-meta-chip">${escapeHtml(recording.label)}</span>
      <span class="dataset-meta-chip">${escapeHtml(recording.split || "unspecified")}</span>
      <span class="dataset-meta-chip">${recording.sample_rate_hz} Hz</span>
      <span class="dataset-meta-chip">${formatMilliseconds(recording.duration_ms)}</span>
    </div>
    <audio controls preload="metadata" src="${audioUrl}"></audio>
    <form id="dataset-edit-form" class="dataset-form">
      <label>
        <span>Label</span>
        <select id="dataset-edit-label">
          ${buildSupportedClassOptions(recording.label)}
        </select>
      </label>
      <label>
        <span>Dataset Split</span>
        <select id="dataset-edit-split">
          <option value="" ${recording.split === "" ? "selected" : ""}>unspecified</option>
          <option value="train" ${recording.split === "train" ? "selected" : ""}>train</option>
          <option value="val" ${recording.split === "val" ? "selected" : ""}>val</option>
          <option value="test" ${recording.split === "test" ? "selected" : ""}>test</option>
        </select>
      </label>
      <div class="button-row">
        <button type="submit">Save Changes</button>
        <button type="button" id="dataset-delete-button" class="subtle-button">Delete Clip</button>
      </div>
    </form>
    <p class="muted">Created ${formatTimestamp(recording.created_at)}. Last updated ${formatTimestamp(recording.updated_at)}.</p>`;
  datasetDetail.querySelector("#dataset-edit-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    void saveDatasetRecordingChanges(recording.recording_id);
  });
  datasetDetail.querySelector("#dataset-delete-button")?.addEventListener("click", () => {
    void deleteDatasetRecording(recording.recording_id);
  });
}

function buildSupportedClassOptions(selectedLabel) {
  return ["speech", "keyboard", "dog_bark", "traffic", "siren", "vacuum", "music"]
    .map((label) => `<option value="${escapeHtml(label)}" ${label === selectedLabel ? "selected" : ""}>${escapeHtml(label)}</option>`)
    .join("");
}

async function saveDatasetRecordingChanges(recordingId) {
  const labelNode = datasetDetail.querySelector("#dataset-edit-label");
  const splitNode = datasetDetail.querySelector("#dataset-edit-split");
  if (!labelNode || !splitNode) return;
  try {
    const response = await fetch(`${API_BASE_URL}/recordings/${encodeURIComponent(recordingId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: labelNode.value, split: splitNode.value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to update the recording.");
    await loadDatasetRecording(payload.recording_id, true);
    await refreshDatasetManager();
    setRecordingStatus(`Updated training clip to ${payload.relative_path}.`, false);
  } catch (error) {
    setRecordingStatus(error.message || "Failed to update the training clip.", true);
  }
}

async function deleteDatasetRecording(recordingId) {
  try {
    const response = await fetch(`${API_BASE_URL}/recordings/${encodeURIComponent(recordingId)}`, {
      method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to delete the recording.");
    currentDatasetRecordingId = "";
    currentDatasetRecording = null;
    setEmpty(datasetDetail, "dataset-detail", "Select a saved training clip to preview and edit it.");
    await refreshDatasetManager();
    setRecordingStatus("Deleted the selected training clip.", false);
  } catch (error) {
    setRecordingStatus(error.message || "Failed to delete the training clip.", true);
  }
}

async function buildDatasetManifest() {
  try {
    setRecordingStatus("Building dataset manifest...", false);
    const response = await fetch(`${API_BASE_URL}/recordings/build-manifest`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to build the manifest.");
    await refreshDatasetManager();
    setRecordingStatus(
      `Built ${payload.manifest_relative_path} with ${payload.total_examples} examples.`,
      false,
    );
  } catch (error) {
    setRecordingStatus(error.message || "Failed to build the dataset manifest.", true);
  }
}

async function startTrainingRun() {
  try {
    setRecordingStatus("Starting model training...", false);
    const response = await fetch(`${API_BASE_URL}/training/run`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to start training.");
    await refreshDatasetManager();
    setRecordingStatus(`Training started with run ${payload.run_id}.`, false);
  } catch (error) {
    setRecordingStatus(error.message || "Failed to start training.", true);
  }
}

async function activateModelArtifact(artifactId) {
  if (!artifactId || artifactActivationInFlight) return;
  artifactActivationInFlight = true;
  try {
    setRecordingStatus("Activating the selected trained model version...", false);
    const response = await fetch(`${API_BASE_URL}/artifacts/activate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ artifact_id: artifactId }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to activate the selected model artifact.");
    await refreshDatasetManager();
    setRecordingStatus(
      `Activated ${payload.relative_path}. Older versions remain available as fallback candidates.`,
      false,
    );
  } catch (error) {
    setRecordingStatus(error.message || "Failed to activate the selected model artifact.", true);
  } finally {
    artifactActivationInFlight = false;
  }
}

function syncTrainingStatusPolling(status) {
  if (status === "running") {
    if (trainingStatusRefreshHandle) return;
    trainingStatusRefreshHandle = window.setInterval(() => {
      void refreshDatasetManager();
    }, 3000);
    return;
  }
  if (trainingStatusRefreshHandle) {
    window.clearInterval(trainingStatusRefreshHandle);
    trainingStatusRefreshHandle = null;
  }
}

async function loadSavedSession(sessionId) {
  try {
    setStatus("Loading saved session...", false);
    const response = await fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to load session.");
    currentSessionId = payload.session_id;
    currentFile = buildFileFromBase64(payload.original_audio_base64, payload.filename);
    currentUploadConversionPromise = null;
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
function formatTimestamp(value) {
  if (!value) return "n/a";
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime()) ? "n/a" : timestamp.toLocaleString();
}

updateRecordingUiState();
