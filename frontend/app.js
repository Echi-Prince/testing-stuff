const API_BASE_URL = "http://127.0.0.1:8000";

const uploadForm = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const submitButton = document.querySelector("#submit-button");
const statusBanner = document.querySelector("#status-banner");
const sessionSummary = document.querySelector("#session-summary");
const metadataSummary = document.querySelector("#metadata-summary");
const featureSummary = document.querySelector("#feature-summary");
const spectralSummary = document.querySelector("#spectral-summary");
const detectionsContainer = document.querySelector("#detections");
const playbackPanel = document.querySelector("#playback-panel");
const timelineContainer = document.querySelector("#timeline");
const selectionSummary = document.querySelector("#selection-summary");

let currentAudioUrl = "";
let currentAudioElement = null;
let currentAnalysis = null;
let activeDetectionIndex = -1;
let currentWaveformPeaks = [];
let currentWaveformDurationMs = 0;
let audioContext = null;

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Choose a WAV file before running analysis.", true);
    return;
  }

  submitButton.disabled = true;
  setStatus(`Uploading ${file.name}...`, false);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Analysis request failed.");
    }

    await prepareAudioPreview(file);
    currentAnalysis = payload;
    activeDetectionIndex = payload.detections.length ? 0 : -1;
    renderAnalysis(payload);
    setStatus(`Analysis complete for ${payload.filename}.`, false);
  } catch (error) {
    clearResults();
    setStatus(error.message || "Unexpected error during analysis.", true);
  } finally {
    submitButton.disabled = false;
  }
});

window.addEventListener("resize", () => {
  if (currentAnalysis) {
    drawWaveform(getCurrentPlaybackMs());
  }
});

async function prepareAudioPreview(file) {
  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
  }

  currentAudioUrl = URL.createObjectURL(file);
  currentWaveformPeaks = [];
  currentWaveformDurationMs = 0;

  try {
    const buffer = await file.arrayBuffer();
    const context = getAudioContext();
    const audioBuffer = await context.decodeAudioData(buffer.slice(0));
    currentWaveformPeaks = buildWaveformPeaks(audioBuffer, 720);
    currentWaveformDurationMs = Math.round(audioBuffer.duration * 1000);
  } catch {
    currentWaveformPeaks = [];
    currentWaveformDurationMs = 0;
  }
}

function renderAnalysis(payload) {
  renderMetricList(sessionSummary, [
    ["Status", payload.status],
    ["Filename", payload.filename],
    ["Detections", String(payload.detections.length)],
  ]);

  renderMetricList(metadataSummary, [
    ["Input Rate", `${payload.metadata.sample_rate_hz} Hz`],
    ["Processed Rate", `${payload.metadata.processed_sample_rate_hz} Hz`],
    ["Duration", `${payload.metadata.duration_ms} ms`],
    ["Processed Samples", String(payload.metadata.processed_sample_count)],
    ["Normalization Gain", payload.metadata.normalization_gain.toFixed(3)],
    ["Resampled", payload.metadata.was_resampled ? "Yes" : "No"],
  ]);

  renderMetricList(featureSummary, [
    ["RMS", payload.features.rms.toFixed(6)],
    ["Peak", payload.features.peak_amplitude.toFixed(6)],
    ["Zero Crossings", payload.features.zero_crossing_rate.toFixed(6)],
    ["Activity Ratio", payload.features.dominant_activity_ratio.toFixed(6)],
  ]);

  renderMetricList(spectralSummary, [
    ["Frames", String(payload.spectral_features.frame_count)],
    ["Mel Bins", String(payload.spectral_features.mel_bin_count)],
    ["Mean dB", payload.spectral_features.mean_db.toFixed(3)],
    ["Dynamic Range", payload.spectral_features.dynamic_range_db.toFixed(3)],
    ["Low Band", payload.spectral_features.low_band_mean_db.toFixed(3)],
    ["Mid Band", payload.spectral_features.mid_band_mean_db.toFixed(3)],
    ["High Band", payload.spectral_features.high_band_mean_db.toFixed(3)],
  ]);

  renderPlayback(payload);
  renderInteractiveViews();
}

function renderPlayback(payload) {
  if (!currentAudioUrl) {
    playbackPanel.className = "playback-panel empty-state";
    playbackPanel.textContent = "Upload a WAV file to enable playback controls.";
    return;
  }

  const waveformDuration = currentWaveformDurationMs || payload.metadata.duration_ms;
  playbackPanel.className = "playback-panel";
  playbackPanel.innerHTML = `
    <div class="player-shell">
      <div class="player-meta">
        <div>
          <h3>${escapeHtml(payload.filename)}</h3>
          <p class="muted">Use the waveform, event chips, or timeline rows to jump to detected regions.</p>
        </div>
        <div class="player-actions">
          <button id="play-toggle" type="button">Play</button>
          <button id="jump-selection" type="button" class="subtle-button">Jump To Selection</button>
        </div>
      </div>
      <div class="waveform-panel">
        <div class="waveform-meta">
          <strong>Waveform Preview</strong>
          <p class="muted">${formatMilliseconds(waveformDuration)} total</p>
        </div>
        <canvas id="waveform-canvas" class="waveform-canvas" width="960" height="180"></canvas>
      </div>
      <audio id="audio-player" preload="metadata" controls src="${escapeHtml(currentAudioUrl)}"></audio>
      <input
        id="timeline-scrubber"
        class="scrubber"
        type="range"
        min="0"
        max="${payload.metadata.duration_ms}"
        value="0"
        step="1"
      />
      <div class="time-row">
        <span id="current-time">0:00.000</span>
        <span>${formatMilliseconds(payload.metadata.duration_ms)}</span>
      </div>
    </div>
  `;

  currentAudioElement = playbackPanel.querySelector("#audio-player");
  const playToggle = playbackPanel.querySelector("#play-toggle");
  const jumpSelection = playbackPanel.querySelector("#jump-selection");
  const scrubber = playbackPanel.querySelector("#timeline-scrubber");
  const currentTime = playbackPanel.querySelector("#current-time");
  const waveformCanvas = playbackPanel.querySelector("#waveform-canvas");

  currentAudioElement.addEventListener("timeupdate", () => {
    const currentMs = getCurrentPlaybackMs();
    scrubber.value = String(currentMs);
    currentTime.textContent = formatMilliseconds(currentMs);
    updateTimelinePlayhead(currentMs, payload.metadata.duration_ms);
    drawWaveform(currentMs);
  });

  currentAudioElement.addEventListener("loadedmetadata", () => {
    const durationMs = Math.round(currentAudioElement.duration * 1000);
    scrubber.max = String(durationMs || payload.metadata.duration_ms);
    drawWaveform(getCurrentPlaybackMs());
  });

  currentAudioElement.addEventListener("play", () => {
    playToggle.textContent = "Pause";
  });

  currentAudioElement.addEventListener("pause", () => {
    playToggle.textContent = "Play";
  });

  playToggle.addEventListener("click", () => {
    if (currentAudioElement.paused) {
      void currentAudioElement.play();
      return;
    }
    currentAudioElement.pause();
  });

  jumpSelection.addEventListener("click", () => {
    if (activeDetectionIndex >= 0) {
      focusDetection(activeDetectionIndex, false);
    }
  });

  scrubber.addEventListener("input", () => {
    const targetMs = Number(scrubber.value);
    seekAudio(targetMs);
    updateTimelinePlayhead(targetMs, payload.metadata.duration_ms);
    drawWaveform(targetMs);
  });

  waveformCanvas.addEventListener("click", (event) => {
    const rect = waveformCanvas.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    const durationMs = currentWaveformDurationMs || payload.metadata.duration_ms;
    const targetMs = Math.max(0, Math.min(durationMs, Math.round(durationMs * ratio)));
    seekAudio(targetMs);
    updateTimelinePlayhead(targetMs, payload.metadata.duration_ms);
    drawWaveform(targetMs);
  });

  drawWaveform(0);
}

function renderInteractiveViews() {
  if (!currentAnalysis) {
    return;
  }

  renderDetections(currentAnalysis.detections);
  renderTimeline(currentAnalysis.detections, currentAnalysis.metadata.duration_ms);
  renderSelectionSummary(
    currentAnalysis.detections,
    currentAnalysis.metadata.duration_ms,
  );
  updateTimelinePlayhead(getCurrentPlaybackMs(), currentAnalysis.metadata.duration_ms);
  drawWaveform(getCurrentPlaybackMs());
}

function renderMetricList(target, entries) {
  target.classList.remove("empty-state");
  target.innerHTML = entries
    .map(
      ([label, value]) => `
        <div>
          <dt>${escapeHtml(label)}</dt>
          <dd>${escapeHtml(value)}</dd>
        </div>
      `,
    )
    .join("");
}

function renderDetections(detections) {
  if (!detections.length) {
    detectionsContainer.className = "detections empty-state";
    detectionsContainer.textContent = "No detections returned.";
    return;
  }

  detectionsContainer.className = "detections";
  detectionsContainer.innerHTML = detections
    .map(
      (detection, index) => `
        <button
          type="button"
          class="chip is-button ${index === activeDetectionIndex ? "is-active" : ""}"
          data-detection-index="${index}"
        >
          <h3>${escapeHtml(detection.label)}</h3>
          <p>Confidence ${Number(detection.confidence).toFixed(3)}</p>
          <p>${detection.start_ms} ms to ${detection.end_ms} ms</p>
        </button>
      `,
    )
    .join("");

  detectionsContainer.querySelectorAll("[data-detection-index]").forEach((element) => {
    element.addEventListener("click", () => {
      focusDetection(Number(element.dataset.detectionIndex), true);
    });
  });
}

function renderTimeline(detections, durationMs) {
  if (!detections.length || durationMs <= 0) {
    timelineContainer.className = "timeline empty-state";
    timelineContainer.textContent = "No timeline available.";
    return;
  }

  timelineContainer.className = "timeline";
  timelineContainer.innerHTML = detections
    .map((detection, index) => {
      const left = (detection.start_ms / durationMs) * 100;
      const width = ((detection.end_ms - detection.start_ms) / durationMs) * 100;
      return `
        <div class="timeline-row">
          <div class="timeline-label">${escapeHtml(detection.label)}</div>
          <button
            type="button"
            class="timeline-track ${index === activeDetectionIndex ? "is-active" : ""}"
            data-timeline-index="${index}"
            aria-label="Jump to ${escapeHtml(detection.label)} event"
          >
            <div
              class="timeline-bar"
              style="left:${left}%; width:${Math.max(width, 2)}%;"
              title="${escapeHtml(`${detection.start_ms} ms to ${detection.end_ms} ms`)}"
            ></div>
            <div class="timeline-playhead" style="left:0%;"></div>
          </button>
        </div>
      `;
    })
    .join("");

  timelineContainer.querySelectorAll("[data-timeline-index]").forEach((element) => {
    element.addEventListener("click", () => {
      focusDetection(Number(element.dataset.timelineIndex), true);
    });
  });
}

function renderSelectionSummary(detections, durationMs) {
  if (!detections.length || activeDetectionIndex < 0) {
    selectionSummary.className = "selection-summary empty-state";
    selectionSummary.textContent = "No event selected.";
    return;
  }

  const detection = detections[activeDetectionIndex];
  const share = (((detection.end_ms - detection.start_ms) / durationMs) * 100).toFixed(1);
  selectionSummary.className = "selection-summary";
  selectionSummary.innerHTML = `
    <strong>${escapeHtml(detection.label)}</strong>
    <p>Confidence ${Number(detection.confidence).toFixed(3)}. Span ${detection.start_ms} ms to ${detection.end_ms} ms, covering ${share}% of the clip.</p>
  `;
}

function focusDetection(index, shouldPlay) {
  if (!currentAnalysis || index < 0 || index >= currentAnalysis.detections.length) {
    return;
  }

  activeDetectionIndex = index;
  const detection = currentAnalysis.detections[index];
  renderInteractiveViews();
  seekAudio(detection.start_ms);

  if (shouldPlay && currentAudioElement) {
    void currentAudioElement.play();
  }
}

function updateTimelinePlayhead(currentMs, durationMs) {
  if (durationMs <= 0) {
    return;
  }

  const percent = (currentMs / durationMs) * 100;
  timelineContainer.querySelectorAll(".timeline-playhead").forEach((playhead) => {
    playhead.style.left = `${Math.max(0, Math.min(percent, 100))}%`;
  });
}

function drawWaveform(currentMs) {
  const canvas = playbackPanel.querySelector("#waveform-canvas");
  if (!canvas) {
    return;
  }

  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  const width = canvas.clientWidth || 960;
  const height = canvas.clientHeight || 180;
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * pixelRatio);
  canvas.height = Math.floor(height * pixelRatio);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

  context.clearRect(0, 0, width, height);
  context.fillStyle = "rgba(255, 255, 255, 0.4)";
  context.fillRect(0, 0, width, height);

  const selection = getActiveDetection();
  const durationMs = currentWaveformDurationMs || currentAnalysis?.metadata.duration_ms || 1;

  if (selection) {
    const selectionLeft = (selection.start_ms / durationMs) * width;
    const selectionWidth = ((selection.end_ms - selection.start_ms) / durationMs) * width;
    context.fillStyle = "rgba(0, 109, 119, 0.14)";
    context.fillRect(selectionLeft, 0, Math.max(selectionWidth, 2), height);
  }

  context.strokeStyle = "rgba(0, 109, 119, 0.92)";
  context.lineWidth = 1.2;
  context.beginPath();

  if (currentWaveformPeaks.length) {
    const middleY = height / 2;
    const stepX = width / currentWaveformPeaks.length;
    currentWaveformPeaks.forEach((peak, index) => {
      const x = index * stepX;
      const amplitude = peak * (height * 0.42);
      context.moveTo(x, middleY - amplitude);
      context.lineTo(x, middleY + amplitude);
    });
  } else {
    context.moveTo(0, height / 2);
    context.lineTo(width, height / 2);
  }

  context.stroke();

  const playheadX = (currentMs / durationMs) * width;
  context.strokeStyle = "rgba(27, 28, 29, 0.9)";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(playheadX, 0);
  context.lineTo(playheadX, height);
  context.stroke();
}

function seekAudio(targetMs) {
  if (!currentAudioElement) {
    return;
  }

  currentAudioElement.currentTime = targetMs / 1000;
  const scrubber = playbackPanel.querySelector("#timeline-scrubber");
  const currentTime = playbackPanel.querySelector("#current-time");
  if (scrubber) {
    scrubber.value = String(targetMs);
  }
  if (currentTime) {
    currentTime.textContent = formatMilliseconds(targetMs);
  }
}

function getCurrentPlaybackMs() {
  if (!currentAudioElement) {
    return 0;
  }
  return Math.round(currentAudioElement.currentTime * 1000);
}

function getActiveDetection() {
  if (!currentAnalysis || activeDetectionIndex < 0) {
    return null;
  }
  return currentAnalysis.detections[activeDetectionIndex] || null;
}

function buildWaveformPeaks(audioBuffer, bucketCount) {
  const channelData = audioBuffer.getChannelData(0);
  const bucketSize = Math.max(1, Math.floor(channelData.length / bucketCount));
  const peaks = [];

  for (let bucketIndex = 0; bucketIndex < bucketCount; bucketIndex += 1) {
    const start = bucketIndex * bucketSize;
    const end = Math.min(start + bucketSize, channelData.length);
    let peak = 0;

    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
      peak = Math.max(peak, Math.abs(channelData[sampleIndex]));
    }

    peaks.push(peak);
  }

  return peaks;
}

function getAudioContext() {
  if (!audioContext) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      throw new Error("This browser does not support AudioContext for waveform previews.");
    }
    audioContext = new AudioContextClass();
  }
  return audioContext;
}

function clearResults() {
  if (currentAudioElement) {
    currentAudioElement.pause();
  }

  sessionSummary.className = "metric-list empty-state";
  metadataSummary.className = "metric-list empty-state";
  featureSummary.className = "metric-list empty-state";
  spectralSummary.className = "metric-list empty-state";
  detectionsContainer.className = "detections empty-state";
  playbackPanel.className = "playback-panel empty-state";
  timelineContainer.className = "timeline empty-state";
  selectionSummary.className = "selection-summary empty-state";

  sessionSummary.innerHTML = "<div><dt>Status</dt><dd>No analysis yet</dd></div>";
  metadataSummary.innerHTML = "<div><dt>Processed Rate</dt><dd>Waiting for results</dd></div>";
  featureSummary.innerHTML = "<div><dt>RMS</dt><dd>Waiting for results</dd></div>";
  spectralSummary.innerHTML = "<div><dt>Frames</dt><dd>Waiting for results</dd></div>";
  detectionsContainer.textContent = "No detections yet.";
  playbackPanel.textContent = "Upload a WAV file to enable playback controls.";
  timelineContainer.textContent = "No timeline yet.";
  selectionSummary.textContent = "No event selected.";

  currentAnalysis = null;
  activeDetectionIndex = -1;
  currentWaveformPeaks = [];
  currentWaveformDurationMs = 0;
}

function setStatus(message, isError) {
  statusBanner.textContent = message;
  statusBanner.classList.toggle("error", Boolean(isError));
}

function formatMilliseconds(value) {
  const totalMilliseconds = Math.max(0, Math.round(value));
  const minutes = Math.floor(totalMilliseconds / 60000);
  const seconds = Math.floor((totalMilliseconds % 60000) / 1000);
  const milliseconds = totalMilliseconds % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(milliseconds).padStart(3, "0")}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
