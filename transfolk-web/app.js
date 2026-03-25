const API = "http://127.0.0.1:8000";

let models = [];
let inputOsmd = null;
let outputOsmd = null;

let synth = null;
let gainNode = null;
let tonePart = null;
let currentMidiBaseBpm = 120;
let isPaused = false;
let hasGeneratedAudio = false;

const els = {};

window.addEventListener("load", async () => {
  cacheDom();
  bindSliders();
  bindEvents();
  initPanels();
  disablePlayback();

  try {
    await initAudio();
    await loadModels();
    updateGenerationModeUI();
  } catch (error) {
    logError("Startup error", error);
  }
});

function cacheDom() {
  els.modelId = document.getElementById("modelId");
  els.modelInfo = document.getElementById("modelInfo");

  els.file = document.getElementById("file");
  els.fileStatus = document.getElementById("fileStatus");
  els.clearFileBtn = document.getElementById("clearFileBtn");

  els.timeSignatureSelect = document.getElementById("timeSignatureSelect");
  els.tonalitySelect = document.getElementById("tonalitySelect");
  els.modeBadge = document.getElementById("modeBadge");

  els.temperature = document.getElementById("temperature");
  els.temperatureValue = document.getElementById("temperatureValue");
  els.penalty = document.getElementById("penalty");
  els.penaltyValue = document.getElementById("penaltyValue");
  els.topk = document.getElementById("topk");
  els.topkValue = document.getElementById("topkValue");
  els.topp = document.getElementById("topp");
  els.toppValue = document.getElementById("toppValue");
  els.maxLen = document.getElementById("maxLen");

  els.generateBtn = document.getElementById("generateBtn");
  els.clearLogBtn = document.getElementById("clearLogBtn");
  els.log = document.getElementById("log");

  els.score = document.getElementById("score");
  els.outputScore = document.getElementById("outputScore");

  els.playBtn = document.getElementById("playBtn");
  els.pauseBtn = document.getElementById("pauseBtn");
  els.stopBtn = document.getElementById("stopBtn");

  els.volume = document.getElementById("volume");
  els.tempo = document.getElementById("tempo");
  els.tempoValue = document.getElementById("tempoValue");

  els.downloadXml = document.getElementById("downloadXml");
  els.downloadMidi = document.getElementById("downloadMidi");
}

function initPanels() {
  els.score.textContent = "No prompt score loaded";
  els.outputScore.textContent = "No generation yet";
}

function bindSliders() {
  bindSlider(els.temperature, els.temperatureValue);
  bindSlider(els.penalty, els.penaltyValue);
  bindSlider(els.topk, els.topkValue);
  bindSlider(els.topp, els.toppValue);
  bindSlider(els.tempo, els.tempoValue);
}

function bindSlider(input, output) {
  if (!input || !output) return;
  output.textContent = input.value;
  input.addEventListener("input", () => {
    output.textContent = input.value;
  });
}

function bindEvents() {
  els.modelId.addEventListener("change", () => {
    updateModelInfo();
    updateConditioningDefaultsFromModel();
  });

  els.file.addEventListener("change", handleFileChange);
  els.clearFileBtn.addEventListener("click", clearPromptFile);

  els.timeSignatureSelect.addEventListener("change", updateGenerationModeUI);
  els.tonalitySelect.addEventListener("change", updateGenerationModeUI);

  els.generateBtn.addEventListener("click", generateMusic);
  els.clearLogBtn.addEventListener("click", clearLog);

  els.playBtn.addEventListener("click", playGeneratedMidi);
  els.pauseBtn.addEventListener("click", pauseGeneratedMidi);
  els.stopBtn.addEventListener("click", stopGeneratedMidi);

  els.volume.addEventListener("input", () => {
    if (gainNode) {
      gainNode.gain.value = parseFloat(els.volume.value);
    }
  });

  els.tempo.addEventListener("input", () => {
    const bpm = parseInt(els.tempo.value, 10);
    els.tempoValue.textContent = String(bpm);
    applyTempoToTransport(bpm);
  });
}

async function initAudio() {
  gainNode = new Tone.Gain(parseFloat(els.volume.value)).toDestination();
  synth = new Tone.PolySynth(Tone.Synth).connect(gainNode);
  Tone.Transport.bpm.value = parseInt(els.tempo.value, 10);
  Tone.Transport.seconds = 0;
}

async function loadModels() {
  log("Loading models...");
  const response = await fetch(`${API}/models`);

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while loading models`);
  }

  const data = await response.json();

  if (!data || data.status !== "ok" || !Array.isArray(data.models)) {
    throw new Error("Invalid /models response");
  }

  models = data.models;
  populateModelSelect();
  populateConditioningOptions();
  updateModelInfo();
  updateConditioningDefaultsFromModel();

  log(`Loaded ${models.length} models`);
}

function populateModelSelect() {
  els.modelId.innerHTML = "";

  if (!models.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No models available";
    els.modelId.appendChild(opt);
    return;
  }

  models.forEach((model) => {
    const opt = document.createElement("option");
    opt.value = model.id;
    opt.textContent = buildModelLabel(model);
    els.modelId.appendChild(opt);
  });

  els.modelId.selectedIndex = 0;
}

function buildModelLabel(model) {
  const tokenizer = model?.experiment?.tokenizer || model?.algorithm || "tokenizer";
  return `${model.name} · ${tokenizer}`;
}

function populateConditioningOptions() {
  const tsSelect = document.getElementById("timeSignatureSelect");
  const tonSelect = document.getElementById("tonalitySelect");

  // reset
  tsSelect.innerHTML = "";
  tonSelect.innerHTML = "";

  // ---- TIME SIGNATURES (CANÓNICO) ----
  const timeSignatures = ["", "2/4", "3/4", "4/4", "3/8", "6/8"];

  timeSignatures.forEach((ts, i) => {
    const opt = document.createElement("option");
    opt.value = ts;
    opt.textContent = ts === "" ? "Auto / none" : ts;
    tsSelect.appendChild(opt);
  });

  // ---- TONALITY (CANÓNICO) ----
  const tonalities = ["", "major", "minor"];

  tonalities.forEach((t, i) => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t === "" ? "Auto / none" : t;
    tonSelect.appendChild(opt);
  });
}

function refillSelect(selectEl, values, emptyLabel) {
  const current = selectEl.value;
  selectEl.innerHTML = "";

  values.forEach((value, index) => {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = index === 0 ? emptyLabel : value;
    selectEl.appendChild(opt);
  });

  if (values.includes(current)) {
    selectEl.value = current;
  } else {
    selectEl.value = "";
  }
}

function normalizeSelectableValue(value) {
  if (value === null || value === undefined) return "";
  const v = String(value).trim();
  if (!v || v.toLowerCase() === "x" || v.toLowerCase() === "none" || v.toLowerCase() === "null") {
    return "";
  }
  return v;
}

function sortTimeSignatures(values) {
  return values.sort((a, b) => {
    const [an, ad] = a.split("/").map(Number);
    const [bn, bd] = b.split("/").map(Number);

    if (Number.isFinite(an) && Number.isFinite(ad) && Number.isFinite(bn) && Number.isFinite(bd)) {
      const av = an / ad;
      const bv = bn / bd;
      if (av !== bv) return av - bv;
      if (an !== bn) return an - bn;
      return ad - bd;
    }

    return a.localeCompare(b);
  });
}

function sortTonality(values) {
  const priority = { major: 0, minor: 1 };
  return values.sort((a, b) => {
    const pa = priority[a] ?? 99;
    const pb = priority[b] ?? 99;
    if (pa !== pb) return pa - pb;
    return a.localeCompare(b);
  });
}

function getSelectedModel() {
  return models.find((m) => m.id === els.modelId.value) || null;
}

function updateModelInfo() {
  const model = getSelectedModel();

  if (!model) {
    els.modelInfo.innerHTML = `<div class="model-info__placeholder">No model selected.</div>`;
    return;
  }

  const architecture = model.architecture || {};
  const experiment = model.experiment || {};
  const musicContext = experiment.music_context || {};
  const training = model.training || {};
  const artifacts = model.artifacts || {};

  const description = model.description || experiment.description || "No description available.";
  const chips = [
    model.algorithm || experiment.tokenizer,
    normalizeSelectableValue(musicContext.time_signature || model.time_signature),
    normalizeSelectableValue(musicContext.tonality || model.mode),
    architecture.type,
    architecture.name
  ].filter(Boolean);

  els.modelInfo.innerHTML = `
    <div class="model-title">
      <strong>${escapeHtml(model.name || model.id)}</strong>
      <div class="model-description">${escapeHtml(description)}</div>
    </div>

    <div class="chips">
      ${chips.map((chip) => `<span class="chip">${escapeHtml(String(chip))}</span>`).join("")}
    </div>

    <div class="meta-grid">
      ${metaItem("Model ID", model.id)}
      ${metaItem("Corpus", experiment.corpus)}
      ${metaItem("Tokenizer", experiment.tokenizer || model.algorithm)}
      ${metaItem("Time signature", musicContext.time_signature || model.time_signature)}
      ${metaItem("Tonality", musicContext.tonality || model.mode)}
      ${metaItem("Architecture", architecture.name || architecture.type)}
      ${metaItem("Layers", architecture.n_layers)}
      ${metaItem("Heads", architecture.n_heads)}
      ${metaItem("d_model", architecture.d_model)}
      ${metaItem("Max seq len", architecture.max_seq_len)}
      ${metaItem("Training date", training.date)}
      ${metaItem("Artifacts", buildArtifactsLabel(artifacts))}
    </div>
  `;
}

function metaItem(label, value) {
  const safeValue = value === undefined || value === null || value === "" ? "—" : String(value);
  return `
    <div class="meta-item">
      <span class="meta-item__label">${escapeHtml(label)}</span>
      <span class="meta-item__value">${escapeHtml(safeValue)}</span>
    </div>
  `;
}

function buildArtifactsLabel(artifacts) {
  const items = [];
  if (artifacts.has_weights) items.push("weights");
  if (artifacts.has_vocab) items.push("vocab");
  if (artifacts.config_available) items.push("config");
  return items.length ? items.join(", ") : "—";
}

function updateConditioningDefaultsFromModel() {
  if (els.file.files.length) {
    updateGenerationModeUI();
    return;
  }

  const model = getSelectedModel();
  if (!model) {
    updateGenerationModeUI();
    return;
  }

  const ts = normalizeSelectableValue(model?.time_signature || model?.experiment?.music_context?.time_signature);
  const ton = normalizeSelectableValue(model?.mode || model?.experiment?.music_context?.tonality).toLowerCase();

  if (ts && selectHasValue(els.timeSignatureSelect, ts)) {
    els.timeSignatureSelect.value = ts;
  }
  if (ton && selectHasValue(els.tonalitySelect, ton)) {
    els.tonalitySelect.value = ton;
  }

  updateGenerationModeUI();
}

function selectHasValue(selectEl, value) {
  return [...selectEl.options].some((opt) => opt.value === value);
}

async function handleFileChange() {
  const file = els.file.files[0];

  if (file) {
    els.fileStatus.textContent = file.name;
    els.clearFileBtn.disabled = false;

    try {
      await renderPromptScore(file);
      log(`Prompt loaded: ${file.name}`);
    } catch (error) {
      logError("Unable to render prompt score", error);
    }
  } else {
    resetPromptScorePanel();
  }

  updateGenerationModeUI();
}

function clearPromptFile() {
  els.file.value = "";
  els.fileStatus.textContent = "No prompt loaded";
  els.clearFileBtn.disabled = true;
  resetPromptScorePanel();
  updateGenerationModeUI();
  log("Prompt removed");
}

async function renderPromptScore(file) {
  const objectUrl = URL.createObjectURL(file);

  try {
    clearEmptyState(els.score);

    if (!inputOsmd) {
      inputOsmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("score", {
        autoResize: true,
        drawTitle: true,
        backend: "svg",
        drawPartNames: false,
        drawPartAbbreviations: false
      });
    }

    await inputOsmd.load(objectUrl);
    inputOsmd.render();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function resetPromptScorePanel() {
  if (inputOsmd) {
    els.score.innerHTML = "";
  }
  els.score.classList.add("empty-state");
  els.score.textContent = "No prompt score loaded";
  els.fileStatus.textContent = "No prompt loaded";
  els.clearFileBtn.disabled = true;
}

function resetOutputScorePanel() {
  if (outputOsmd) {
    els.outputScore.innerHTML = "";
  }
  els.outputScore.classList.add("empty-state");
  els.outputScore.textContent = "No generation yet";
}

function clearEmptyState(container) {
  container.classList.remove("empty-state");
}

function updateGenerationModeUI() {
  const hasPrompt = !!els.file.files.length;
  const hasTs = !!els.timeSignatureSelect.value;
  const hasTonality = !!els.tonalitySelect.value;

  els.timeSignatureSelect.disabled = hasPrompt;
  els.tonalitySelect.disabled = hasPrompt;

  if (hasPrompt) {
    els.modeBadge.textContent = "Mode: Generate from prompt";
  } else if (hasTs && hasTonality) {
    els.modeBadge.textContent = "Mode: Generate from TS + tonality";
  } else {
    els.modeBadge.textContent = "Mode: Generate";
  }
}

function buildSamplingFormData() {
  const form = new FormData();
  form.append("model", els.modelId.value);
  form.append("temperature", els.temperature.value);
  form.append("max_len", els.maxLen.value);
  form.append("penalty", els.penalty.value);
  form.append("topK", els.topk.value);
  form.append("topP", els.topp.value);
  return form;
}

function resolveGenerationRoute() {
  const hasPrompt = !!els.file.files.length;
  const hasTs = !!els.timeSignatureSelect.value;
  const hasTonality = !!els.tonalitySelect.value;

  if (hasPrompt) {
    return { endpoint: "/generate_from_xml", mode: "generate_from_prompt" };
  }
  if (hasTs && hasTonality) {
    return { endpoint: "/generate_from_TS_tonality", mode: "generate_from_ts_tonality" };
  }
  return { endpoint: "/generate", mode: "generate" };
}

async function generateMusic() {
  try {
    if (!els.modelId.value) {
      throw new Error("No model selected");
    }

    setGeneratingState(true);
    stopAudioHard();
    resetOutputScorePanel();
    hideDownloads();

    const route = resolveGenerationRoute();
    const form = buildSamplingFormData();

    if (route.mode === "generate_from_prompt") {
      const file = els.file.files[0];
      if (!file) {
        throw new Error("Prompt mode selected but no prompt file is loaded");
      }
      form.append("file", file);
    } else if (route.mode === "generate_from_ts_tonality") {
      form.append("time_signature", els.timeSignatureSelect.value);
      form.append("tonality", els.tonalitySelect.value);
    }

    log(`Generating with mode: ${route.mode}`);

    const response = await fetch(`${API}${route.endpoint}`, {
      method: "POST",
      body: form
    });

    const data = await parseJsonResponse(response);

    if (data.status !== "ok") {
      throw new Error(data.message || "Generation failed");
    }

    await loadScore(data.artifacts.musicxml);
    await loadGeneratedMidi(data.artifacts.midi);
    setupDownloads(data);
    enablePlayback();

    log(`Generation OK · ${data.generation_id}`);
  } catch (error) {
    stopAudioHard();
    logError("Generation error", error);
  } finally {
    setGeneratingState(false);
  }
}

function setGeneratingState(isGenerating) {
  els.generateBtn.disabled = isGenerating;
  els.generateBtn.textContent = isGenerating ? "Generating..." : "Generate";
}

async function loadScore(path) {
  const url = `${API}${path}`;

  // 🔥 FIX CLAVE
  els.outputScore.innerHTML = "";

  if (!outputOsmd) {
  outputOsmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("outputScore", {
    autoResize: true,
    drawTitle: true,
    backend: "svg",
    drawPartNames: false,
    drawPartAbbreviations: false
  });
}

  await outputOsmd.load(url);
  outputOsmd.render();
}

async function loadGeneratedMidi(path) {
  const midiUrl = `${API}${path}`;
  const response = await fetch(midiUrl);

  if (!response.ok) {
    throw new Error(`Failed to fetch MIDI (${response.status})`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const midi = new Midi(arrayBuffer);

  stopAudioHard(true);

  const notes = [];
  currentMidiBaseBpm = extractFirstTempo(midi);

  midi.tracks.forEach((track) => {
    track.notes.forEach((n) => {
      notes.push({
        time: n.time,
        name: n.name,
        duration: n.duration,
        velocity: n.velocity
      });
    });
  });

  notes.sort((a, b) => a.time - b.time);

  tonePart = new Tone.Part((time, value) => {
    synth.triggerAttackRelease(value.name, value.duration, time, value.velocity);
  }, notes).start(0);

  Tone.Transport.seconds = 0;
  applyTempoToTransport(parseInt(els.tempo.value, 10));
  hasGeneratedAudio = notes.length > 0;
}

function extractFirstTempo(midi) {
  if (midi?.header?.tempos?.length) {
    const bpm = midi.header.tempos[0]?.bpm;
    if (typeof bpm === "number" && bpm > 0) return bpm;
  }
  return 120;
}

function applyTempoToTransport(desiredBpm) {
  if (!desiredBpm || desiredBpm <= 0) return;
  Tone.Transport.bpm.value = desiredBpm;
  Tone.Transport.playbackRate = currentMidiBaseBpm > 0 ? desiredBpm / currentMidiBaseBpm : 1;
}

function stopAudioHard(silent = false) {
  try {
    if (tonePart) {
      tonePart.dispose();
      tonePart = null;
    }

    Tone.Transport.stop();
    Tone.Transport.cancel();
    Tone.Transport.seconds = 0;
    isPaused = false;
    hasGeneratedAudio = false;
    disablePlayback();

    if (!silent) {
      log("Playback stopped");
    }
  } catch (error) {
    console.error("Audio reset error", error);
  }
}

function enablePlayback() {
  els.playBtn.disabled = false;
  els.pauseBtn.disabled = false;
  els.stopBtn.disabled = false;
}

function disablePlayback() {
  els.playBtn.disabled = true;
  els.pauseBtn.disabled = true;
  els.stopBtn.disabled = true;
}

async function playGeneratedMidi() {
  try {
    if (!tonePart) {
      log("No generated MIDI available for playback");
      return;
    }

    await Tone.start();

    if (!isPaused) {
      Tone.Transport.seconds = 0;
    }

    Tone.Transport.start();
    isPaused = false;
    hasGeneratedAudio = true;
    log("Playback started");
  } catch (error) {
    logError("Playback error", error);
  }
}

function pauseGeneratedMidi() {
  if (!tonePart || !hasGeneratedAudio) return;
  Tone.Transport.pause();
  isPaused = true;
  log("Playback paused");
}

function stopGeneratedMidi() {
  if (!tonePart) return;
  Tone.Transport.stop();
  Tone.Transport.seconds = 0;
  isPaused = false;
  log("Playback stopped");
}

function setupDownloads(data) {
  els.downloadXml.onclick = () => {
  const url = `${API}${data.artifacts.musicxml}`;
  const name = url.split("/").pop();
  downloadFile(url, name);
};
  els.downloadMidi.href = `${API}${data.artifacts.midi}`;
  els.downloadXml.classList.remove("hidden");
  els.downloadMidi.classList.remove("hidden");
}

function hideDownloads() {
  els.downloadXml.classList.add("hidden");
  els.downloadMidi.classList.add("hidden");
  els.downloadXml.removeAttribute("href");
  els.downloadMidi.removeAttribute("href");
}

async function parseJsonResponse(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`Invalid JSON response (${response.status})`);
  }

  if (!response.ok && !data?.message) {
    throw new Error(`HTTP ${response.status}`);
  }

  return data;
}

function clearLog() {
  els.log.textContent = "";
}

function log(message) {
  const ts = new Date().toLocaleTimeString();
  els.log.textContent += `[${ts}] ${message}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function logError(prefix, error) {
  const msg = error instanceof Error ? error.message : String(error);
  log(`${prefix}: ${msg}`);
  console.error(prefix, error);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function downloadFile(url, filename) {
  const res = await fetch(url);
  const blob = await res.blob();

  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;

  document.body.appendChild(a);
  a.click();
  a.remove();
}