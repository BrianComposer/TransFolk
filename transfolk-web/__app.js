// ==============================
// Configuration
// ==============================
const API_BASE = "http://127.0.0.1:8000";
let modelsCache = [];

// ==============================
// DOM elements
// ==============================
const generateBtn = document.getElementById("generateBtn");
const logEl = document.getElementById("log");

const fileInput = document.getElementById("file");
const modelSelect = document.getElementById("modelId");
const algorithmSelect = document.getElementById("algorithm");
const timeSignatureSelect = document.getElementById("timeSignature");
const modeSelect = document.getElementById("mode");
const maxLenSelect = document.getElementById("maxLen");

const temperatureSlider = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperatureValue");
const temperatureLabel = document.getElementById("temperatureLabel");

const penaltySlider = document.getElementById("penalty");
const penaltyValue = document.getElementById("penaltyValue");
const penaltyLabel = document.getElementById("penaltyLabel");

const topkSlider = document.getElementById("topk");
const topkValue = document.getElementById("topkValue");
const topkLabel = document.getElementById("topkLabel");

const toppSlider = document.getElementById("topp");
const toppValue = document.getElementById("toppValue");
const toppLabel = document.getElementById("toppLabel");

const outputDiv = document.getElementById("outputScore");
const downloadXml = document.getElementById("downloadXml");
const downloadMidi = document.getElementById("downloadMidi");

// ==============================
// Playback (WebAudio via Tone.js)
// ==============================
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const stopBtn = document.getElementById("stopBtn");
const volumeSlider = document.getElementById("volume");

// Tempo slider
const tempoSlider = document.getElementById("tempo");
const tempoValue = document.getElementById("tempoValue");

let currentMidiUrl = null;
let loadedMidi = null;

let masterGain = null;
let synths = [];
let parts = [];
let isPrepared = false;

function setPlaybackEnabled(enabled) {
  playBtn.disabled = !enabled;
  pauseBtn.disabled = !enabled;
  stopBtn.disabled = !enabled;
}
setPlaybackEnabled(false);

// ==============================
// OpenSheetMusicDisplay
// ==============================
let osmd = null;

// ==============================
// Helpers
// ==============================
function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

// ==============================
// Sliders sync
// ==============================
function updateTemperatureUI() {
  const v = temperatureSlider.value;
  temperatureValue.textContent = v;
  temperatureLabel.textContent = `Temperature: ${v}`;
}
updateTemperatureUI();
temperatureSlider.addEventListener("input", updateTemperatureUI);

function updatePenaltyUI() {
  const v = penaltySlider.value;
  penaltyValue.textContent = v;
  penaltyLabel.textContent = `Penalty: ${v}`;
}
updatePenaltyUI();
penaltySlider.addEventListener("input", updatePenaltyUI);

function updateTopkUI() {
  const v = topkSlider.value;
  topkValue.textContent = v;
  topkLabel.textContent = `Topk: ${v}`;
}
updateTopkUI();
topkSlider.addEventListener("input", updateTopkUI);

function updateToppUI() {
  const v = toppSlider.value;
  toppValue.textContent = v;
  toppLabel.textContent = `Topp: ${v}`;
}
updateToppUI();
toppSlider.addEventListener("input", updateToppUI);

// Tempo UI + live update (BPM real del Transport)
function updateTempoUI() {
  const v = Number(tempoSlider?.value ?? 120);
  if (tempoValue) tempoValue.textContent = v;

  if (window.Tone?.Transport) {
    Tone.Transport.bpm.value = v;
  }
}
updateTempoUI();
tempoSlider?.addEventListener("input", updateTempoUI);

// ==============================
// Load MODELS from API
// ==============================
async function loadModels() {
  try {
    log("🔄 Loading models…");

    const response = await fetch(`${API_BASE}/models`);
    if (!response.ok) throw new Error("Failed to fetch models");

    const data = await response.json();
    modelsCache = data.models;

    modelSelect.innerHTML = "";

    data.models.forEach(model => {
      const option = document.createElement("option");
      option.value = model.modelid;
      option.textContent = `${model.algorithm} – ${model.time_signature} – ${model.mode}`;
      modelSelect.appendChild(option);
    });

    autoSelectModel();
    log(`✅ ${modelsCache.length} models loaded.`);
  } catch (err) {
    console.error(err);
    modelSelect.innerHTML = `<option disabled>Error loading models</option>`;
    log(`❌ ${err.message}`);
  }
}

// ==============================
// Auto-select model
// ==============================
function autoSelectModel() {
  const algorithm = algorithmSelect.value;
  const timeSignature = timeSignatureSelect.value;
  const mode = modeSelect.value;

  const match = modelsCache.find(m =>
    m.algorithm === algorithm &&
    m.time_signature === timeSignature &&
    m.mode === mode
  );

  if (match) {
    modelSelect.value = match.modelid;
    log(`🎯 Selected model: ${match.modelid}`);
  } else {
    modelSelect.value = "";
    log("⚠️ No matching model found");
  }
}

algorithmSelect.addEventListener("change", autoSelectModel);
timeSignatureSelect.addEventListener("change", autoSelectModel);
modeSelect.addEventListener("change", autoSelectModel);

// ==============================
// Preview input MusicXML
// ==============================
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  logEl.textContent = "";
  log("📄 Loading input score…");

  try {
    if (!osmd) {
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("score", {
        drawingParameters: "default",
        drawPartNames: false
      });
    } else {
      osmd.clear();
    }

    const xmlText = await file.text();
    await osmd.load(xmlText);
    await osmd.render();

    log("✅ Input score rendered.");
  } catch (err) {
    console.error(err);
    log(`❌ Failed to render input score: ${err.message}`);
  }
});

// ==============================
// Playback internals (ticks-based scheduling)
// ==============================
function cleanupPlayback() {
  try {
    Tone.Transport.stop();
    Tone.Transport.ticks = 0;
  } catch (_) {}

  parts.forEach(p => {
    try { p.dispose(); } catch (_) {}
  });
  parts = [];

  synths.forEach(s => {
    try { s.dispose(); } catch (_) {}
  });
  synths = [];

  isPrepared = false;
  loadedMidi = null;
}

async function preparePlaybackFromUrl(midiUrl) {
  cleanupPlayback();

  loadedMidi = await Midi.fromUrl(midiUrl);

  // Master gain
  if (!masterGain) {
    masterGain = new Tone.Gain(Number(volumeSlider?.value ?? 0.8)).toDestination();
  } else {
    masterGain.gain.value = Number(volumeSlider?.value ?? 0.8);
  }

  // Make Transport interpret ticks with same PPQ as MIDI
  const ppq = loadedMidi.header?.ppq ?? 480;
  Tone.Transport.PPQ = ppq;

  // BPM is controlled by slider
  const desiredBpm = Number(tempoSlider?.value ?? 120);
  Tone.Transport.bpm.value = desiredBpm;

  // Build parts using ticks (NOT seconds)
  loadedMidi.tracks.forEach((track) => {
    if (!track.notes || track.notes.length === 0) return;

    const synth = new Tone.PolySynth(Tone.Synth).connect(masterGain);
    synths.push(synth);

    const events = track.notes.map(n => ({
      time: Tone.Ticks(n.ticks),      // tick-based time
      name: n.name,
      durationTicks: n.durationTicks, // tick-based duration
      velocity: n.velocity ?? 0.9
    }));

    const part = new Tone.Part((time, note) => {
      const durSec = Tone.Ticks(note.durationTicks).toSeconds();
      synth.triggerAttackRelease(note.name, durSec, time, note.velocity);
    }, events);

    part.start(0);
    parts.push(part);
  });

  Tone.Transport.ticks = 0;
  isPrepared = true;

  log(`🔊 Playback ready. PPQ=${ppq} | BPM=${desiredBpm} | Tracks=${loadedMidi.tracks.length}`);
}

// ==============================
// GENERATE
// ==============================
generateBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a MusicXML file.");
    return;
  }

  const modelId = modelSelect.value;
  if (!modelId) {
    alert("No matching model for the selected parameters.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("modelid", modelId);
  formData.append("temperature", temperatureSlider.value);
  formData.append("max_len", maxLenSelect.value);
  formData.append("penalty", penaltySlider.value);
  formData.append("topk", topkSlider.value);
  formData.append("topp", toppSlider.value);

  log("🚀 Generating music…");

  try {
    cleanupPlayback();

    const response = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) throw new Error("Generation failed");

    const data = await response.json();

    const xmlUrl = `${API_BASE}${data.artifacts.musicxml}`;
    const midiUrl = `${API_BASE}${data.artifacts.midi}`;

    currentMidiUrl = midiUrl;
    setPlaybackEnabled(true);
    isPrepared = false;
    log(`🎧 MIDI ready to play: ${midiUrl}`);

    log("🎼 Fetching generated MusicXML…");

    const xmlResponse = await fetch(xmlUrl);
    if (!xmlResponse.ok) throw new Error("Failed to fetch generated MusicXML");

    const xmlText = await xmlResponse.text();

    log("🎼 Rendering generated score…");

    if (!osmd) {
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(outputDiv, {
        drawingParameters: "compact"
      });
    } else {
      osmd.clear();
    }

    await osmd.load(xmlText);
    await osmd.render();

    downloadXml.href = xmlUrl;
    downloadXml.style.display = "inline";

    downloadMidi.href = midiUrl;
    downloadMidi.style.display = "inline";

    log("✅ Generation complete.");
  } catch (err) {
    console.error(err);
    log(`❌ ${err.message}`);
    alert("Error during generation.");
  }
});

// ==============================
// Init
// ==============================
loadModels();

volumeSlider?.addEventListener("input", () => {
  if (masterGain) masterGain.gain.value = Number(volumeSlider.value);
});

// Play / Pause / Stop
playBtn.addEventListener("click", async () => {
  try {
    if (!currentMidiUrl) return;

    // Must be triggered by user gesture
    await Tone.start();

    // Apply slider BPM before start
    Tone.Transport.bpm.value = Number(tempoSlider?.value ?? 120);

    if (!isPrepared) {
      log("🔄 Loading MIDI into player…");
      await preparePlaybackFromUrl(currentMidiUrl);
    }

    Tone.Transport.start("+0.02");
    log("▶ Playing");
  } catch (err) {
    console.error(err);
    log(`❌ Play error: ${err.message}`);
  }
});

pauseBtn.addEventListener("click", () => {
  try {
    Tone.Transport.pause();
    log("⏸ Paused");
  } catch (err) {
    console.error(err);
    log(`❌ Pause error: ${err.message}`);
  }
});

stopBtn.addEventListener("click", () => {
  try {
    Tone.Transport.stop();
    Tone.Transport.ticks = 0; // reset musically
    log("⏹ Stopped");
  } catch (err) {
    console.error(err);
    log(`❌ Stop error: ${err.message}`);
  }
});

