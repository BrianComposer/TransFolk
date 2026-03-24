const API = "http://127.0.0.1:8000";

let models = [];
let osmd = null;

// -----------------------------
// INIT
// -----------------------------
window.onload = async () => {
  await loadModels();
  setupUI();
};

// -----------------------------
// MODELS
// -----------------------------
async function loadModels() {
  const res = await fetch(`${API}/models`);
  const data = await res.json();

  models = data.models;

  const select = document.getElementById("modelId");
  select.innerHTML = "";

  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = buildLabel(m);
    select.appendChild(opt);
  });
}

function buildLabel(m) {
  return `${m.name} | ${m.experiment?.tokenizer || ""} | ${m.experiment?.music_context?.time_signature || ""}`;
}

// -----------------------------
// GENERATE
// -----------------------------
document.getElementById("generateBtn").onclick = async () => {

  const file = document.getElementById("file").files[0];
  if (!file) return alert("Select a file");

  const form = new FormData();

  form.append("file", file);
  form.append("model", document.getElementById("modelId").value);

  form.append("temperature", document.getElementById("temperature").value);
  form.append("penalty", document.getElementById("penalty").value);
  form.append("topK", document.getElementById("topk").value);
  form.append("topP", document.getElementById("topp").value);
  form.append("max_len", document.getElementById("maxLen").value);

  log("Generating...");

  const res = await fetch(`${API}/generate`, {
    method: "POST",
    body: form
  });

  const data = await res.json();

  if (data.status !== "ok") {
    log("Error: " + data.message);
    return;
  }

  log("Done");

  loadScore(data.artifacts.musicxml);
  setupDownloads(data);
};

// -----------------------------
// SCORE (OSMD)
// -----------------------------
async function loadScore(path) {
  const url = API + path;

  if (!osmd) {
    osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("score");
  }

  await osmd.load(url);
  osmd.render();
}

// -----------------------------
// DOWNLOAD
// -----------------------------
function setupDownloads(data) {
  const xml = document.getElementById("downloadXml");
  const midi = document.getElementById("downloadMidi");

  xml.href = API + data.artifacts.musicxml;
  midi.href = API + data.artifacts.midi;

  xml.classList.remove("hidden");
  midi.classList.remove("hidden");
}

// -----------------------------
// UI
// -----------------------------
function setupUI() {

  bindSlider("temperature");
  bindSlider("penalty");
  bindSlider("topk");
  bindSlider("topp");

}

function bindSlider(id) {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "Value");

  el.oninput = () => out.textContent = el.value;
}

// -----------------------------
// LOG
// -----------------------------
function log(msg) {
  const el = document.getElementById("log");
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}