// openmoss WebUI — vanilla JS, IndexedDB-backed history.
//
// One server loads one model, so the form adapts to whichever family that is.
// The architecture comes from /info and drives both which controls are visible
// (via <body data-arch> and a couple of CSS rules) and which endpoint is used:
// /tts for the two autoregressive families, /sfx for MOSS-SoundEffect, which is
// a diffusion model and shares none of the sampling knobs.

const DB_NAME = "openmoss-tts";
const DB_VERSION = 1;
const STORE = "generations";

// ── IndexedDB helpers ─────────────────────────────────────────────────────
function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const os = db.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
        os.createIndex("createdAt", "createdAt");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

async function dbAdd(record) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const req = tx.objectStore(STORE).add(record);
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

async function dbList() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).index("createdAt").openCursor(null, "prev");
    const out = [];
    req.onsuccess = (e) => {
      const cur = e.target.result;
      if (cur) { out.push(cur.value); cur.continue(); } else { resolve(out); }
    };
    req.onerror = () => reject(req.error);
  });
}

async function dbDelete(id) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const req = tx.objectStore(STORE).delete(id);
    req.onsuccess = () => resolve();
    req.onerror   = () => reject(req.error);
  });
}

async function dbClear() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const req = tx.objectStore(STORE).clear();
    req.onsuccess = () => resolve();
    req.onerror   = () => reject(req.error);
  });
}

// ── Utilities ──────────────────────────────────────────────────────────────
function $(id) { return document.getElementById(id); }

function fmtTime(ts) {
  const d = new Date(ts);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / (1024 * 1024)).toFixed(2) + " MB";
}

function fmtDuration(s) {
  if (!isFinite(s)) return "";
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60);
  return m + "m " + (s - m * 60).toFixed(0) + "s";
}

function readNum(id) {
  const v = $(id).value;
  if (v === "" || v === null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      // dataURL → strip "data:*/*;base64,"
      const s = String(r.result);
      const comma = s.indexOf(",");
      resolve(comma >= 0 ? s.slice(comma + 1) : s);
    };
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

async function blobDuration(blob) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const a = new Audio();
    a.preload = "metadata";
    a.src = url;
    a.onloadedmetadata = () => { URL.revokeObjectURL(url); resolve(a.duration); };
    a.onerror = () => { URL.revokeObjectURL(url); resolve(NaN); };
  });
}


// ── Which family is loaded ─────────────────────────────────────────────────
let ARCH = null;

function isSfx() { return ARCH === "moss_soundeffect"; }

// Called once /info answers. Sets the attribute the CSS keys off, retitles the
// prompt field, and seeds the duration ceiling from the model rather than a
// hardcoded 30.
function applyArch(info) {
  if (ARCH === info.architecture) return;
  ARCH = info.architecture || null;
  if (ARCH) document.body.dataset.arch = ARCH;
  const label = $("promptLabel");
  if (label) label.textContent = isSfx() ? "Sound description" : "Text";
  const ta = $("text");
  if (ta) {
    ta.placeholder = isSfx()
      ? "Describe a sound… e.g. \"a dog barking twice\""
      : "Type the utterance to synthesize…";
  }
  const secs = $("sfx_seconds");
  if (secs && info.max_seconds) {
    secs.max = info.max_seconds;
    // The server rounds to one decimal, so match that here.
    if (Number(secs.value) > info.max_seconds) secs.value = info.max_seconds;
  }
  document.title = isSfx() ? "openmoss SoundEffect" : "openmoss TTS";
}

// ── Server status ─────────────────────────────────────────────────────────
async function refreshStatus() {
  const el = $("status");
  try {
    const r = await fetch("/info");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const info = await r.json();
    applyArch(info);
    const ch = info.n_channels === 2 ? "stereo" : "mono";
    const tail = isSfx()
      ? `up to ${info.max_seconds}s`
      : `codec ${info.codec_loaded ? "on" : "off"}`;
    el.textContent = `${info.architecture} · ${info.sampling_rate} Hz ${ch} · ${tail} · ${info.requests_served} served`;
    el.className = "status ok";
  } catch (e) {
    el.textContent = "server unreachable";
    el.className = "status error";
  }
}

// ── Generation ─────────────────────────────────────────────────────────────
async function buildRequestBody() {
  const text = $("text").value.trim();
  if (!text) throw new Error(isSfx() ? "a sound description is required" : "text is required");

  if (isSfx()) {
    const body = { prompt: text };
    const seconds = readNum("sfx_seconds");
    if (seconds !== undefined) body.seconds = seconds;
    const steps = readNum("sfx_steps");
    if (steps !== undefined) body.steps = steps;
    const cfg = readNum("sfx_cfg_scale");
    if (cfg !== undefined) body.cfg_scale = cfg;
    const shift = readNum("sfx_sigma_shift");
    if (shift !== undefined) body.sigma_shift = shift;
    // 0 is a real seed here, not "unset", so only an empty field omits it.
    const seed = readNum("sfx_seed");
    if (seed !== undefined) body.seed = seed;
    const neg = $("sfx_negative_prompt").value.trim();
    if (neg) body.negative_prompt = neg;
    return body;
  }

  const body = { text };
  const instruction = $("instruction").value.trim();
  if (instruction) body.instruction = instruction;
  const language = $("language").value;
  if (language) body.language = language;

  const max = readNum("max_new_tokens");
  if (max !== undefined) body.max_new_tokens = max;

  const sampling = {};
  for (const k of [
    "text_temperature", "text_top_p", "text_top_k",
    "audio_temperature", "audio_top_p", "audio_top_k",
    "audio_repetition_penalty", "seed",
  ]) {
    const v = readNum(k);
    if (v !== undefined) sampling[k] = v;
  }
  if (Object.keys(sampling).length) body.sampling = sampling;

  const ref = $("reference").files[0];
  if (ref) body.reference_wav_b64 = await readFileAsBase64(ref);

  return body;
}

async function generate() {
  const btn = $("generate");
  const status = $("genStatus");

  let body;
  try {
    body = await buildRequestBody();
  } catch (e) {
    status.textContent = e.message;
    status.className = "gen-status error";
    return;
  }

  btn.disabled = true;
  status.textContent = "generating…";
  status.className = "gen-status";
  const t0 = performance.now();

  try {
    const r = await fetch(isSfx() ? "/sfx" : "/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const msg = await r.text().catch(() => "");
      throw new Error(`HTTP ${r.status}${msg ? ": " + msg.trim() : ""}`);
    }
    const blob = await r.blob();
    const duration = await blobDuration(blob);

    // The two families report different timings: TTS has a generate phase and
    // an audio-frame count, SoundEffect has a solve phase and a latent length.
    const record = {
      createdAt: Date.now(),
      arch: ARCH,
      text: body.text || body.prompt,
      instruction: body.instruction || null,
      language: body.language || null,
      sampling: body.sampling || null,
      maxNewTokens: body.max_new_tokens || null,
      hasReference: !!body.reference_wav_b64,
      sfx: isSfx() ? {
        seconds: body.seconds ?? null,
        steps: body.steps ?? null,
        cfgScale: body.cfg_scale ?? null,
        seed: body.seed ?? null,
      } : null,
      durationSeconds: duration,
      bytes: blob.size,
      wav: blob,
      genSeconds: Number(r.headers.get("X-MOSS-Generate-Seconds"))
               || Number(r.headers.get("X-MOSS-Solve-Seconds")) || null,
      decodeSeconds: Number(r.headers.get("X-MOSS-Decode-Seconds")) || null,
      frames: Number(r.headers.get("X-MOSS-Audio-Frames"))
           || Number(r.headers.get("X-MOSS-Latent-Frames")) || null,
    };
    await dbAdd(record);
    await renderHistory();
    refreshStatus();

    const wall = ((performance.now() - t0) / 1000).toFixed(2);
    status.textContent = `done in ${wall}s · ${fmtBytes(blob.size)}` + (isFinite(duration) ? ` · ${fmtDuration(duration)} audio` : "");
    status.className = "gen-status ok";
  } catch (e) {
    status.textContent = e.message;
    status.className = "gen-status error";
  } finally {
    btn.disabled = false;
  }
}

// ── History rendering ─────────────────────────────────────────────────────
let blobUrls = new Set();
function revokeAll() {
  for (const u of blobUrls) URL.revokeObjectURL(u);
  blobUrls.clear();
}

async function renderHistory() {
  const list = $("history");
  const empty = $("emptyState");
  const count = $("historyCount");
  revokeAll();
  list.innerHTML = "";

  const items = await dbList();
  count.textContent = items.length ? `${items.length} item${items.length === 1 ? "" : "s"}` : "";
  empty.hidden = items.length > 0;

  for (const item of items) {
    const li = document.createElement("li");

    const tags = [];
    if (item.language)     tags.push(`<span class="tag">${item.language}</span>`);
    if (item.hasReference) tags.push(`<span class="tag">cloned</span>`);
    if (item.instruction)  tags.push(`<span class="tag" title="${escapeHtml(item.instruction)}">instruction</span>`);

    const dur = isFinite(item.durationSeconds) ? fmtDuration(item.durationSeconds) : "—";
    const gen = item.genSeconds ? ` · gen ${item.genSeconds.toFixed(2)}s` : "";

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `
      <span>${tags.join(" ")}${fmtTime(item.createdAt)}</span>
      <span>${dur} · ${fmtBytes(item.bytes)}${gen}</span>`;
    li.appendChild(meta);

    const text = document.createElement("div");
    text.className = "text";
    text.textContent = item.text;
    li.appendChild(text);

    const url = URL.createObjectURL(item.wav);
    blobUrls.add(url);
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = url;
    li.appendChild(audio);

    const actions = document.createElement("div");
    actions.className = "item-actions";

    const dl = document.createElement("a");
    dl.href = url;
    dl.download = `openmoss-${item.id}.wav`;
    dl.className = "";
    const dlBtn = document.createElement("button");
    dlBtn.className = "icon";
    dlBtn.textContent = "Download";
    dlBtn.addEventListener("click", () => dl.click());
    actions.appendChild(dlBtn);
    actions.appendChild(dl);

    const reuse = document.createElement("button");
    reuse.className = "icon";
    reuse.textContent = "Reuse text";
    reuse.addEventListener("click", () => {
      $("text").value = item.text;
      if (item.instruction) $("instruction").value = item.instruction;
      if (item.language)    $("language").value = item.language;
      $("text").focus();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    actions.appendChild(reuse);

    const del = document.createElement("button");
    del.className = "icon danger ghost";
    del.textContent = "Delete";
    del.addEventListener("click", async () => {
      await dbDelete(item.id);
      await renderHistory();
    });
    actions.appendChild(del);

    li.appendChild(actions);
    list.appendChild(li);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[c]));
}

// ── Wire up ────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  $("generate").addEventListener("click", generate);
  $("text").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate();
  });
  $("clearAll").addEventListener("click", async () => {
    const items = await dbList();
    if (!items.length) return;
    if (!confirm(`Delete all ${items.length} generation${items.length === 1 ? "" : "s"} from local storage?`)) return;
    await dbClear();
    await renderHistory();
  });

  await renderHistory();
  await refreshStatus();
  setInterval(refreshStatus, 15000);
});
