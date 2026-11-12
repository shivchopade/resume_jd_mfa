/*
 * app.js - all frontend behaviour. Plain JavaScript: no framework, no build step.
 *
 *   1. On load, ask the backend for its status ......... GET  /api/health
 *   2. "Screen resumes" sends the form ................. POST /api/screen  (multipart)
 *   3. "Try sample data" fills the form and runs ....... GET  /api/sample + POST /api/screen/sample
 *   4. render() draws the ranking, score bars, integrity flags and the nDCG card
 */

// In Docker, nginx forwards /api to the backend. For local development without
// Docker, point the page at the backend:  index.html?api=http://localhost:8000/api
const API = new URLSearchParams(location.search).get("api") || "/api";

const $ = (id) => document.getElementById(id);
const fileInput = $("resumes");
let busy = false;

// Human-readable names and explanations for the integrity flags from the backend
const FLAG_TEXT = {
  hidden_text: "Hidden text",
  tiny_font: "Tiny font",
  jd_copy: "Copied from JD",
  duplicate_file: "Duplicate",
  corrupt_pdf: "Corrupt PDF",
  encrypted_pdf: "Password-protected",
  no_text_layer: "Scanned / no text",
  too_many_pages: "Too many pages",
};
const FLAG_HELP = {
  hidden_text: "White or invisible text was found. It was removed before scoring.",
  tiny_font: "Text too small to read was found. It was removed before scoring.",
  jd_copy: "Sentences copied word-for-word from the job description. They were removed before scoring.",
  duplicate_file: "The same file was uploaded more than once.",
  corrupt_pdf: "The file could not be opened as a PDF.",
  encrypted_pdf: "The PDF is password-protected, so it could not be read.",
  no_text_layer: "Very little text found. The PDF is probably a scanned image.",
  too_many_pages: "Only the first pages were read.",
};

// ------------------------------------------------------------------ helpers
function escapeHtml(value) {
  // Everything that comes from a file or the LLM is escaped before it touches the page.
  return String(value ?? "").replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

const fmt = (x, digits = 2) => (x == null ? "–" : Number(x).toFixed(digits));

function prettyName(filename) {
  // "rahul_verma.pdf" -> "Rahul Verma"
  const base = filename.replace(/\.pdf$/i, "").replace(/[_\-.]+/g, " ").trim();
  return base.replace(/\b\w/g, (c) => c.toUpperCase()) || filename;
}

function weightsText(w) {
  return `semantic ${fmt(w.semantic)} · lexical ${fmt(w.lexical)} · LLM ${fmt(w.llm)}`;
}

function kValue() {
  const k = Number($("k").value || 10);
  return Number.isInteger(k) && k >= 1 && k <= 100 ? k : null;
}

// ------------------------------------------------------------------ errors + busy state
function showError(message, field) {
  const box = $("error");
  box.textContent = message;
  box.hidden = false;
  if (field) {
    field.setAttribute("aria-invalid", "true");
    field.focus();
  }
}

function clearError() {
  $("error").hidden = true;
  document.querySelectorAll("[aria-invalid]").forEach((el) => el.removeAttribute("aria-invalid"));
}

function setBusy(on) {
  $("run").disabled = on;
  $("sample").disabled = on;
  $("run").textContent = on ? "Scoring…" : "Screen resumes";
}

function emptyState() {
  return `<div class="empty"><p class="empty-title">No results yet</p>
    <p>Upload a job description and resumes, or press <strong>Try sample data</strong>.</p></div>`;
}

// ------------------------------------------------------------------ 1. backend status
async function loadStatus() {
  const el = $("status");
  try {
    const res = await fetch(`${API}/health`);
    if (!res.ok) throw new Error(String(res.status));
    const h = await res.json();
    const model = h.embedding_model.split("/").pop();
    el.className = "status ok";
    el.innerHTML = `<span class="dot"></span><span>${escapeHtml(model)} · LLM ${h.llm_enabled ? "on" : "off"}</span>`;
  } catch {
    el.className = "status down";
    el.innerHTML = `<span class="dot"></span><span>Backend offline</span>`;
  }
}

// ------------------------------------------------------------------ file picker / drag & drop
function updateDropzone() {
  const n = fileInput.files.length;
  $("dz-main").textContent = n ? `${n} PDF${n > 1 ? "s" : ""} selected` : "Choose PDFs or drop them here";
  $("dropzone").classList.toggle("has-files", n > 0);
}

fileInput.addEventListener("change", () => { updateDropzone(); clearError(); });
["dragenter", "dragover"].forEach((type) => $("dropzone").addEventListener(type, (e) => {
  e.preventDefault();
  $("dropzone").classList.add("dragging");
}));
["dragleave", "drop"].forEach((type) => $("dropzone").addEventListener(type, (e) => {
  e.preventDefault();
  $("dropzone").classList.remove("dragging");
}));
$("dropzone").addEventListener("drop", (e) => {
  const pdfs = [...e.dataTransfer.files].filter((f) => /\.pdf$/i.test(f.name) || f.type === "application/pdf");
  if (!pdfs.length) return showError("Only PDF files can be screened.");
  const transfer = new DataTransfer();
  pdfs.forEach((f) => transfer.items.add(f));
  fileInput.files = transfer.files;
  updateDropzone();
  clearError();
});
["jd", "k"].forEach((id) => $(id).addEventListener("input", clearError));

// ------------------------------------------------------------------ 2. screen uploaded resumes
$("screen-form").addEventListener("submit", (event) => {
  event.preventDefault();
  clearError();
  const jd = $("jd").value.trim();
  const k = kValue();
  if (!jd) return showError("Paste a job description first.", $("jd"));
  if (!fileInput.files.length) return showError("Choose at least one resume PDF.", fileInput);
  if (k == null) return showError("k must be a whole number from 1 to 100.", $("k"));

  const form = new FormData();
  form.append("jd_text", jd);
  form.append("must_have", $("must").value);
  form.append("nice_to_have", $("nice").value);
  form.append("k", String(k));
  for (const file of fileInput.files) form.append("resumes", file);
  if ($("labels").files.length) form.append("labels", $("labels").files[0]);

  run(() => fetch(`${API}/screen`, { method: "POST", body: form }));
});

// ------------------------------------------------------------------ 3. sample data
$("sample").addEventListener("click", () => {
  clearError();
  const k = kValue();
  if (k == null) return showError("k must be a whole number from 1 to 100.", $("k"));
  run(async () => {
    const res = await fetch(`${API}/sample`);
    if (!res.ok) return res;
    const sample = await res.json();
    // show the user exactly what is being screened
    $("jd").value = sample.jd_text;
    $("must").value = sample.must_have;
    $("nice").value = sample.nice_to_have;
    return fetch(`${API}/screen/sample?k=${k}`, { method: "POST" });
  });
});

// Shared request handler: loading state, error messages, rendering
async function run(makeRequest) {
  if (busy) return;
  busy = true;
  setBusy(true);
  $("output").innerHTML = `<div class="loading"><span class="spinner"></span>Scoring resumes…</div>`;
  try {
    const res = await makeRequest();
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
      throw new Error(detail || `Request failed (${res.status})`);
    }
    render(data);
  } catch (err) {
    $("output").innerHTML = emptyState();
    $("formula").hidden = true;
    showError(err instanceof TypeError ? "Cannot reach the backend. Is it running?" : err.message);
  } finally {
    busy = false;
    setBusy(false);
  }
}

// ------------------------------------------------------------------ 4. rendering
function render(data) {
  renderFormula(data);
  const job = data.job;
  const skills = (list) => (list.length ? escapeHtml(list.join(", ")) : "–");
  $("output").innerHTML = `
    <p class="meta-line">${data.candidates.length} resume${data.candidates.length === 1 ? "" : "s"}
      · must-have: ${skills(job.must_have)} · nice-to-have: ${skills(job.nice_to_have)}
      ${job.skills_auto_extracted ? "(picked from the JD)" : ""} · ${data.elapsed_seconds}s</p>
    <div>${data.candidates.map(candidateCard).join("")}</div>
    ${data.evaluation ? evaluationCard(data.evaluation, data.weights) : ""}`;
}

function renderFormula(data) {
  const w = data.weights;
  let note = "";
  if (data.llm.status === "off") {
    note = "LLM signal is off (no API key), so its 20% is shared between semantic and lexical.";
  } else if (data.llm.status === "failed") {
    note = `LLM call failed (${data.llm.error}), so its weight is shared between semantic and lexical.`;
  }
  $("formula").hidden = false;
  $("formula").innerHTML =
    `Final = <b>${fmt(w.semantic, 3)}</b>·semantic + <b>${fmt(w.lexical, 3)}</b>·lexical + ` +
    `<b>${fmt(w.llm, 3)}</b>·LLM, × (1 − penalty)` +
    `<span class="formula-note">Encoder: ${escapeHtml(data.embedding_model)}. ` +
    `Each signal is min-max normalised across these candidates first.${note ? " " + escapeHtml(note) : ""}</span>`;
}

function signal(label, sig) {
  if (!sig) {
    return `<span class="sig off"><span class="sig-top"><span>${label}</span><b>off</b></span>
      <span class="bar"><i style="width:0"></i></span><span class="sig-raw">&nbsp;</span></span>`;
  }
  const width = Math.round(Math.max(0, Math.min(1, sig.normalised)) * 100);
  return `<span class="sig" title="raw ${fmt(sig.raw, 3)} → normalised ${fmt(sig.normalised)}">
      <span class="sig-top"><span>${label}</span><b>${fmt(sig.normalised)}</b></span>
      <span class="bar"><i style="width:${width}%"></i></span>
      <span class="sig-raw">raw ${fmt(sig.raw, 3)}</span></span>`;
}

function candidateCard(c) {
  const s = c.scores;
  const flags = c.flags.map((f) =>
    `<span class="chip flag" title="${escapeHtml(FLAG_HELP[f] || "")}">${escapeHtml(FLAG_TEXT[f] || f)}</span>`
  ).join("");
  const penalty = c.penalty > 0 ? `<span class="chip">−${Math.round(c.penalty * 100)}% penalty</span>` : "";
  const hr = c.hr_label != null ? `<span class="chip">HR label ${fmt(c.hr_label, 0)}</span>` : "";
  const signals = s
    ? `<span class="signals">${signal("Semantic", s.semantic)}${signal("Lexical", s.lexical)}${signal("LLM", s.llm)}</span>`
    : `<span class="unscored">Could not be scored</span>`;

  return `
    <details class="cand${c.rank === 1 ? " top" : ""}">
      <summary>
        <span class="rank">${c.rank ?? "–"}</span>
        <span class="who"><span class="name">${escapeHtml(prettyName(c.filename))}</span>
          <span class="file">${escapeHtml(c.filename)}</span></span>
        <span class="final"><span class="num">${fmt(c.final_score, 3)}</span><span class="lbl">final</span></span>
        ${signals}
        ${flags || penalty || hr ? `<span class="chips">${flags}${penalty}${hr}</span>` : ""}
      </summary>
      <div class="cand-body">${candidateDetails(c)}</div>
    </details>`;
}

function skillList(matched, missing) {
  const ok = matched.map((s) => `<span class="skill">✓ ${escapeHtml(s)}</span>`);
  const no = missing.map((s) => `<span class="skill miss">✗ ${escapeHtml(s)}</span>`);
  return [...ok, ...no].join("") || "–";
}

function candidateDetails(c) {
  const s = c.scores;
  const rows = [];
  if (s) {
    const lx = s.lexical;
    if (lx.must_matched.length || lx.must_missing.length) rows.push(["Must-have", skillList(lx.must_matched, lx.must_missing)]);
    if (lx.nice_matched.length || lx.nice_missing.length) rows.push(["Nice-to-have", skillList(lx.nice_matched, lx.nice_missing)]);
    rows.push(["Semantic",
      `cosine ${fmt(s.semantic.raw, 3)} → normalised ${fmt(s.semantic.normalised)} ` +
      `<span class="muted">· Manhattan similarity ${fmt(s.semantic.manhattan_similarity, 4)} (comparison only)</span>`]);
    if (s.llm) {
      const r = s.llm.rubric;
      rows.push(["LLM rubric", `skills ${r.skills} · experience ${r.experience} · projects ${r.projects} · education ${r.education}`]);
      rows.push(["LLM reasoning", escapeHtml(s.llm.reasoning || "–")]);
    }
    rows.push(["Sections found", escapeHtml(c.sections_found.join(", ") || "–")]);
  }
  for (const flag of c.flags) {
    const quotes = (c.evidence[flag] || []).map((q) => `<span class="quote">${escapeHtml(q)}</span>`).join("");
    rows.push([FLAG_TEXT[flag] || flag, escapeHtml(FLAG_HELP[flag] || "") + quotes]);
  }
  return `<dl class="kv">${rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
}

function evaluationCard(ev, current) {
  const unlabelled = ev.unlabelled.length
    ? `<p class="hint">No HR label (counted as 0): ${escapeHtml(ev.unlabelled.join(", "))}</p>`
    : "";
  return `
    <section class="eval-card" aria-label="Ranking quality">
      <h3>Ranking quality · nDCG@${ev.k}</h3>
      <div class="eval-grid">
        <div class="metric"><span class="val">${fmt(ev.ndcg_current, 3)}</span>
          <span class="cap">current weights<br>${weightsText(current)}</span></div>
        <div class="metric"><span class="val">${fmt(ev.ndcg_best, 3)}</span>
          <span class="cap">best of ${ev.combinations_tried} weight combinations<br>${weightsText(ev.best_weights)}</span></div>
      </div>
      <p class="hint">1.000 = same order as HR. Based on ${ev.labelled} labelled resume${ev.labelled === 1 ? "" : "s"};
        use 30–50 per job before trusting tuned weights.</p>
      ${unlabelled}
    </section>`;
}

loadStatus();
