// ============================================
// Image to LaTeX — Dual Mode Conversion (v2)
// ============================================

let canvas, ctx;
let isDrawing = false;
let hasDrawn = false;
let uploadedImage = null;
let results = null;
let currentInputTab = 'draw';
let isEraserMode = false;
let undoStack = [];
const MAX_UNDO = 30;

// --- Label Graph Converter (ported from Python) ---
const RELATIONSHIP_VOCABS = ['Right', 'NoRel', 'Sup', 'Sub', 'Below', 'Inside', 'Above', 'COMMA'];
const FUNCTION_VOCABS = [
  '\\sqrt', '\\sin', '\\sum', '\\int', '\\cos', '\\log', '\\lim', '\\tan',
  '\\frac', '\\alpha', '\\beta', '\\gamma', '\\delta', '\\theta', '\\pi',
  '\\sigma', '\\phi', '\\omega', '\\infty', '\\rightarrow', '\\leftarrow',
  '\\leq', '\\geq', '\\neq', '\\times', '\\div', '\\pm', '\\mp'
];

function labelToLatex(labelGraph) {
  const labels = labelGraph.split(/\s+/);
  const stack = [];
  const processed = [];
  let j = 0;
  while (j < labels.length) {
    if (j + 3 < labels.length && labels[j + 1] === 'NoRel' && labels[j + 2] === '-' && labels[j + 3] === 'Below' && !RELATIONSHIP_VOCABS.includes(labels[j])) {
      if (j + 4 < labels.length && !RELATIONSHIP_VOCABS.includes(labels[j + 4])) {
        processed.push('FRAC_START', labels[j], 'FRAC_MID', labels[j + 4], 'FRAC_END');
        j += 5;
      } else { processed.push(labels[j]); j++; }
    } else { processed.push(labels[j]); j++; }
  }
  let latex = '';
  for (let i = 0; i < processed.length; i++) {
    const l = processed[i];
    if (l === 'FRAC_START') { latex += '\\frac{'; continue; }
    if (l === 'FRAC_MID') { latex += '}{'; continue; }
    if (l === 'FRAC_END') { latex += '}'; continue; }
    if (l === 'Right') { latex += ' '; continue; }
    if (l === 'Sub') { latex += '_{'; stack.push('Sub'); continue; }
    if (l === 'Sup') { latex += '^{'; stack.push('Sup'); continue; }
    if (l === 'COMMA') { latex += ','; continue; }
    if (l === 'Above') { latex += '\\frac{'; stack.push('\\frac'); continue; }
    if (l === 'Inside') { latex += '{'; stack.push('Inside'); continue; }
    if (l === 'Below') { latex += '_{'; stack.push('Below'); continue; }
    if (l === 'NoRel') {
      const cur = stack.length ? stack[stack.length - 1] : null;
      if (cur === '\\frac') { latex += '}{'; stack.pop(); stack.push('NoRel'); }
      else if (['Below', 'Inside', 'Sub', 'Sup'].includes(cur)) { latex += '}'; stack.pop(); }
      else if (cur === 'NoRel') { latex += '}'; stack.pop(); }
      else { latex += ' '; }
      continue;
    }
    if (['\\sqrt', '\\sum', '\\int', '\\lim', '\\frac'].includes(l)) { latex += l; continue; }
    if (l === '{' || l === '}') continue;
    if (FUNCTION_VOCABS.includes(l)) { latex += l; continue; }
    if (!RELATIONSHIP_VOCABS.includes(l) && !l.startsWith('FRAC_')) { latex += l; }
  }
  while (stack.length) { latex += '}'; stack.pop(); }
  return latex;
}

// --- LaTeX Normalizer ---
const JSON_KEYS = ['latex', 'result', 'prediction', 'text', 'output', 'content'];
const MATH_DELIMS = [['$$', '$$'], ['\\[', '\\]'], ['\\(', '\\)'], ['$', '$']];

function normalizeLatex(raw) {
  if (!raw) return '';
  let t = String(raw).trim();
  if (!t) return '';
  try { const p = JSON.parse(t); if (typeof p === 'string') t = p; else if (typeof p === 'object' && p) { for (const k of JSON_KEYS) { if (typeof p[k] === 'string') { t = p[k]; break; } } } } catch {}
  if (t.length >= 2 && t[0] === t[t.length - 1] && (t[0] === "'" || t[0] === '"')) t = t.slice(1, -1);
  t = t.trim();
  const s = t.trim(); if (s.startsWith('```') && s.endsWith('```')) { const lines = s.split('\n'); t = lines.length < 3 ? s.replace(/`/g, '') : lines.slice(1, -1).join('\n'); }
  t = t.replace(/\r\n/g, '\n').replace(/\r/g, '\n').replace(/\\n/g, '\n').trim();
  let cur = t.trim();
  while (true) { let u = cur; for (const [a, b] of MATH_DELIMS) { if (u.startsWith(a) && u.endsWith(b)) { const inner = u.slice(a.length, u.length - b.length).trim(); if (inner) { u = inner; break; } } } if (u === cur) break; cur = u; }
  return cur.split('\n').map(l => l.trimEnd()).join('\n').trim();
}

// --- Image Helpers ---
function imageToHex(dataUrl) {
  const b64 = dataUrl.split(',')[1];
  const bin = atob(b64);
  return Array.from(bin, c => c.charCodeAt(0).toString(16).padStart(2, '0')).join('');
}

// --- API ---
async function callApi(hex, url, prompt, type) {
  const timeout = type === 'Label' ? 120000 : 30000;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_bytes: hex, prompt, Type: type }), signal: ctrl.signal });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const text = await res.text();
    const norm = normalizeLatex(text);
    if (!norm) throw new Error('Empty response');
    return { code: norm, error: null };
  } catch (e) { clearTimeout(timer); return { code: null, error: e.message }; }
}

async function convertBoth(hex, url, prompt) {
  updateLoader('Converting with Pix2Text (Latex)…', 20);
  const latex = await callApi(hex, url, '', 'Latex');
  updateLoader('Converting with Qwen3-VL (Label)…', 60);
  const label = await callApi(hex, url, prompt, 'Label');
  updateLoader('Done!', 100);
  return { latex, label };
}

// ============================================
// DOM Init
// ============================================
document.addEventListener('DOMContentLoaded', () => {
  initDarkMode();
  initCanvas();
  initTabs();
  initUpload();
  initConvert();
  initResultTabs();
  initSettings();
  initKeyboard();
});

// --- Dark Mode ---
function initDarkMode() {
  const btn = document.getElementById('darkModeBtn');
  const sunIcon = document.getElementById('sunIcon');
  const moonIcon = document.getElementById('moonIcon');
  const saved = localStorage.getItem('latex_theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    applyTheme('dark');
  }

  btn.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    applyTheme(isDark ? 'light' : 'dark');
  });

  function applyTheme(theme) {
    if (theme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
      sunIcon.classList.add('hidden');
      moonIcon.classList.remove('hidden');
    } else {
      document.documentElement.removeAttribute('data-theme');
      sunIcon.classList.remove('hidden');
      moonIcon.classList.add('hidden');
    }
    localStorage.setItem('latex_theme', theme);
  }
}

// --- Canvas ---
function initCanvas() {
  canvas = document.getElementById('drawCanvas');
  ctx = canvas.getContext('2d');
  resetCanvas();
  saveUndoState();

  canvas.addEventListener('mousedown', startDraw);
  canvas.addEventListener('mousemove', drawing);
  canvas.addEventListener('mouseup', stopDraw);
  canvas.addEventListener('mouseout', stopDraw);
  canvas.addEventListener('touchstart', e => { e.preventDefault(); startDraw(touchEvt(e)); }, { passive: false });
  canvas.addEventListener('touchmove', e => { e.preventDefault(); drawing(touchEvt(e)); }, { passive: false });
  canvas.addEventListener('touchend', stopDraw);

  const penSize = document.getElementById('penSize');
  const penSizeVal = document.getElementById('penSizeValue');
  penSize.addEventListener('input', () => { ctx.lineWidth = penSize.value; penSizeVal.textContent = penSize.value; });
  document.getElementById('penColor').addEventListener('input', e => { if (!isEraserMode) ctx.strokeStyle = e.target.value; });
  document.getElementById('clearBtn').addEventListener('click', () => { resetCanvas(); hasDrawn = false; undoStack = []; saveUndoState(); showCanvasHint(); });
  document.getElementById('undoBtn').addEventListener('click', undo);

  // Pen / Eraser toggle
  const penBtn = document.getElementById('penModeBtn');
  const eraserBtn = document.getElementById('eraserModeBtn');
  penBtn.addEventListener('click', () => setToolMode('pen'));
  eraserBtn.addEventListener('click', () => setToolMode('eraser'));
}

function setToolMode(mode) {
  isEraserMode = mode === 'eraser';
  const penBtn = document.getElementById('penModeBtn');
  const eraserBtn = document.getElementById('eraserModeBtn');
  const colorGroup = document.getElementById('colorGroup');
  penBtn.classList.toggle('active', !isEraserMode);
  eraserBtn.classList.toggle('active', isEraserMode);
  canvas.classList.toggle('eraser-active', isEraserMode);
  if (colorGroup) colorGroup.style.opacity = isEraserMode ? '.4' : '1';

  if (isEraserMode) {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.strokeStyle = 'rgba(0,0,0,1)';
  } else {
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = document.getElementById('penColor').value;
  }
}

function resetCanvas() {
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  // Draw subtle grid
  ctx.strokeStyle = '#f0ede8';
  ctx.lineWidth = 0.5;
  const step = 40;
  for (let x = step; x < canvas.width; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }
  for (let y = step; y < canvas.height; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }
  // Restore drawing settings
  ctx.lineWidth = document.getElementById('penSize')?.value || 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = document.getElementById('penColor')?.value || '#1a1a2e';
  if (isEraserMode) setToolMode('eraser');
}

// --- Undo ---
function saveUndoState() {
  if (undoStack.length >= MAX_UNDO) undoStack.shift();
  undoStack.push(canvas.toDataURL());
}

function undo() {
  if (undoStack.length <= 1) { toast('Nothing to undo'); return; }
  undoStack.pop(); // remove current
  const prev = undoStack[undoStack.length - 1];
  const img = new Image();
  img.onload = () => {
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    if (isEraserMode) {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.strokeStyle = 'rgba(0,0,0,1)';
    } else {
      ctx.strokeStyle = document.getElementById('penColor').value;
    }
    ctx.lineWidth = document.getElementById('penSize').value;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  };
  img.src = prev;
}

function showCanvasHint() { document.getElementById('canvasHint')?.classList.remove('hidden'); }
function hideCanvasHint() { document.getElementById('canvasHint')?.classList.add('hidden'); }

function getPos(e) {
  const r = canvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) * (canvas.width / r.width), y: (e.clientY - r.top) * (canvas.height / r.height) };
}
function startDraw(e) { isDrawing = true; if (!hasDrawn) { hasDrawn = true; hideCanvasHint(); } const p = getPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); }
function drawing(e) { if (!isDrawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); }
function stopDraw() { if (isDrawing) { isDrawing = false; saveUndoState(); } }
function touchEvt(e) { const t = e.touches[0]; return { clientX: t.clientX, clientY: t.clientY }; }

// --- Input Tabs ---
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const tab = btn.dataset.tab;
      document.getElementById(`${tab}-panel`).classList.add('active');
      currentInputTab = tab;
    });
  });
}

// --- Upload ---
function initUpload() {
  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('imageUpload');
  const preview = document.getElementById('previewContainer');
  const previewImg = document.getElementById('imagePreview');
  const meta = document.getElementById('previewMeta');

  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('dragover'); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });

  document.getElementById('removeImageBtn').addEventListener('click', () => {
    uploadedImage = null; input.value = '';
    zone.classList.remove('hidden'); preview.classList.add('hidden');
  });

  function handleFile(file) {
    if (!file.type.match('image/(png|jpeg|jpg)')) { showError('Upload a PNG or JPEG image.'); return; }
    if (file.size > 10 * 1024 * 1024) { showError('Max 10 MB.'); return; }
    const reader = new FileReader();
    reader.onload = e => {
      uploadedImage = e.target.result;
      previewImg.src = uploadedImage;
      zone.classList.add('hidden');
      preview.classList.remove('hidden');
      const img = new Image();
      img.onload = () => { meta.innerHTML = `<span>${file.type.split('/')[1].toUpperCase()}</span><span>${img.width}×${img.height}</span><span>${(file.size / 1024).toFixed(1)} KB</span>`; };
      img.src = uploadedImage;
    };
    reader.readAsDataURL(file);
  }
}

// --- Convert ---
function initConvert() {
  const btn = document.getElementById('convertBtn');
  btn.addEventListener('click', doConvert);

  document.querySelectorAll('.copy-btn').forEach(b => {
    b.addEventListener('click', () => {
      const text = document.getElementById(b.dataset.target)?.textContent;
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        b.classList.add('copied'); b.textContent = '✅';
        toast('Copied!', 'success');
        setTimeout(() => { b.classList.remove('copied'); b.textContent = '📋'; }, 2000);
      }).catch(() => showError('Copy failed.'));
    });
  });

  document.getElementById('errorClose').addEventListener('click', () => document.getElementById('errorBanner').classList.add('hidden'));
  document.getElementById('clearResultsBtn').addEventListener('click', clearResults);
}

async function doConvert() {
  let dataUrl;
  if (currentInputTab === 'draw') {
    if (!hasDrawn) { showError('Draw something on the canvas first.'); return; }
    dataUrl = canvas.toDataURL('image/png');
  } else {
    if (!uploadedImage) { showError('Upload an image first.'); return; }
    dataUrl = uploadedImage;
  }

  hideResults();
  setStep(2);
  const btn = document.getElementById('convertBtn');
  const btnText = document.getElementById('convertBtnText');
  btn.disabled = true;
  btnText.textContent = 'Converting…';
  showLoader();

  try {
    const hex = imageToHex(dataUrl);
    const url = document.getElementById('apiUrl').value;
    const prompt = document.getElementById('promptText').value;
    results = await convertBoth(hex, url, prompt);
    hideLoader();
    setStep(3);
    renderResults(results);
  } catch (err) {
    hideLoader();
    showError(err.message);
    setStep(1);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Convert to LaTeX';
  }
}

// --- Keyboard ---
function initKeyboard() {
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      doConvert();
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
      e.preventDefault();
      undo();
    }
    if (e.key === 'Escape') closeSettings();
  });
}

// --- Step Indicator ---
function setStep(n) {
  document.querySelectorAll('.step').forEach(s => {
    const sn = parseInt(s.dataset.step);
    s.classList.remove('active', 'completed');
    if (sn === n) s.classList.add('active');
    else if (sn < n) s.classList.add('completed');
  });
  document.querySelectorAll('.step-connector').forEach((c, i) => {
    c.classList.toggle('active', i < n - 1);
  });
}

// --- Result Tabs ---
function initResultTabs() {
  document.querySelectorAll('.rtab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.rtab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.rpanel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`panel-${btn.dataset.rtab}`).classList.add('active');
      retypeset();
    });
  });
}

// --- Render Results ---
function renderResults(r) {
  document.getElementById('emptyState').classList.add('hidden');
  document.getElementById('resultContent').classList.remove('hidden');

  // Latex
  setText('latexCodeLatex', r.latex.error ? `Error: ${r.latex.error}` : r.latex.code || '');
  setText('latexLenLatex', r.latex.code ? `${r.latex.code.length} chars` : '');
  renderMath('renderLatex', r.latex.error ? '' : r.latex.code);

  // Label
  if (r.label.error) {
    setText('labelGraphCode', `Error: ${r.label.error}`);
    setText('labelConvertedCode', ''); setText('labelConvertedLen', '');
    renderMath('renderLabel', '');
  } else if (r.label.code) {
    setText('labelGraphCode', r.label.code);
    let converted = '';
    try { converted = labelToLatex(r.label.code); } catch (ex) { converted = `Error: ${ex.message}`; }
    setText('labelConvertedCode', converted);
    setText('labelConvertedLen', converted ? `${converted.length} chars` : '');
    renderMath('renderLabel', converted);
  }

  // Compare
  setText('cmpLatexCode', r.latex.code || 'No result');
  setText('cmpLatexLen', r.latex.code ? `${r.latex.code.length} chars` : '');
  renderMath('cmpRenderLatex', r.latex.code || '');

  let cmpLabelLatex = '';
  if (r.label.code) {
    setText('cmpLabelGraph', r.label.code);
    try { cmpLabelLatex = labelToLatex(r.label.code); } catch {}
    setText('cmpLabelLatex', cmpLabelLatex);
    setText('cmpLabelLen', cmpLabelLatex ? `${cmpLabelLatex.length} chars` : '');
    renderMath('cmpRenderLabel', cmpLabelLatex);
  } else {
    setText('cmpLabelGraph', 'No result');
    setText('cmpLabelLatex', '');
    setText('cmpLabelLen', '');
    setHtml('cmpRenderLabel', '<span style="color:var(--text-3)">No result</span>');
  }

  // Metrics
  if (r.latex.code && r.label.code) {
    const diff = r.label.code.length - r.latex.code.length;
    setText('metricLenDiff', `${Math.abs(diff)}`);
    setText('metricLenNote', `${diff > 0 ? 'Label' : 'Latex'} is longer`);
    if (cmpLabelLatex) {
      const match = r.latex.code.trim() === cmpLabelLatex.trim();
      const el = document.getElementById('metricMatch');
      el.textContent = match ? 'Same' : 'Different';
      el.style.color = match ? 'var(--green)' : 'var(--coral)';
      setText('metricMatchNote', 'After label→LaTeX conversion');
    } else { setText('metricMatch', '—'); setText('metricMatchNote', 'Conversion failed'); }
  }

  // Scroll to results
  document.getElementById('resultContent').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setText(id, t) { const el = document.getElementById(id); if (el) el.textContent = t; }
function setHtml(id, h) { const el = document.getElementById(id); if (el) el.innerHTML = h; }

function renderMath(id, latex) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!latex) { el.innerHTML = '<span style="color:var(--text-3)">—</span>'; return; }
  el.innerHTML = `\\[${latex}\\]`;
  retypeset(el);
}
function retypeset(el) {
  if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise(el ? [el] : undefined).catch(() => {});
}

// --- UI Helpers ---
function showLoader() { document.getElementById('loadingState').classList.remove('hidden'); document.getElementById('emptyState').classList.add('hidden'); document.getElementById('progressBar').style.width = '5%'; }
function hideLoader() { document.getElementById('loadingState').classList.add('hidden'); }
function updateLoader(text, pct) { document.getElementById('loaderText').textContent = text; document.getElementById('progressBar').style.width = `${pct}%`; }
function showError(msg) { document.getElementById('errorText').textContent = msg; document.getElementById('errorBanner').classList.remove('hidden'); }
function hideResults() {
  document.getElementById('resultContent').classList.add('hidden');
  document.getElementById('errorBanner').classList.add('hidden');
  document.getElementById('loadingState').classList.add('hidden');
}
function clearResults() {
  results = null;
  hideResults();
  document.getElementById('emptyState').classList.remove('hidden');
  setStep(1);
}

function toast(msg, type = '') {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3200);
}

// --- Settings Panel ---
function initSettings() {
  const btn = document.getElementById('settingsBtn');
  const panel = document.getElementById('settingsPanel');
  const backdrop = document.getElementById('settingsBackdrop');
  const close = document.getElementById('settingsClose');

  btn.addEventListener('click', openSettings);
  close.addEventListener('click', closeSettings);
  backdrop.addEventListener('click', closeSettings);

  // Persist
  const apiInput = document.getElementById('apiUrl');
  const promptInput = document.getElementById('promptText');
  const saved = localStorage.getItem('latex_apiUrl');
  const savedPrompt = localStorage.getItem('latex_prompt');
  if (saved) apiInput.value = saved;
  if (savedPrompt) promptInput.value = savedPrompt;

  apiInput.addEventListener('change', () => localStorage.setItem('latex_apiUrl', apiInput.value));
  promptInput.addEventListener('change', () => localStorage.setItem('latex_prompt', promptInput.value));

  document.getElementById('testConnectionBtn').addEventListener('click', testConnection);
}

function openSettings() {
  document.getElementById('settingsPanel').classList.remove('hidden');
  document.getElementById('settingsBackdrop').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}
function closeSettings() {
  document.getElementById('settingsPanel').classList.add('hidden');
  document.getElementById('settingsBackdrop').classList.add('hidden');
  document.body.style.overflow = '';
}

async function testConnection() {
  const btn = document.getElementById('testConnectionBtn');
  const url = document.getElementById('apiUrl').value;
  btn.disabled = true;
  btn.textContent = 'Testing…';
  try {
    const res = await fetch(url.replace('/predict', '/health'), { method: 'GET' });
    toast(res.ok ? 'Connected!' : `Status: ${res.status}`, res.ok ? 'success' : 'error');
    btn.textContent = res.ok ? '✅ Connected' : `⚠️ ${res.status}`;
  } catch {
    toast('Connection failed', 'error');
    btn.textContent = '❌ Failed';
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Test Connection'; }, 3000);
}
