const STORAGE_KEYS = {
  rowRatio: "viewer-layout-row-ratio",
  colRatio: "viewer-layout-col-ratio",
  fitToPanel: "viewer-fit-to-panel",
};

const LAYOUT_LIMITS = {
  min: 0.18,
  max: 0.82,
};

const state = {
  total: 0,
  index: 0,
  configYaml: "",
  loading: false,
  fitToPanel: true,
  rowRatio: 0.58,
  colRatio: 0.5,
};

const els = {
  workspace: document.getElementById("workspace"),
  rowTop: document.getElementById("row-top"),
  rowBottom: document.getElementById("row-bottom"),
  splitterRow: document.getElementById("splitter-row"),
  splitterCol: document.getElementById("splitter-col"),
  splitterColBottom: document.getElementById("splitter-col-bottom"),
  datasetPath: document.getElementById("dataset-path"),
  sampleIndex: document.getElementById("sample-index"),
  sampleId: document.getElementById("sample-id"),
  sampleCount: document.getElementById("sample-count"),
  statusText: document.getElementById("status-text"),
  clickLabel: document.getElementById("click-label"),
  objectLabel: document.getElementById("object-label"),
  renderPanel: document.getElementById("render-panel"),
  renderCanvas: document.getElementById("render-canvas"),
  maskPanel: document.getElementById("mask-panel"),
  maskImage: document.getElementById("mask-image"),
  annotationJson: document.getElementById("annotation-json"),
  configYaml: document.getElementById("config-yaml"),
  fitToPanel: document.getElementById("fit-to-panel"),
  btnFirst: document.getElementById("btn-first"),
  btnPrev: document.getElementById("btn-prev"),
  btnNext: document.getElementById("btn-next"),
  btnLast: document.getElementById("btn-last"),
  btnCopyJson: document.getElementById("btn-copy-json"),
  btnCopyConfig: document.getElementById("btn-copy-config"),
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function loadLayoutPrefs() {
  const row = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.rowRatio));
  const col = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.colRatio));
  const fit = localStorage.getItem(STORAGE_KEYS.fitToPanel);

  if (!Number.isNaN(row)) {
    state.rowRatio = clamp(row, LAYOUT_LIMITS.min, LAYOUT_LIMITS.max);
  }
  if (!Number.isNaN(col)) {
    state.colRatio = clamp(col, LAYOUT_LIMITS.min, LAYOUT_LIMITS.max);
  }
  if (fit !== null) {
    state.fitToPanel = fit === "true";
  }
}

function saveLayoutPrefs() {
  localStorage.setItem(STORAGE_KEYS.rowRatio, String(state.rowRatio));
  localStorage.setItem(STORAGE_KEYS.colRatio, String(state.colRatio));
  localStorage.setItem(STORAGE_KEYS.fitToPanel, String(state.fitToPanel));
}

function applyLayout() {
  const topWeight = state.rowRatio;
  const bottomWeight = 1 - state.rowRatio;
  const leftWeight = state.colRatio;
  const rightWeight = 1 - state.colRatio;

  els.rowTop.style.flex = `${topWeight} 1 0%`;
  els.rowBottom.style.flex = `${bottomWeight} 1 0%`;

  for (const row of [els.rowTop, els.rowBottom]) {
    const [leftPanel, , rightPanel] = row.children;
    leftPanel.style.flex = `${leftWeight} 1 0%`;
    rightPanel.style.flex = `${rightWeight} 1 0%`;
  }
}

function applyFitMode() {
  const modeClass = state.fitToPanel ? "fit-mode" : "native-mode";
  for (const panel of [els.renderPanel, els.maskPanel]) {
    panel.classList.remove("fit-mode", "native-mode");
    panel.classList.add(modeClass);
  }
  els.fitToPanel.checked = state.fitToPanel;
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function updateNavButtons() {
  const atStart = state.index <= 0;
  const atEnd = state.index >= state.total - 1;
  els.btnFirst.disabled = atStart || state.loading;
  els.btnPrev.disabled = atStart || state.loading;
  els.btnNext.disabled = atEnd || state.loading;
  els.btnLast.disabled = atEnd || state.loading;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function drawCross(ctx, x, y, width, height) {
  const size = Math.max(10, Math.min(width, height) * 0.04);
  const lineWidth = Math.max(2, size * 0.12);

  ctx.save();
  ctx.strokeStyle = "#ff2d2d";
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";

  ctx.beginPath();
  ctx.moveTo(x - size, y);
  ctx.lineTo(x + size, y);
  ctx.moveTo(x, y - size);
  ctx.lineTo(x, y + size);
  ctx.stroke();

  ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
  ctx.lineWidth = lineWidth + 1.5;
  ctx.globalCompositeOperation = "destination-over";
  ctx.beginPath();
  ctx.moveTo(x - size, y);
  ctx.lineTo(x + size, y);
  ctx.moveTo(x, y - size);
  ctx.lineTo(x, y + size);
  ctx.stroke();
  ctx.restore();
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    img.src = url;
  });
}

async function renderImageWithCross(imagePath, point) {
  const url = `/media/${imagePath}?t=${encodeURIComponent(state.index)}`;
  const img = await loadImage(url);
  const canvas = els.renderCanvas;
  const ctx = canvas.getContext("2d");

  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  const [x, y] = point;
  drawCross(ctx, x, y, canvas.width, canvas.height);
}

async function showSample(index) {
  if (state.loading || state.total === 0) {
    return;
  }

  const clamped = Math.max(0, Math.min(index, state.total - 1));
  state.loading = true;
  updateNavButtons();
  setStatus(`Loading sample ${clamped + 1}…`);

  try {
    const payload = await fetchJson(`/api/sample/index/${clamped}`);
    const record = payload.record;
    state.index = payload.index;

    els.sampleIndex.value = String(state.index);
    els.sampleId.value = record.id;
    els.sampleCount.textContent = `/ ${state.total.toLocaleString()}`;

    const [x, y] = record.point;
    els.clickLabel.textContent = `point [${x}, ${y}]`;
    els.objectLabel.textContent = `object ${record.object_id} / ${record.num_objects}`;

    await renderImageWithCross(record.image, record.point);
    els.maskImage.src = `/media/${record.mask}?t=${state.index}`;
    els.annotationJson.textContent = JSON.stringify(record, null, 2);

    setStatus(`Sample ${state.index + 1} · id ${record.id}`);
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${error.message}`);
  } finally {
    state.loading = false;
    updateNavButtons();
  }
}

async function jumpToId(sampleId) {
  const trimmed = sampleId.trim();
  if (!trimmed) {
    return;
  }
  try {
    const payload = await fetchJson(`/api/sample/${encodeURIComponent(trimmed)}`);
    await showSample(payload.index);
  } catch (error) {
    setStatus(`Unknown id: ${trimmed}`);
  }
}

function setupSplitters() {
  bindRowSplitter(els.splitterRow, els.workspace, els.rowTop);
  bindColumnSplitter(els.splitterCol, els.rowTop);
  bindColumnSplitter(els.splitterColBottom, els.rowBottom);
}

function bindRowSplitter(splitter, container, topRow) {
  startDrag(splitter, (event) => {
    const rect = container.getBoundingClientRect();
    const splitterHeight = els.splitterRow.offsetHeight;
    const available = rect.height - splitterHeight;
    if (available <= 0) {
      return;
    }

    const y = (event.clientY ?? event.touches?.[0]?.clientY ?? 0) - rect.top;
    state.rowRatio = clamp(y / available, LAYOUT_LIMITS.min, LAYOUT_LIMITS.max);
    applyLayout();
    saveLayoutPrefs();
  });
}

function bindColumnSplitter(splitter, row) {
  startDrag(splitter, (event) => {
    const rect = row.getBoundingClientRect();
    const splitterWidth = splitter.offsetWidth;
    const available = rect.width - splitterWidth;
    if (available <= 0) {
      return;
    }

    const x = (event.clientX ?? event.touches?.[0]?.clientX ?? 0) - rect.left;
    state.colRatio = clamp(x / available, LAYOUT_LIMITS.min, LAYOUT_LIMITS.max);
    applyLayout();
    saveLayoutPrefs();
  });
}

function startDrag(splitter, onMove) {
  let dragging = false;

  const finish = () => {
    if (!dragging) {
      return;
    }
    dragging = false;
    splitter.classList.remove("dragging");
    document.body.classList.remove("dragging-splitter");
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", finish);
    window.removeEventListener("touchmove", move);
    window.removeEventListener("touchend", finish);
  };

  const move = (event) => {
    if (!dragging) {
      return;
    }
    event.preventDefault();
    onMove(event);
  };

  const begin = (event) => {
    dragging = true;
    splitter.classList.add("dragging");
    document.body.classList.add("dragging-splitter");
    onMove(event);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", finish);
    window.addEventListener("touchmove", move, { passive: false });
    window.addEventListener("touchend", finish);
  };

  splitter.addEventListener("mousedown", begin);
  splitter.addEventListener("touchstart", begin, { passive: true });
}

async function init() {
  loadLayoutPrefs();
  applyLayout();
  applyFitMode();
  setupSplitters();

  try {
    const meta = await fetchJson("/api/meta");
    state.total = meta.count;
    els.datasetPath.textContent = meta.dataset_dir;
    els.sampleIndex.max = Math.max(0, state.total - 1);
    els.sampleCount.textContent = `/ ${state.total.toLocaleString()}`;

    const config = await fetchJson("/api/config");
    state.configYaml = config.yaml;
    els.configYaml.textContent = state.configYaml;

    if (state.total === 0) {
      setStatus("Dataset is empty");
      updateNavButtons();
      return;
    }

    await showSample(0);
  } catch (error) {
    console.error(error);
    setStatus(`Failed to initialize: ${error.message}`);
  }
}

els.btnFirst.addEventListener("click", () => showSample(0));
els.btnPrev.addEventListener("click", () => showSample(state.index - 1));
els.btnNext.addEventListener("click", () => showSample(state.index + 1));
els.btnLast.addEventListener("click", () => showSample(state.total - 1));

els.sampleIndex.addEventListener("change", () => {
  const value = Number.parseInt(els.sampleIndex.value, 10);
  if (Number.isNaN(value)) {
    els.sampleIndex.value = String(state.index);
    return;
  }
  showSample(value);
});

els.sampleId.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    jumpToId(els.sampleId.value);
  }
});

els.fitToPanel.addEventListener("change", () => {
  state.fitToPanel = els.fitToPanel.checked;
  applyFitMode();
  saveLayoutPrefs();
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) {
    return;
  }
  if (event.key === "ArrowLeft" || event.key === "k") {
    event.preventDefault();
    showSample(state.index - 1);
  } else if (event.key === "ArrowRight" || event.key === "j") {
    event.preventDefault();
    showSample(state.index + 1);
  } else if (event.key === "Home") {
    event.preventDefault();
    showSample(0);
  } else if (event.key === "End") {
    event.preventDefault();
    showSample(state.total - 1);
  }
});

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    setStatus(`Copied ${label}`);
  } catch {
    setStatus(`Could not copy ${label}`);
  }
}

els.btnCopyJson.addEventListener("click", () => {
  copyText(els.annotationJson.textContent, "annotation JSON");
});

els.btnCopyConfig.addEventListener("click", () => {
  copyText(state.configYaml, "config.yaml");
});

window.addEventListener("resize", applyLayout);

init();
