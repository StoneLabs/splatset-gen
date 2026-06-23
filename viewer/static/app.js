const STORAGE_KEYS = {
  rowRatio: "viewer-layout-row-ratio",
  colRatio: "viewer-layout-col-ratio",
  topCol1: "viewer-layout-top-col1",
  topCol2: "viewer-layout-top-col2",
  fitToPanel: "viewer-fit-to-panel",
  maskOpacity: "viewer-mask-opacity",
};

const LAYOUT_LIMITS = {
  min: 0.15,
  max: 0.7,
};

const state = {
  total: 0,
  index: 0,
  configYaml: "",
  loading: false,
  fitToPanel: true,
  rowRatio: 0.58,
  colRatio: 0.5,
  topCol1: 0.33,
  topCol2: 0.34,
  maskOpacity: 0.5,
  imageCache: null,
};

const els = {
  workspace: document.getElementById("workspace"),
  rowTop: document.getElementById("row-top"),
  rowBottom: document.getElementById("row-bottom"),
  splitterRow: document.getElementById("splitter-row"),
  splitterCol: document.getElementById("splitter-col"),
  splitterColOverlay: document.getElementById("splitter-col-overlay"),
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
  overlayPanel: document.getElementById("overlay-panel"),
  overlayCanvas: document.getElementById("overlay-canvas"),
  maskOpacity: document.getElementById("mask-opacity"),
  maskOpacityLabel: document.getElementById("mask-opacity-label"),
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

function getRowPanels(row) {
  return [...row.children].filter((child) => child.classList.contains("panel"));
}

function getVerticalSplitterWidth(row) {
  return [...row.children]
    .filter((child) => child.classList.contains("splitter-v"))
    .reduce((total, splitter) => total + splitter.offsetWidth, 0);
}

function loadLayoutPrefs() {
  const row = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.rowRatio));
  const col = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.colRatio));
  const topCol1 = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.topCol1));
  const topCol2 = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.topCol2));
  const fit = localStorage.getItem(STORAGE_KEYS.fitToPanel);
  const maskOpacity = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.maskOpacity));

  if (!Number.isNaN(row)) {
    state.rowRatio = clamp(row, LAYOUT_LIMITS.min, LAYOUT_LIMITS.max);
  }
  if (!Number.isNaN(col)) {
    state.colRatio = clamp(col, LAYOUT_LIMITS.min, LAYOUT_LIMITS.max);
  }
  if (!Number.isNaN(topCol1)) {
    state.topCol1 = topCol1;
  }
  if (!Number.isNaN(topCol2)) {
    state.topCol2 = topCol2;
  }
  if (fit !== null) {
    state.fitToPanel = fit === "true";
  }
  if (!Number.isNaN(maskOpacity)) {
    state.maskOpacity = clamp(maskOpacity, 0, 1);
  }

  normalizeTopColumns();
}

function normalizeTopColumns() {
  state.topCol1 = clamp(state.topCol1, LAYOUT_LIMITS.min, LAYOUT_LIMITS.max);
  state.topCol2 = clamp(state.topCol2, LAYOUT_LIMITS.min, LAYOUT_LIMITS.max);
  const maxSecond = 1 - state.topCol1 - LAYOUT_LIMITS.min;
  state.topCol2 = clamp(state.topCol2, LAYOUT_LIMITS.min, maxSecond);
  const overflow = state.topCol1 + state.topCol2 - (1 - LAYOUT_LIMITS.min);
  if (overflow > 0) {
    state.topCol2 -= overflow;
  }
}

function saveLayoutPrefs() {
  localStorage.setItem(STORAGE_KEYS.rowRatio, String(state.rowRatio));
  localStorage.setItem(STORAGE_KEYS.colRatio, String(state.colRatio));
  localStorage.setItem(STORAGE_KEYS.topCol1, String(state.topCol1));
  localStorage.setItem(STORAGE_KEYS.topCol2, String(state.topCol2));
  localStorage.setItem(STORAGE_KEYS.fitToPanel, String(state.fitToPanel));
  localStorage.setItem(STORAGE_KEYS.maskOpacity, String(state.maskOpacity));
}

function applyLayout() {
  const topWeight = state.rowRatio;
  const bottomWeight = 1 - state.rowRatio;
  const leftWeight = state.colRatio;
  const rightWeight = 1 - state.colRatio;

  els.rowTop.style.flex = `${topWeight} 1 0%`;
  els.rowBottom.style.flex = `${bottomWeight} 1 0%`;

  const topPanels = getRowPanels(els.rowTop);
  const topCol3 = 1 - state.topCol1 - state.topCol2;
  topPanels[0].style.flex = `${state.topCol1} 1 0%`;
  topPanels[1].style.flex = `${state.topCol2} 1 0%`;
  topPanels[2].style.flex = `${topCol3} 1 0%`;

  const bottomPanels = getRowPanels(els.rowBottom);
  bottomPanels[0].style.flex = `${leftWeight} 1 0%`;
  bottomPanels[1].style.flex = `${rightWeight} 1 0%`;
}

function applyFitMode() {
  const modeClass = state.fitToPanel ? "fit-mode" : "native-mode";
  for (const panel of [els.renderPanel, els.maskPanel, els.overlayPanel]) {
    panel.classList.remove("fit-mode", "native-mode");
    panel.classList.add(modeClass);
  }
  els.fitToPanel.checked = state.fitToPanel;
}

function applyMaskOpacityUi() {
  const percent = Math.round(state.maskOpacity * 100);
  els.maskOpacity.value = String(percent);
  els.maskOpacityLabel.textContent = `${percent}%`;
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

async function renderImageWithCross(rgbImage, point) {
  const canvas = els.renderCanvas;
  const ctx = canvas.getContext("2d");

  canvas.width = rgbImage.naturalWidth;
  canvas.height = rgbImage.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(rgbImage, 0, 0);

  const [x, y] = point;
  drawCross(ctx, x, y, canvas.width, canvas.height);
}

function renderMaskOverlay(rgbImage, maskImage, opacity) {
  const canvas = els.overlayCanvas;
  const ctx = canvas.getContext("2d");
  const width = rgbImage.naturalWidth;
  const height = rgbImage.naturalHeight;

  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(rgbImage, 0, 0);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext("2d");
  maskCtx.drawImage(maskImage, 0, 0);

  const maskPixels = maskCtx.getImageData(0, 0, width, height);
  const overlayPixels = maskCtx.createImageData(width, height);
  const alphaScale = opacity * 255;

  for (let i = 0; i < maskPixels.data.length; i += 4) {
    const maskValue = maskPixels.data[i] / 255;
    if (maskValue <= 0) {
      continue;
    }
    overlayPixels.data[i] = 255;
    overlayPixels.data[i + 1] = 0;
    overlayPixels.data[i + 2] = 0;
    overlayPixels.data[i + 3] = Math.round(maskValue * alphaScale);
  }

  maskCtx.putImageData(overlayPixels, 0, 0);
  ctx.drawImage(maskCanvas, 0, 0);
}

async function loadSampleImages(imagePath, maskPath) {
  const cacheKey = `${state.index}:${imagePath}:${maskPath}`;
  if (state.imageCache?.key === cacheKey) {
    return state.imageCache;
  }

  const rgbUrl = `/media/${imagePath}?t=${encodeURIComponent(state.index)}`;
  const maskUrl = `/media/${maskPath}?t=${encodeURIComponent(state.index)}`;
  const [rgbImage, maskImage] = await Promise.all([loadImage(rgbUrl), loadImage(maskUrl)]);

  state.imageCache = {
    key: cacheKey,
    rgbImage,
    maskImage,
  };
  return state.imageCache;
}

function refreshOverlay() {
  if (!state.imageCache) {
    return;
  }
  renderMaskOverlay(state.imageCache.rgbImage, state.imageCache.maskImage, state.maskOpacity);
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
    state.imageCache = null;

    els.sampleIndex.value = String(state.index);
    els.sampleId.value = record.id;
    els.sampleCount.textContent = `/ ${state.total.toLocaleString()}`;

    const [x, y] = record.point;
    els.clickLabel.textContent = `point [${x}, ${y}]`;
    els.objectLabel.textContent = `object ${record.object_id} / ${record.num_objects}`;

    const images = await loadSampleImages(record.image, record.mask);
    renderImageWithCross(images.rgbImage, record.point);
    els.maskImage.src = `/media/${record.mask}?t=${state.index}`;
    renderMaskOverlay(images.rgbImage, images.maskImage, state.maskOpacity);
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
  bindRowSplitter(els.splitterRow, els.workspace);
  bindTopColumnSplitter(els.splitterCol, 0);
  bindTopColumnSplitter(els.splitterColOverlay, 1);
  bindBottomColumnSplitter(els.splitterColBottom, els.rowBottom);
}

function bindRowSplitter(splitter, container) {
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

function bindTopColumnSplitter(splitter, splitterIndex) {
  startDrag(splitter, (event) => {
    const row = els.rowTop;
    const rect = row.getBoundingClientRect();
    const available = rect.width - getVerticalSplitterWidth(row);
    if (available <= 0) {
      return;
    }

    const x = (event.clientX ?? event.touches?.[0]?.clientX ?? 0) - rect.left;
    const ratio = clamp(x / available, LAYOUT_LIMITS.min, 1 - 2 * LAYOUT_LIMITS.min);

    if (splitterIndex === 0) {
      state.topCol1 = clamp(ratio, LAYOUT_LIMITS.min, state.topCol1 + state.topCol2 - LAYOUT_LIMITS.min);
    } else {
      const combined = clamp(ratio, state.topCol1 + LAYOUT_LIMITS.min, 1 - LAYOUT_LIMITS.min);
      state.topCol2 = combined - state.topCol1;
    }

    normalizeTopColumns();
    applyLayout();
    saveLayoutPrefs();
  });
}

function bindBottomColumnSplitter(splitter, row) {
  startDrag(splitter, (event) => {
    const rect = row.getBoundingClientRect();
    const available = rect.width - getVerticalSplitterWidth(row);
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
  applyMaskOpacityUi();
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

els.maskOpacity.addEventListener("input", () => {
  state.maskOpacity = Number.parseInt(els.maskOpacity.value, 10) / 100;
  applyMaskOpacityUi();
  refreshOverlay();
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
