const STORAGE_KEYS = {
  rowRatio: "viewer-layout-row-ratio",
  colRatio: "viewer-layout-col-ratio",
  topCol1: "viewer-layout-top-col1",
  topCol2: "viewer-layout-top-col2",
  topCol3: "viewer-layout-top-col3",
  bottomCol1: "viewer-layout-bottom-col1",
  bottomCol2: "viewer-layout-bottom-col2",
  bottomCol3: "viewer-layout-bottom-col3",
  fitToPanel: "viewer-fit-to-panel",
  maskOpacity: "viewer-mask-opacity",
  datasetName: "viewer-dataset-name",
  aiOutputFormat: "viewer-ai-output-format",
  aiDisplayMode: "viewer-ai-display-mode",
  aiComposeMode: "viewer-ai-compose-mode",
  aiMaskOpacity: "viewer-ai-mask-opacity",
  aiAutoRun: "viewer-ai-auto-run",
  aiMaskBlackBackground: "viewer-ai-mask-black-bg",
};

const LAYOUT_LIMITS = {
  min: 0.12,
  max: 0.7,
};

const ZOOM_LIMITS = {
  min: 0.25,
  max: 8,
};

const panelZoom = {
  render: { scale: 1, panX: 0, panY: 0 },
  mask: { scale: 1, panX: 0, panY: 0 },
  ai: { scale: 1, panX: 0, panY: 0 },
};

const state = {
  total: 0,
  index: 0,
  configYaml: "",
  trainingConfigYaml: "",
  hasTrainingConfig: false,
  loading: false,
  fitToPanel: true,
  rowRatio: 0.58,
  colRatio: 0.5,
  topCol1: 0.33,
  topCol2: 0.34,
  topCol3: 0.33,
  bottomCol1: 0.34,
  bottomCol2: 0.33,
  bottomCol3: 0.33,
  maskOpacity: 0.5,
  gtViewMode: "mask",
  modelLoaded: false,
  modelMetadata: null,
  aiPredictionUrl: null,
  aiAlphaImage: null,
  aiOutputFormat: "alpha",
  aiDisplayMode: "raw",
  aiComposeMode: "mask",
  aiMaskOpacity: 0.5,
  aiMaskThreshold: 0.5,
  aiMaskBlackBackground: false,
  aiMetrics: null,
  imageCache: null,
  selectedDataset: null,
  mediaGeneration: 0,
  aiAutoRun: false,
};

const els = {
  workspace: document.getElementById("workspace"),
  rowTop: document.getElementById("row-top"),
  rowBottom: document.getElementById("row-bottom"),
  splitterRow: document.getElementById("splitter-row"),
  splitterCol: document.getElementById("splitter-col"),
  splitterColAi: document.getElementById("splitter-col-ai"),
  splitterColBottom: document.getElementById("splitter-col-bottom"),
  splitterColBottom2: document.getElementById("splitter-col-bottom-2"),
  datasetSelect: document.getElementById("dataset-select"),
  btnReloadDataset: document.getElementById("btn-reload-dataset"),
  btnReloadModel: document.getElementById("btn-reload-model"),
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
  maskOverlayCanvas: document.getElementById("mask-overlay-canvas"),
  maskSplitCanvas: document.getElementById("mask-split-canvas"),
  gtViewButtons: [...document.querySelectorAll("[data-gt-view]")],
  gtOverlayOpacity: document.getElementById("gt-overlay-opacity"),
  aiPanel: document.getElementById("ai-panel"),
  aiPredictionImage: document.getElementById("ai-prediction-image"),
  aiPredictionCanvas: document.getElementById("ai-prediction-canvas"),
  aiPlaceholder: document.getElementById("ai-placeholder"),
  aiModelLabel: document.getElementById("ai-model-label"),
  aiPanelControls: document.getElementById("ai-panel-controls"),
  aiFormatButtons: [...document.querySelectorAll("[data-ai-format]")],
  aiModeButtons: [...document.querySelectorAll("[data-ai-mode]")],
  aiMetricsBar: document.getElementById("ai-metrics-bar"),
  aiF1Label: document.getElementById("ai-f1-label"),
  aiComposeButtons: [...document.querySelectorAll("[data-ai-compose]")],
  aiComposeControls: document.getElementById("ai-compose-controls"),
  aiOverlayOpacity: document.getElementById("ai-overlay-opacity"),
  aiMaskBgControls: document.getElementById("ai-mask-bg-controls"),
  aiMaskBgButtons: [...document.querySelectorAll("[data-ai-mask-bg]")],
  aiMaskOpacity: document.getElementById("ai-mask-opacity"),
  aiMaskOpacityLabel: document.getElementById("ai-mask-opacity-label"),
  maskOpacity: document.getElementById("mask-opacity"),
  maskOpacityLabel: document.getElementById("mask-opacity-label"),
  annotationJson: document.getElementById("annotation-json"),
  configYaml: document.getElementById("config-yaml"),
  trainingConfigPanel: document.getElementById("training-config-panel"),
  trainingConfigYaml: document.getElementById("training-config-yaml"),
  fitToPanel: document.getElementById("fit-to-panel"),
  btnFirst: document.getElementById("btn-first"),
  btnPrev: document.getElementById("btn-prev"),
  btnNext: document.getElementById("btn-next"),
  btnLast: document.getElementById("btn-last"),
  btnCopyJson: document.getElementById("btn-copy-json"),
  btnCopyConfig: document.getElementById("btn-copy-config"),
  btnCopyTrainingConfig: document.getElementById("btn-copy-training-config"),
  btnRunAi: document.getElementById("btn-run-ai"),
  errorDialogBackdrop: document.getElementById("error-dialog-backdrop"),
  errorDialogTitle: document.getElementById("error-dialog-title"),
  errorDialogMessage: document.getElementById("error-dialog-message"),
  errorDialogClose: document.getElementById("error-dialog-close"),
  errorDialogOk: document.getElementById("error-dialog-ok"),
};

function resetPanelZoom(panelId) {
  panelZoom[panelId] = { scale: 1, panX: 0, panY: 0 };
  applyPanelZoom(panelId);
}

function resetAllPanelZoom() {
  for (const panelId of Object.keys(panelZoom)) {
    resetPanelZoom(panelId);
  }
}

function applyPanelZoom(panelId) {
  const content = document.querySelector(`[data-zoom-content="${panelId}"]`);
  if (!content) {
    return;
  }
  const { scale, panX, panY } = panelZoom[panelId];
  content.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
}

function setupPanelZoom(viewport, panelId) {
  let dragging = false;
  let startX = 0;
  let startY = 0;
  let startPanX = 0;
  let startPanY = 0;

  viewport.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const zoom = panelZoom[panelId];
      const rect = viewport.getBoundingClientRect();
      const cursorX = event.clientX - rect.left;
      const cursorY = event.clientY - rect.top;
      const factor = Math.exp(-event.deltaY * 0.0015);
      const nextScale = clamp(zoom.scale * factor, ZOOM_LIMITS.min, ZOOM_LIMITS.max);
      const ratio = nextScale / zoom.scale;
      zoom.panX = cursorX - ratio * (cursorX - zoom.panX);
      zoom.panY = cursorY - ratio * (cursorY - zoom.panY);
      zoom.scale = nextScale;
      applyPanelZoom(panelId);
    },
    { passive: false },
  );

  viewport.addEventListener("mousedown", (event) => {
    if (event.button !== 0) {
      return;
    }
    dragging = true;
    viewport.classList.add("dragging");
    startX = event.clientX;
    startY = event.clientY;
    startPanX = panelZoom[panelId].panX;
    startPanY = panelZoom[panelId].panY;
    event.preventDefault();
  });

  window.addEventListener("mousemove", (event) => {
    if (!dragging) {
      return;
    }
    panelZoom[panelId].panX = startPanX + (event.clientX - startX);
    panelZoom[panelId].panY = startPanY + (event.clientY - startY);
    applyPanelZoom(panelId);
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) {
      return;
    }
    dragging = false;
    viewport.classList.remove("dragging");
  });

  viewport.addEventListener("dblclick", () => {
    resetPanelZoom(panelId);
  });
}

function setupPanelZoomControls() {
  for (const viewport of document.querySelectorAll("[data-zoom-panel]")) {
    setupPanelZoom(viewport, viewport.dataset.zoomPanel);
  }
}

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
  const topCol3 = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.topCol3));
  const bottomCol1 = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.bottomCol1));
  const bottomCol2 = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.bottomCol2));
  const bottomCol3 = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.bottomCol3));
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
  if (!Number.isNaN(topCol3)) {
    state.topCol3 = topCol3;
  }
  if (!Number.isNaN(bottomCol1)) {
    state.bottomCol1 = bottomCol1;
  }
  if (!Number.isNaN(bottomCol2)) {
    state.bottomCol2 = bottomCol2;
  }
  if (!Number.isNaN(bottomCol3)) {
    state.bottomCol3 = bottomCol3;
  }
  const legacySum = state.topCol1 + state.topCol2 + state.topCol3;
  if (legacySum > 0 && legacySum < 0.95) {
    const legacyAi = 1 - legacySum;
    state.topCol2 = state.topCol2 + state.topCol3;
    state.topCol3 = legacyAi;
  }
  if (fit !== null) {
    state.fitToPanel = fit === "true";
  }
  if (!Number.isNaN(maskOpacity)) {
    state.maskOpacity = clamp(maskOpacity, 0, 1);
  }

  normalizeTopColumns();
  normalizeBottomColumns();
}

function normalizeBottomColumns() {
  const min = LAYOUT_LIMITS.min;
  state.bottomCol1 = clamp(state.bottomCol1, min, 1 - 2 * min);
  state.bottomCol2 = clamp(state.bottomCol2, min, 1 - state.bottomCol1 - min);
  state.bottomCol3 = clamp(1 - state.bottomCol1 - state.bottomCol2, min, 1 - state.bottomCol1 - min);
}

function normalizeTopColumns() {
  const min = LAYOUT_LIMITS.min;
  state.topCol1 = clamp(state.topCol1, min, 1 - 2 * min);
  state.topCol2 = clamp(state.topCol2, min, 1 - state.topCol1 - min);
  state.topCol3 = clamp(1 - state.topCol1 - state.topCol2, min, 1 - state.topCol1 - min);
}

function saveLayoutPrefs() {
  localStorage.setItem(STORAGE_KEYS.rowRatio, String(state.rowRatio));
  localStorage.setItem(STORAGE_KEYS.colRatio, String(state.colRatio));
  localStorage.setItem(STORAGE_KEYS.topCol1, String(state.topCol1));
  localStorage.setItem(STORAGE_KEYS.topCol2, String(state.topCol2));
  localStorage.setItem(STORAGE_KEYS.topCol3, String(state.topCol3));
  localStorage.setItem(STORAGE_KEYS.bottomCol1, String(state.bottomCol1));
  localStorage.setItem(STORAGE_KEYS.bottomCol2, String(state.bottomCol2));
  localStorage.setItem(STORAGE_KEYS.bottomCol3, String(state.bottomCol3));
  localStorage.setItem(STORAGE_KEYS.fitToPanel, String(state.fitToPanel));
  localStorage.setItem(STORAGE_KEYS.maskOpacity, String(state.maskOpacity));
}

const AI_FORMATS = new Set(["alpha", "binary"]);
const AI_DISPLAY_MODES = new Set(["raw", "compare"]);
const AI_COMPOSE_MODES = new Set(["mask", "overlay", "split"]);
const AI_MASK_BACKGROUNDS = new Set(["transparent", "black"]);

function loadAiPrefs() {
  const outputFormat = localStorage.getItem(STORAGE_KEYS.aiOutputFormat);
  const displayMode = localStorage.getItem(STORAGE_KEYS.aiDisplayMode);
  const composeMode = localStorage.getItem(STORAGE_KEYS.aiComposeMode);
  const maskOpacity = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.aiMaskOpacity));
  const autoRun = localStorage.getItem(STORAGE_KEYS.aiAutoRun);
  const maskBlackBg = localStorage.getItem(STORAGE_KEYS.aiMaskBlackBackground);

  if (AI_FORMATS.has(outputFormat)) {
    state.aiOutputFormat = outputFormat;
  }
  if (AI_DISPLAY_MODES.has(displayMode)) {
    state.aiDisplayMode = displayMode;
  }
  if (AI_COMPOSE_MODES.has(composeMode)) {
    state.aiComposeMode = composeMode;
  }
  if (!Number.isNaN(maskOpacity)) {
    state.aiMaskOpacity = clamp(maskOpacity, 0, 1);
  }
  if (autoRun !== null) {
    state.aiAutoRun = autoRun === "true";
  }
  if (AI_MASK_BACKGROUNDS.has(maskBlackBg)) {
    state.aiMaskBlackBackground = maskBlackBg === "black";
  }
}

function saveAiPrefs() {
  localStorage.setItem(STORAGE_KEYS.aiOutputFormat, state.aiOutputFormat);
  localStorage.setItem(STORAGE_KEYS.aiDisplayMode, state.aiDisplayMode);
  localStorage.setItem(STORAGE_KEYS.aiComposeMode, state.aiComposeMode);
  localStorage.setItem(STORAGE_KEYS.aiMaskOpacity, String(state.aiMaskOpacity));
  localStorage.setItem(STORAGE_KEYS.aiAutoRun, String(state.aiAutoRun));
  localStorage.setItem(
    STORAGE_KEYS.aiMaskBlackBackground,
    state.aiMaskBlackBackground ? "black" : "transparent",
  );
}

function applyLayout() {
  const topWeight = state.rowRatio;
  const bottomWeight = 1 - state.rowRatio;

  els.rowTop.style.flex = `${topWeight} 1 0%`;
  els.rowBottom.style.flex = `${bottomWeight} 1 0%`;

  const topPanels = getRowPanels(els.rowTop);
  topPanels[0].style.flex = `${state.topCol1} 1 0%`;
  topPanels[1].style.flex = `${state.topCol2} 1 0%`;
  topPanels[2].style.flex = `${state.topCol3} 1 0%`;

  const bottomPanels = getRowPanels(els.rowBottom);
  if (state.hasTrainingConfig && bottomPanels.length >= 3) {
    bottomPanels[0].style.flex = `${state.bottomCol1} 1 0%`;
    bottomPanels[1].style.flex = `${state.bottomCol2} 1 0%`;
    bottomPanels[2].style.flex = `${state.bottomCol3} 1 0%`;
  } else {
    bottomPanels[0].style.flex = `${state.colRatio} 1 0%`;
    bottomPanels[1].style.flex = `${1 - state.colRatio} 1 0%`;
    if (bottomPanels.length >= 3) {
      bottomPanels[2].style.flex = "0 0 0";
    }
  }
}

function applyFitMode() {
  const modeClass = state.fitToPanel ? "fit-mode" : "native-mode";
  for (const panel of [els.renderPanel, els.maskPanel, els.aiPanel]) {
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

function applyAiMaskOpacityUi() {
  const percent = Math.round(state.aiMaskOpacity * 100);
  els.aiMaskOpacity.value = String(percent);
  els.aiMaskOpacityLabel.textContent = `${percent}%`;
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function showErrorDialog(title, message) {
  els.errorDialogTitle.textContent = title;
  els.errorDialogMessage.textContent = message;
  els.errorDialogBackdrop.hidden = false;
}

function hideErrorDialog() {
  els.errorDialogBackdrop.hidden = true;
}

function reportError(title, message, error) {
  console.error(title, error ?? message);
  showErrorDialog(title, message);
  setStatus(`${title}: ${message.split("\n")[0]}`);
}

async function readErrorResponse(response) {
  const text = await response.text();
  try {
    const payload = JSON.parse(text);
    if (typeof payload.error === "string" && payload.error) {
      return payload.error;
    }
  } catch {
    // Not JSON — fall through to HTML/plain parsing.
  }

  const htmlMatch = text.match(/<p>([^<]+)<\/p>/i);
  if (htmlMatch?.[1]) {
    return htmlMatch[1].trim();
  }

  const trimmed = text.trim();
  if (trimmed) {
    return trimmed.slice(0, 500);
  }
  return `Request failed: ${response.status}`;
}

function updateTrainingConfigUi(meta) {
  const tc = meta.training_config ?? {};
  state.hasTrainingConfig = true;
  els.trainingConfigPanel.hidden = false;
  els.splitterColBottom2.hidden = false;
  state.trainingConfigYaml = tc.yaml ?? "training / inference config data not found";
  els.trainingConfigYaml.textContent = state.trainingConfigYaml;
  applyLayout();
}

function populateDatasetSelect(datasets, selected) {
  els.datasetSelect.replaceChildren();
  for (const item of datasets) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = `${item.name} (${item.count.toLocaleString()})`;
    option.selected = item.name === selected;
    els.datasetSelect.append(option);
  }
}

function updateDatasetUi(meta) {
  state.total = meta.count;
  state.selectedDataset = meta.selected ?? null;
  populateDatasetSelect(meta.datasets ?? [], meta.selected);
  localStorage.setItem(STORAGE_KEYS.datasetName, meta.selected ?? "");
  els.sampleIndex.max = Math.max(0, state.total - 1);
  els.sampleCount.textContent = `/ ${state.total.toLocaleString()}`;
  setModelUi(meta.model);
}

async function applyDatasetMeta(meta, { autoRunAi = false } = {}) {
  updateDatasetUi(meta);
  updateTrainingConfigUi(meta);

  const config = await fetchJson("/api/config");
  state.configYaml = config.yaml;
  els.configYaml.textContent = state.configYaml;

  state.index = 0;
  state.imageCache = null;
  state.mediaGeneration += 1;
  clearAiPrediction();
  resetAllPanelZoom();

  if (state.total === 0) {
    els.annotationJson.textContent = "";
    els.maskImage.removeAttribute("src");
    els.renderCanvas.width = 0;
    els.renderCanvas.height = 0;
    els.clickLabel.textContent = "";
    els.objectLabel.textContent = "";
    els.sampleId.value = "";
    els.sampleIndex.value = "0";
    renderGtPanelView();
    setStatus("Dataset is empty");
    return;
  }

  await fetchAndRenderSample(0);
  if (autoRunAi && state.aiAutoRun && state.modelLoaded) {
    await runAiPrediction({ manageLoading: false });
  }
  setStatus(`Loaded ${meta.selected}`);
}

async function selectDataset(name) {
  if (!name || state.loading || name === state.selectedDataset) {
    return;
  }

  state.loading = true;
  els.datasetSelect.disabled = true;
  els.btnReloadDataset.disabled = true;
  els.btnReloadModel.disabled = true;
  updateNavButtons();
  setStatus("Loading dataset…");

  try {
    const response = await fetch("/api/dataset/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!response.ok) {
      throw new Error(await readErrorResponse(response));
    }

    await applyDatasetMeta(await response.json(), { autoRunAi: true });
  } catch (error) {
    reportError("Dataset switch failed", error.message, error);
  } finally {
    state.loading = false;
    els.datasetSelect.disabled = false;
    els.btnReloadDataset.disabled = false;
    els.btnReloadModel.disabled = false;
    updateNavButtons();
  }
}

async function reloadDataset() {
  const name = els.datasetSelect.value || state.selectedDataset;
  if (!name || state.loading) {
    return;
  }

  state.loading = true;
  els.datasetSelect.disabled = true;
  els.btnReloadDataset.disabled = true;
  els.btnReloadModel.disabled = true;
  updateNavButtons();
  setStatus("Reloading dataset…");

  try {
    const response = await fetch("/api/dataset/reload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!response.ok) {
      throw new Error(await readErrorResponse(response));
    }

    await applyDatasetMeta(await response.json(), { autoRunAi: true });
    setStatus(`Reloaded ${name}`);
  } catch (error) {
    reportError("Dataset reload failed", error.message, error);
  } finally {
    state.loading = false;
    els.datasetSelect.disabled = false;
    els.btnReloadDataset.disabled = false;
    els.btnReloadModel.disabled = false;
    updateNavButtons();
  }
}

async function reloadModel() {
  if (state.loading) {
    return;
  }

  state.loading = true;
  els.datasetSelect.disabled = true;
  els.btnReloadDataset.disabled = true;
  els.btnReloadModel.disabled = true;
  updateNavButtons();
  setStatus("Reloading model…");

  try {
    const response = await fetch("/api/model/reload", { method: "POST" });
    if (!response.ok) {
      throw new Error(await readErrorResponse(response));
    }

    const meta = await response.json();
    updateTrainingConfigUi(meta);
    setModelUi(meta.model);
    clearAiPrediction();
    if (state.aiAutoRun && state.modelLoaded && state.total > 0) {
      await runAiPrediction({ manageLoading: false });
    }
    const label = meta.model?.loaded
      ? meta.model.checkpoint.split(/[/\\]/).pop()
      : "no model";
    setStatus(`Reloaded model: ${label}`);
  } catch (error) {
    reportError("Model reload failed", error.message, error);
  } finally {
    state.loading = false;
    els.datasetSelect.disabled = false;
    els.btnReloadDataset.disabled = false;
    els.btnReloadModel.disabled = false;
    updateNavButtons();
  }
}

function updateNavButtons() {
  const atStart = state.index <= 0;
  const atEnd = state.index >= state.total - 1;
  els.btnFirst.disabled = atStart || state.loading;
  els.btnPrev.disabled = atStart || state.loading;
  els.btnNext.disabled = atEnd || state.loading;
  els.btnLast.disabled = atEnd || state.loading;
  els.btnRunAi.disabled = state.loading || state.total === 0;
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(3);
}

function updateAiMetricsUi() {
  if (!state.aiMetrics) {
    els.aiMetricsBar.hidden = true;
    els.aiF1Label.textContent = "F1 —";
    return;
  }

  const { softF1, binF1 } = state.aiMetrics;
  els.aiF1Label.innerHTML =
    `soft F1 <strong>${formatMetric(softF1)}</strong> · bin F1 ${formatMetric(binF1)}`;
  els.aiMetricsBar.hidden = false;
}

function imageDataUsesAlphaChannel(pixels) {
  for (let i = 3; i < pixels.length; i += 4) {
    if (pixels[i] !== 255) {
      return true;
    }
  }
  return false;
}

function maskWeightFromPixel(pixels, pixelIndex, useAlphaChannel) {
  const offset = pixelIndex * 4;
  const value = useAlphaChannel ? pixels[offset + 3] : pixels[offset];
  return value / 255;
}

function readAlphaPlane(img) {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const useAlphaChannel = imageDataUsesAlphaChannel(pixels);
  const plane = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < plane.length; i += 1) {
    plane[i] = useAlphaChannel ? pixels[i * 4 + 3] : pixels[i * 4];
  }
  return { plane, width: canvas.width, height: canvas.height };
}

function readGrayscaleFromImage(img) {
  return readAlphaPlane(img).plane;
}

function setCanvasPixel(data, offset, red, green, blue, alpha = 255) {
  data[offset] = red;
  data[offset + 1] = green;
  data[offset + 2] = blue;
  data[offset + 3] = alpha;
}

function blitImageData(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

function composeRgbOverlay(outCanvas, rgbImage, overlayCanvas) {
  const width = rgbImage.naturalWidth;
  const height = rgbImage.naturalHeight;
  outCanvas.width = width;
  outCanvas.height = height;
  const ctx = outCanvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(rgbImage, 0, 0);
  ctx.drawImage(overlayCanvas, 0, 0);
}

const AI_COMPARE_COLORS = {
  tp: [56, 203, 92],
  fp: [235, 64, 64],
  fn: [255, 255, 255],
};

function thresholdMaskCanvas(sourceImage, threshold) {
  const canvas = document.createElement("canvas");
  canvas.width = sourceImage.naturalWidth || sourceImage.width;
  canvas.height = sourceImage.naturalHeight || sourceImage.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(sourceImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const useAlphaChannel = imageDataUsesAlphaChannel(imageData.data);
  const cutoff = Math.round(threshold * 255);
  for (let i = 0; i < imageData.data.length; i += 4) {
    const value = useAlphaChannel ? imageData.data[i + 3] : imageData.data[i];
    const on = value > cutoff ? 255 : 0;
    imageData.data[i] = on;
    imageData.data[i + 1] = on;
    imageData.data[i + 2] = on;
    imageData.data[i + 3] = on ? 255 : 0;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function getAiMaskSource() {
  if (!state.aiAlphaImage) {
    return null;
  }
  if (state.aiOutputFormat === "alpha") {
    return state.aiAlphaImage;
  }
  return thresholdMaskCanvas(state.aiAlphaImage, state.aiMaskThreshold);
}

/** Mask-only preview: clear = RGBA/transparent bg, black = white mask on #000. */
function renderAiMaskPreview(canvas, maskSource, { blackBackground = false } = {}) {
  const width = maskSource.naturalWidth || maskSource.width;
  const height = maskSource.naturalHeight || maskSource.height;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");

  if (!blackBackground) {
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(maskSource, 0, 0);
    return;
  }

  const sample = document.createElement("canvas");
  sample.width = width;
  sample.height = height;
  const sampleCtx = sample.getContext("2d");
  sampleCtx.drawImage(maskSource, 0, 0);
  const pixels = sampleCtx.getImageData(0, 0, width, height);
  const useAlphaChannel = imageDataUsesAlphaChannel(pixels.data);
  const out = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i += 1) {
    const offset = i * 4;
    const value = useAlphaChannel ? pixels.data[offset + 3] : pixels.data[offset];
    setCanvasPixel(out.data, offset, value, value, value, 255);
  }
  ctx.putImageData(out, 0, 0);
}

function classifyComparePixel(pred, gt) {
  const binary = state.aiOutputFormat === "binary";
  const cutoff = Math.round(state.aiMaskThreshold * 255);
  if (binary) {
    const p = pred > cutoff;
    const g = gt > cutoff;
    if (p && g) {
      return { kind: "tp" };
    }
    if (!p && !g) {
      return { kind: "tn" };
    }
    if (!p && g) {
      return { kind: "fn" };
    }
    return { kind: "fp" };
  }

  const predA = pred / 255;
  const gtA = gt / 255;
  const overlap = Math.min(predA, gtA);
  const fnAmount = Math.max(0, gtA - predA);
  const fpAmount = Math.max(0, predA - gtA);
  const strength = Math.max(overlap, fnAmount, fpAmount);
  if (strength < 0.01) {
    return { kind: "tn", strength: 0, overlap, fnAmount, fpAmount };
  }

  let kind = "tp";
  if (fpAmount >= overlap && fpAmount >= fnAmount) {
    kind = "fp";
  } else if (fnAmount >= overlap && fnAmount >= fpAmount) {
    kind = "fn";
  }
  return { kind, strength, overlap, fnAmount, fpAmount };
}

function fillCompareBlackBackground(data) {
  for (let offset = 0; offset < data.length; offset += 4) {
    setCanvasPixel(data, offset, 0, 0, 0, 255);
  }
}

/**
 * Write one compare pixel. Three output conventions:
 *   binary               — opaque hue at `opacity` (no soft strength).
 *   black bg (alpha)     — hue premultiplied by strength, alpha=255 (canvas can't
 *                          blend within putImageData, so composite onto black here).
 *   transparent/overlay  — hue with alpha=strength; overlay blends toward white for FN.
 */
function writeComparePixel(data, offset, pixel, { blackBackground, opacity, overlayStyle = false }) {
  if (pixel.kind === "tn") {
    return;
  }

  const [hueR, hueG, hueB] = AI_COMPARE_COLORS[pixel.kind];

  if (state.aiOutputFormat === "binary") {
    const alpha = Math.round(255 * opacity);
    if (alpha > 0) {
      setCanvasPixel(data, offset, hueR, hueG, hueB, alpha);
    }
    return;
  }

  if (blackBackground && !overlayStyle) {
    const blend = (pixel.strength ?? 0) * opacity;
    if (blend <= 0) {
      return;
    }
    setCanvasPixel(
      data,
      offset,
      Math.round(hueR * blend),
      Math.round(hueG * blend),
      Math.round(hueB * blend),
      255,
    );
    return;
  }

  const { strength, overlap, fnAmount, fpAmount } = pixel;
  const alpha = Math.round(strength * 255 * opacity);
  if (alpha <= 0) {
    return;
  }

  if (overlayStyle) {
    const [, tpG, tpB] = AI_COMPARE_COLORS.tp;
    const [fpR] = AI_COMPARE_COLORS.fp;
    const red = Math.min(255, Math.round(fpAmount * fpR + fnAmount * 255));
    const green = Math.min(255, Math.round(overlap * tpG + fnAmount * 255));
    const blue = Math.min(255, Math.round(overlap * tpB + fnAmount * 255));
    setCanvasPixel(data, offset, red, green, blue, alpha);
    return;
  }

  setCanvasPixel(data, offset, hueR, hueG, hueB, alpha);
}

function buildCompareImageData({ blackBackground, opacity, overlayStyle = false }) {
  const pred = readAlphaPlane(state.aiAlphaImage);
  const gt = readAlphaPlane(state.imageCache.maskImage);
  const imageData = new ImageData(pred.width, pred.height);
  const data = imageData.data;

  if (blackBackground && !overlayStyle) {
    fillCompareBlackBackground(data);
  }

  for (let i = 0; i < pred.plane.length; i += 1) {
    writeComparePixel(data, i * 4, classifyComparePixel(pred.plane[i], gt.plane[i]), {
      blackBackground,
      opacity,
      overlayStyle,
    });
  }

  return imageData;
}

function renderAiComposedView(canvas, { drawMask, drawOverlay }) {
  const rgbImage = state.imageCache?.rgbImage;
  const { aiComposeMode, aiMaskBlackBackground, aiMaskOpacity } = state;

  if (aiComposeMode === "overlay") {
    if (!rgbImage) {
      return;
    }
    const overlayCanvas = document.createElement("canvas");
    drawOverlay(overlayCanvas, { opacity: aiMaskOpacity });
    composeRgbOverlay(canvas, rgbImage, overlayCanvas);
    return;
  }

  if (aiComposeMode === "split") {
    if (!rgbImage) {
      return;
    }
    const maskPane = document.createElement("canvas");
    drawMask(maskPane, { blackBackground: aiMaskBlackBackground, opacity: 1 });
    const overlayPane = document.createElement("canvas");
    drawOverlay(overlayPane, { opacity: aiMaskOpacity });
    const rgbOverlay = document.createElement("canvas");
    composeRgbOverlay(rgbOverlay, rgbImage, overlayPane);
    renderSideBySideCanvas(canvas, maskPane, rgbOverlay);
    return;
  }

  drawMask(canvas, { blackBackground: aiMaskBlackBackground, opacity: 1 });
}

function renderAiRawView(canvas) {
  const maskSource = getAiMaskSource();
  const rgbImage = state.imageCache?.rgbImage;
  if (!maskSource) {
    return;
  }

  renderAiComposedView(canvas, {
    drawMask(target, { blackBackground }) {
      renderAiMaskPreview(target, maskSource, { blackBackground });
    },
    drawOverlay(target, { opacity }) {
      if (!rgbImage) {
        return;
      }
      renderMaskOverlayOnCanvas(target, rgbImage, maskSource, opacity);
    },
  });
}

function renderAiCompareView(canvas) {
  if (!state.imageCache?.maskImage) {
    return;
  }

  renderAiComposedView(canvas, {
    drawMask(target, { blackBackground, opacity }) {
      blitImageData(target, buildCompareImageData({
        blackBackground,
        opacity,
        overlayStyle: false,
      }));
    },
    drawOverlay(target, { opacity }) {
      blitImageData(target, buildCompareImageData({
        blackBackground: false,
        opacity,
        overlayStyle: true,
      }));
    },
  });
}

function renderAiPanelView() {
  if (!state.aiAlphaImage) {
    setAiContentVisibility({ showImage: false, showCanvas: false });
    updateAiControlUi();
    return;
  }

  updateAiControlUi();

  if (state.aiDisplayMode === "compare") {
    renderAiCompareView(els.aiPredictionCanvas);
  } else {
    renderAiRawView(els.aiPredictionCanvas);
  }

  setAiContentVisibility({ showImage: false, showCanvas: true });
}

function syncSegmentedButtons(buttons, isActive, disabled) {
  for (const button of buttons) {
    button.classList.toggle("active", isActive(button));
    button.disabled = disabled;
  }
}

function bindAiPanelButtons(buttons, onSelect) {
  for (const button of buttons) {
    button.addEventListener("click", () => {
      if (!state.aiAlphaImage) {
        return;
      }
      onSelect(button);
      saveAiPrefs();
      renderAiPanelView();
    });
  }
}

function updateAiControlUi() {
  const hasPrediction = Boolean(state.aiAlphaImage);
  els.aiPanelControls.hidden = !hasPrediction;

  syncSegmentedButtons(
    els.aiFormatButtons,
    (button) => button.dataset.aiFormat === state.aiOutputFormat,
    !hasPrediction,
  );
  syncSegmentedButtons(
    els.aiModeButtons,
    (button) => button.dataset.aiMode === state.aiDisplayMode,
    !hasPrediction,
  );

  els.aiComposeControls.hidden = !hasPrediction;
  syncSegmentedButtons(
    els.aiComposeButtons,
    (button) => button.dataset.aiCompose === state.aiComposeMode,
    !hasPrediction,
  );

  const showAiOverlayOpacity =
    hasPrediction && (state.aiComposeMode === "overlay" || state.aiComposeMode === "split");
  els.aiOverlayOpacity.hidden = !showAiOverlayOpacity;

  const showMaskBackgroundToggle =
    hasPrediction && state.aiComposeMode !== "overlay";
  els.aiMaskBgControls.hidden = !showMaskBackgroundToggle;
  syncSegmentedButtons(
    els.aiMaskBgButtons,
    (button) =>
      button.dataset.aiMaskBg === (state.aiMaskBlackBackground ? "black" : "transparent"),
    !hasPrediction,
  );
}

function updateGtControlUi() {
  const hasSample = Boolean(state.imageCache);
  for (const button of els.gtViewButtons) {
    const active = button.dataset.gtView === state.gtViewMode;
    button.classList.toggle("active", active);
    button.disabled = !hasSample;
  }
  els.gtOverlayOpacity.hidden = !hasSample || (state.gtViewMode !== "overlay" && state.gtViewMode !== "split");
}

function renderSideBySideCanvas(canvas, leftImage, rightImage) {
  const width = leftImage.naturalWidth || leftImage.width;
  const height = leftImage.naturalHeight || leftImage.height;
  canvas.width = width * 2;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(leftImage, 0, 0, width, height);
  ctx.drawImage(rightImage, width, 0, width, height);

  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
  ctx.lineWidth = Math.max(1, Math.round(width * 0.004));
  ctx.beginPath();
  ctx.moveTo(width + 0.5, 0);
  ctx.lineTo(width + 0.5, height);
  ctx.stroke();
  ctx.restore();
}

function setGtContentVisibility(mode) {
  els.maskImage.hidden = mode !== "mask";
  els.maskOverlayCanvas.hidden = mode !== "overlay";
  els.maskSplitCanvas.hidden = mode !== "split";
}

function setAiContentVisibility({ showImage, showCanvas }) {
  els.aiPredictionImage.hidden = !showImage;
  els.aiPredictionCanvas.hidden = !showCanvas;
}

function renderMaskOverlayOnCanvas(canvas, rgbImage, maskImage, opacity) {
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
  const useAlphaChannel = imageDataUsesAlphaChannel(maskPixels.data);
  const alphaScale = opacity * 255;

  for (let i = 0; i < maskPixels.data.length; i += 4) {
    const maskValue = maskWeightFromPixel(maskPixels.data, i / 4, useAlphaChannel);
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

function renderGtSplitView() {
  const { rgbImage, maskImage } = state.imageCache;
  const overlayCanvas = document.createElement("canvas");
  renderMaskOverlayOnCanvas(overlayCanvas, rgbImage, maskImage, state.maskOpacity);
  renderSideBySideCanvas(els.maskSplitCanvas, maskImage, overlayCanvas);
}

function renderGtPanelView() {
  if (!state.imageCache) {
    setGtContentVisibility("mask");
    els.maskImage.hidden = true;
    els.maskOverlayCanvas.hidden = true;
    els.maskSplitCanvas.hidden = true;
    updateGtControlUi();
    return;
  }

  updateGtControlUi();

  if (state.gtViewMode === "overlay") {
    renderMaskOverlayOnCanvas(
      els.maskOverlayCanvas,
      state.imageCache.rgbImage,
      state.imageCache.maskImage,
      state.maskOpacity,
    );
    setGtContentVisibility("overlay");
    return;
  }

  if (state.gtViewMode === "split") {
    renderGtSplitView();
    setGtContentVisibility("split");
    return;
  }

  setGtContentVisibility("mask");
}

function clearAiPrediction() {
  if (state.aiPredictionUrl) {
    URL.revokeObjectURL(state.aiPredictionUrl);
    state.aiPredictionUrl = null;
  }
  state.aiAlphaImage = null;
  state.aiMetrics = null;
  els.aiPredictionImage.removeAttribute("src");
  setAiContentVisibility({ showImage: false, showCanvas: false });
  els.aiPlaceholder.hidden = false;
  updateAiMetricsUi();
  renderAiPanelView();
}

function formatModelMetadata(metadata, threshold) {
  if (!metadata) {
    return "No checkpoint metadata available.";
  }

  const lines = [
    `checkpoint: ${metadata.checkpoint ?? "—"}`,
    `epoch: ${metadata.epoch ?? "—"}`,
    `device: ${metadata.device ?? "—"}`,
    `mask threshold: ${typeof threshold === "number" ? threshold : "—"}`,
    `has optimizer state: ${metadata.has_optimizer ? "yes" : "no"}`,
    `has scheduler state: ${metadata.has_scheduler ? "yes" : "no"}`,
    `has scaler state: ${metadata.has_scaler ? "yes" : "no"}`,
  ];

  const trainingState = metadata.training_state;
  if (trainingState && typeof trainingState === "object") {
    lines.push("", "training_state:");
    for (const [key, value] of Object.entries(trainingState)) {
      lines.push(`  ${key}: ${value}`);
    }
  }

  return lines.join("\n");
}

function setModelUi(model) {
  state.modelLoaded = Boolean(model?.loaded);
  state.modelMetadata = model?.metadata ?? null;
  if (typeof model?.threshold === "number") {
    state.aiMaskThreshold = model.threshold;
  }
  if (state.modelLoaded) {
    const name = model.checkpoint.split(/[/\\]/).pop();
    els.aiModelLabel.textContent = `${name} · ep ${model.epoch}`;
    els.aiModelLabel.classList.remove("muted-chip");
    els.aiModelLabel.disabled = false;
  } else {
    els.aiModelLabel.textContent = "no model";
    els.aiModelLabel.classList.add("muted-chip");
    els.aiModelLabel.disabled = true;
  }
  updateNavButtons();
}

async function runAiPrediction({ manageLoading = true } = {}) {
  if (state.total === 0) {
    return;
  }
  if (manageLoading && state.loading) {
    return;
  }

  if (!state.modelLoaded) {
    reportError(
      "AI unavailable",
      "No model loaded. Pass --model or set inference.checkpoint in training_config.yaml.",
    );
    return;
  }

  if (manageLoading) {
    state.loading = true;
    updateNavButtons();
  }
  setStatus("Running AI prediction…");

  try {
    const response = await fetch(`/api/predict/index/${state.index}`);
    if (!response.ok) {
      throw new Error(await readErrorResponse(response));
    }
    const blob = await response.blob();
    if (!blob.type.startsWith("image/")) {
      throw new Error("Server returned a non-image response for prediction.");
    }

    clearAiPrediction();
    const objectUrl = URL.createObjectURL(blob);
    const loadedImage = await new Promise((resolve, reject) => {
      const onLoad = () => {
        cleanup();
        resolve(els.aiPredictionImage);
      };
      const onError = () => {
        cleanup();
        URL.revokeObjectURL(objectUrl);
        reject(new Error("Failed to decode AI prediction image."));
      };
      const cleanup = () => {
        els.aiPredictionImage.removeEventListener("load", onLoad);
        els.aiPredictionImage.removeEventListener("error", onError);
      };
      els.aiPredictionImage.addEventListener("load", onLoad);
      els.aiPredictionImage.addEventListener("error", onError);
      state.aiPredictionUrl = objectUrl;
      els.aiPredictionImage.src = objectUrl;
      if (els.aiPredictionImage.complete) {
        onLoad();
      }
    });

    const thresholdHeader = response.headers.get("X-AI-Threshold");
    const parsedThreshold = Number.parseFloat(thresholdHeader ?? "");
    if (!Number.isNaN(parsedThreshold)) {
      state.aiMaskThreshold = parsedThreshold;
    }

    state.aiMetrics = {
      softF1: Number.parseFloat(response.headers.get("X-AI-Soft-F1") ?? ""),
      binF1: Number.parseFloat(response.headers.get("X-AI-Bin-F1") ?? ""),
    };
    state.aiAlphaImage = loadedImage;
    state.aiAutoRun = true;
    saveAiPrefs();
    applyAiMaskOpacityUi();
    els.aiPlaceholder.hidden = true;
    updateAiMetricsUi();
    renderAiPanelView();
    setStatus(`AI prediction ready · sample ${state.index + 1}`);
  } catch (error) {
    reportError("AI prediction failed", error.message, error);
  } finally {
    if (manageLoading) {
      state.loading = false;
      updateNavButtons();
    }
  }
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

function mediaCacheToken(index = state.index) {
  return `${state.mediaGeneration}-${index}`;
}

function mediaUrl(path, index = state.index) {
  return `/media/${path}?t=${encodeURIComponent(mediaCacheToken(index))}`;
}

async function loadSampleImages(imagePath, maskPath) {
  const cacheKey = `${state.selectedDataset}:${mediaCacheToken()}:${imagePath}:${maskPath}`;
  if (state.imageCache?.key === cacheKey) {
    return state.imageCache;
  }

  const [rgbImage, maskImage] = await Promise.all([
    loadImage(mediaUrl(imagePath)),
    loadImage(mediaUrl(maskPath)),
  ]);

  state.imageCache = {
    key: cacheKey,
    rgbImage,
    maskImage,
  };
  return state.imageCache;
}

function refreshGtPanel() {
  if (!state.imageCache) {
    return;
  }
  renderGtPanelView();
}

function refreshAiPanel() {
  if (!state.aiAlphaImage) {
    return;
  }
  renderAiPanelView();
}

async function fetchAndRenderSample(index) {
  const clamped = Math.max(0, Math.min(index, state.total - 1));
  const payload = await fetchJson(`/api/sample/index/${clamped}`);
  const record = payload.record;
  state.index = payload.index;
  state.imageCache = null;
  clearAiPrediction();
  resetAllPanelZoom();

  els.sampleIndex.value = String(state.index);
  els.sampleId.value = record.id;
  els.sampleCount.textContent = `/ ${state.total.toLocaleString()}`;

  const [x, y] = record.point;
  els.clickLabel.textContent = `point [${x}, ${y}]`;
  els.objectLabel.textContent = `object ${record.object_id} / ${record.num_objects}`;

  const images = await loadSampleImages(record.image, record.mask);
  renderImageWithCross(images.rgbImage, record.point);
  els.maskImage.src = mediaUrl(record.mask);
  renderGtPanelView();
  els.annotationJson.textContent = JSON.stringify(record, null, 2);

  return record;
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
    const record = await fetchAndRenderSample(clamped);
    if (state.aiAutoRun && state.modelLoaded) {
      await runAiPrediction({ manageLoading: false });
    } else {
      setStatus(`Sample ${state.index + 1} · id ${record.id}`);
    }
  } catch (error) {
    reportError("Sample load failed", error.message, error);
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
  bindTopColumnSplitter(els.splitterColAi, 1);
  bindBottomPrimarySplitter(els.splitterColBottom, els.rowBottom);
  bindBottomColumnSplitter(els.splitterColBottom2, 1);
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
      state.topCol1 = clamp(ratio, LAYOUT_LIMITS.min, 1 - 2 * LAYOUT_LIMITS.min);
    } else {
      const combined = clamp(
        ratio,
        state.topCol1 + LAYOUT_LIMITS.min,
        1 - LAYOUT_LIMITS.min,
      );
      state.topCol2 = combined - state.topCol1;
    }

    normalizeTopColumns();
    applyLayout();
    saveLayoutPrefs();
  });
}

function bindBottomPrimarySplitter(splitter, row) {
  startDrag(splitter, (event) => {
    const rect = row.getBoundingClientRect();
    const available = rect.width - getVerticalSplitterWidth(row);
    if (available <= 0) {
      return;
    }

    const x = (event.clientX ?? event.touches?.[0]?.clientX ?? 0) - rect.left;
    const ratio = clamp(x / available, LAYOUT_LIMITS.min, 1 - LAYOUT_LIMITS.min);

    if (state.hasTrainingConfig) {
      state.bottomCol1 = clamp(ratio, LAYOUT_LIMITS.min, 1 - 2 * LAYOUT_LIMITS.min);
      normalizeBottomColumns();
    } else {
      state.colRatio = ratio;
    }

    applyLayout();
    saveLayoutPrefs();
  });
}

function bindBottomColumnSplitter(splitter, splitterIndex) {
  startDrag(splitter, (event) => {
    if (!state.hasTrainingConfig) {
      return;
    }

    const row = els.rowBottom;
    const rect = row.getBoundingClientRect();
    const available = rect.width - getVerticalSplitterWidth(row);
    if (available <= 0) {
      return;
    }

    const x = (event.clientX ?? event.touches?.[0]?.clientX ?? 0) - rect.left;
    const ratio = clamp(x / available, LAYOUT_LIMITS.min, 1 - LAYOUT_LIMITS.min);

    if (splitterIndex === 0) {
      state.bottomCol1 = clamp(ratio, LAYOUT_LIMITS.min, 1 - 2 * LAYOUT_LIMITS.min);
    } else {
      const combined = clamp(
        ratio,
        state.bottomCol1 + LAYOUT_LIMITS.min,
        1 - LAYOUT_LIMITS.min,
      );
      state.bottomCol2 = combined - state.bottomCol1;
    }

    normalizeBottomColumns();
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
  loadAiPrefs();
  applyLayout();
  applyFitMode();
  applyMaskOpacityUi();
  applyAiMaskOpacityUi();
  updateAiControlUi();
  setupPanelZoomControls();
  setupSplitters();

  try {
    const meta = await fetchJson("/api/meta");
    clearAiPrediction();
    await applyDatasetMeta(meta);
  } catch (error) {
    reportError("Initialization failed", error.message, error);
  }
}

els.btnFirst.addEventListener("click", () => showSample(0));
els.btnReloadDataset.addEventListener("click", () => reloadDataset());
els.btnReloadModel.addEventListener("click", () => reloadModel());
els.datasetSelect.addEventListener("change", () => {
  selectDataset(els.datasetSelect.value);
});
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
  resetAllPanelZoom();
  saveLayoutPrefs();
});

els.maskOpacity.addEventListener("input", () => {
  state.maskOpacity = Number.parseInt(els.maskOpacity.value, 10) / 100;
  applyMaskOpacityUi();
  if (state.gtViewMode === "overlay" || state.gtViewMode === "split") {
    refreshGtPanel();
  }
  saveLayoutPrefs();
});

els.aiMaskOpacity.addEventListener("input", () => {
  state.aiMaskOpacity = Number.parseInt(els.aiMaskOpacity.value, 10) / 100;
  applyAiMaskOpacityUi();
  saveAiPrefs();
  if (
    state.aiAlphaImage &&
    (state.aiComposeMode === "overlay" || state.aiComposeMode === "split")
  ) {
    refreshAiPanel();
  }
});

for (const button of els.gtViewButtons) {
  button.addEventListener("click", () => {
    if (!state.imageCache) {
      return;
    }
    state.gtViewMode = button.dataset.gtView;
    renderGtPanelView();
  });
}

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
  } else if (event.key === "p" || event.key === "P") {
    event.preventDefault();
    runAiPrediction();
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
  copyText(state.configYaml, "dataset creation config");
});

els.btnCopyTrainingConfig.addEventListener("click", () => {
  copyText(state.trainingConfigYaml, "training / inference config");
});

els.aiModelLabel.addEventListener("click", () => {
  if (!state.modelLoaded) {
    return;
  }
  showErrorDialog(
    "Model checkpoint metadata",
    formatModelMetadata(state.modelMetadata, state.aiMaskThreshold),
  );
});

els.btnRunAi.addEventListener("click", () => {
  runAiPrediction();
});

bindAiPanelButtons(els.aiFormatButtons, (button) => {
  state.aiOutputFormat = button.dataset.aiFormat;
});
bindAiPanelButtons(els.aiModeButtons, (button) => {
  state.aiDisplayMode = button.dataset.aiMode;
});
bindAiPanelButtons(els.aiComposeButtons, (button) => {
  state.aiComposeMode = button.dataset.aiCompose;
});
bindAiPanelButtons(els.aiMaskBgButtons, (button) => {
  state.aiMaskBlackBackground = button.dataset.aiMaskBg === "black";
});

for (const btn of [els.errorDialogClose, els.errorDialogOk]) {
  btn.addEventListener("click", hideErrorDialog);
}

els.errorDialogBackdrop.addEventListener("click", (event) => {
  if (event.target === els.errorDialogBackdrop) {
    hideErrorDialog();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.errorDialogBackdrop.hidden) {
    hideErrorDialog();
  }
});

window.addEventListener("resize", applyLayout);

init();
