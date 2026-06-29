import { STORAGE_KEYS, LAYOUT_LIMITS, state, els, datasetAiCtx, interactiveAiCtx } from "./shared.js";
import { fetchJson, clamp } from "./utils.js";
import {
  applyLayout,
  getRowPanels,
  getVerticalSplitterWidth,
  saveLayoutPrefs,
  normalizeTopColumns,
  normalizeBottomColumns,
} from "./layout.js";
import { setStatus, reportError, updateBottomPanelPaths, updateTrainingConfigUi } from "./ui.js";
import { resetAllPanelZoom } from "./zoom.js";
import {
  clearAiPrediction,
  runAiPrediction,
  runInteractiveAiPrediction,
} from "./ai-panel.js";
import { setModelUi, updateNavButtons } from "./nav.js";
import {
  renderGtPanelView,
  renderImageWithCross,
  loadSampleImages,
  mediaUrl,
} from "./render.js";

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
  updateBottomPanelPaths(meta);
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
  clearAiPrediction(datasetAiCtx);
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
    updateBottomPanelPaths(meta);
    setModelUi(meta.model);
    clearAiPrediction(datasetAiCtx);
    clearAiPrediction(interactiveAiCtx);
    if (state.aiAutoRun && state.modelLoaded && state.total > 0) {
      await runAiPrediction({ manageLoading: false });
    }
    if (state.interactive.point && state.interactive.rgbImage && state.modelLoaded) {
      await runInteractiveAiPrediction(state.interactive.point);
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

async function fetchAndRenderSample(index) {
  const clamped = Math.max(0, Math.min(index, state.total - 1));
  const payload = await fetchJson(`/api/sample/index/${clamped}`);
  const record = payload.record;
  state.index = payload.index;
  state.imageCache = null;
  state.aiPredictionGeneration += 1;
  clearAiPrediction(datasetAiCtx);
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
    setStatus(`Sample ${state.index + 1} · id ${record.id}`);
    if (state.aiAutoRun && state.modelLoaded) {
      runAiPrediction({ manageLoading: false });
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
export {
  populateDatasetSelect,
  updateDatasetUi,
  applyDatasetMeta,
  selectDataset,
  reloadDataset,
  reloadModel,
  fetchAndRenderSample,
  showSample,
  jumpToId,
  setupSplitters,
  startDrag,
};
