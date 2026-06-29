import {
  STORAGE_KEYS,
  LAYOUT_LIMITS,
  state,
  els,
  interactiveAiCtx,
  datasetAiCtx,
} from "./shared.js";
import { clamp } from "./utils.js";
import { applyLayout } from "./layout.js";
import { setStatus, reportError } from "./ui.js";
import { resetPanelZoom } from "./zoom.js";
import {
  clearAiPrediction,
  renderAiPanelView,
  updateAiControlUi,
  runInteractiveAiPrediction,
} from "./ai-panel.js";
import {
  loadImage,
  clientToCanvasPixels,
  renderInteractiveImageWithCross,
} from "./render.js";
import { getVerticalSplitterWidth } from "./layout.js";
import { startDrag } from "./dataset.js";

function clearInteractiveImage() {
  if (state.interactive.imageObjectUrl) {
    URL.revokeObjectURL(state.interactive.imageObjectUrl);
  }
  state.interactive.rgbImage = null;
  state.interactive.imageFile = null;
  state.interactive.imageObjectUrl = null;
  state.interactive.point = null;
  els.interactiveRenderCanvas.width = 0;
  els.interactiveRenderCanvas.height = 0;
  els.interactiveClickLabel.textContent = "";
  els.interactiveRenderPanel.classList.remove("has-image");
  els.interactiveUploadPlaceholder.hidden = false;
  clearAiPrediction(interactiveAiCtx);
}

async function loadInteractiveImage(file) {
  if (!file || !file.type.startsWith("image/")) {
    reportError("Invalid image", "Please choose a PNG, JPEG, or other image file.");
    return;
  }

  clearInteractiveImage();

  const objectUrl = URL.createObjectURL(file);
  try {
    const rgbImage = await loadImage(objectUrl);
    state.interactive.rgbImage = rgbImage;
    state.interactive.imageFile = file;
    state.interactive.imageObjectUrl = objectUrl;
    els.interactiveRenderPanel.classList.add("has-image");
    els.interactiveUploadPlaceholder.hidden = true;
    els.interactiveRenderCanvas.style.cursor = "crosshair";
    resetPanelZoom("interactive");
    const canvas = els.interactiveRenderCanvas;
    const ctx = canvas.getContext("2d");
    canvas.width = rgbImage.naturalWidth;
    canvas.height = rgbImage.naturalHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(rgbImage, 0, 0);
    setStatus(`Loaded ${file.name} · click to predict`);
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    reportError("Image load failed", error.message, error);
  }
}

function applyAppModeUi() {
  const interactive = state.appMode === "interactive";
  els.workspaceDataset.hidden = interactive;
  els.workspaceInteractive.hidden = !interactive;
  els.datasetControls.hidden = interactive;
  els.navControls.hidden = interactive;
  els.btnToggleMode.textContent = interactive ? "Dataset" : "Interactive";
  els.btnToggleMode.title = interactive ? "Switch to dataset browser" : "Switch to interactive mode";
  localStorage.setItem(STORAGE_KEYS.appMode, state.appMode);
  applyLayout();
  updateAiControlUi(datasetAiCtx);
  updateAiControlUi(interactiveAiCtx);
  if (datasetAiCtx.getAlphaImage()) {
    renderAiPanelView(datasetAiCtx);
  }
  if (interactiveAiCtx.getAlphaImage()) {
    renderAiPanelView(interactiveAiCtx);
  }
}

function switchAppMode(mode) {
  if (mode === state.appMode) {
    return;
  }
  state.appMode = mode;
  applyAppModeUi();
  if (mode === "interactive") {
    setStatus(state.interactive.rgbImage ? "Interactive mode" : "Upload an image to begin");
  } else if (state.total > 0) {
    setStatus(`Sample ${state.index + 1} · id ${els.sampleId.value}`);
  } else {
    setStatus("");
  }
}

function setupInteractiveColumnSplitter() {
  startDrag(els.splitterInteractiveCol, (event) => {
    const row = els.interactiveRowTop;
    const rect = row.getBoundingClientRect();
    const available = rect.width - getVerticalSplitterWidth(row);
    if (available <= 0) {
      return;
    }

    const x = (event.clientX ?? event.touches?.[0]?.clientX ?? 0) - rect.left;
    state.interactiveCol1 = clamp(x / available, LAYOUT_LIMITS.min, 1 - LAYOUT_LIMITS.min);
    applyLayout();
    localStorage.setItem(STORAGE_KEYS.interactiveCol1, String(state.interactiveCol1));
  });
}

function setupInteractiveImageClick() {
  els.interactiveRenderCanvas.addEventListener("click", (event) => {
    if (!state.interactive.rgbImage) {
      return;
    }
    const point = clientToCanvasPixels(event, els.interactiveRenderCanvas);
    if (!point) {
      return;
    }
    runInteractiveAiPrediction(point);
  });
}

function setupInteractiveUpload() {
  els.btnInteractiveUpload.addEventListener("click", () => {
    els.interactiveImageUpload.click();
  });

  els.interactiveImageUpload.addEventListener("change", () => {
    const file = els.interactiveImageUpload.files?.[0];
    els.interactiveImageUpload.value = "";
    if (file) {
      loadInteractiveImage(file);
    }
  });

  els.interactiveRenderPanel.addEventListener("dragover", (event) => {
    event.preventDefault();
  });

  els.interactiveRenderPanel.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      loadInteractiveImage(file);
    }
  });
}
export {
  clearInteractiveImage,
  loadInteractiveImage,
  applyAppModeUi,
  switchAppMode,
  setupInteractiveColumnSplitter,
  setupInteractiveImageClick,
  setupInteractiveUpload,
};
