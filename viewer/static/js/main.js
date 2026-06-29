import { datasetAiCtx, interactiveAiCtx } from "./shared.js";
import { fetchJson } from "./utils.js";
import {
  loadLayoutPrefs,
  loadAiPrefs,
  applyFitMode,
  applyMaskOpacityUi,
  applyAiMaskOpacityUi,
} from "./layout.js";
import { setupPanelZoomControls } from "./zoom.js";
import { setupSplitters, applyDatasetMeta } from "./dataset.js";
import {
  setupInteractiveColumnSplitter,
  setupInteractiveImageClick,
  setupInteractiveUpload,
  applyAppModeUi,
} from "./interactive.js";
import { clearAiPrediction, updateAiControlUi } from "./ai-panel.js";
import { reportError } from "./ui.js";
import "./events.js";

async function init() {
  loadLayoutPrefs();
  loadAiPrefs();
  applyAppModeUi();
  applyFitMode();
  applyMaskOpacityUi();
  applyAiMaskOpacityUi();
  updateAiControlUi(datasetAiCtx);
  updateAiControlUi(interactiveAiCtx);
  setupPanelZoomControls();
  setupSplitters();
  setupInteractiveColumnSplitter();
  setupInteractiveImageClick();
  setupInteractiveUpload();

  try {
    const meta = await fetchJson("/api/meta");
    clearAiPrediction(datasetAiCtx);
    await applyDatasetMeta(meta);
  } catch (error) {
    reportError("Initialization failed", error.message, error);
  }
}

init();
