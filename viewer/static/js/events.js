import { state, els, datasetAiCtx, interactiveAiCtx } from "./shared.js";
import {
  applyFitMode,
  applyMaskOpacityUi,
  applyAiMaskOpacityUi,
  saveLayoutPrefs,
  saveAiPrefs,
  applyLayout,
} from "./layout.js";
import { resetAllPanelZoom } from "./zoom.js";
import { refreshGtPanel, renderGtPanelView } from "./render.js";
import {
  bindAiPanelButtons,
  refreshAiPanel,
  runAiPrediction,
  runInteractiveAiPrediction,
} from "./ai-panel.js";
import {
  showSample,
  jumpToId,
  selectDataset,
  reloadDataset,
  reloadModel,
} from "./dataset.js";
import { switchAppMode } from "./interactive.js";
import { showErrorDialog, hideErrorDialog, copyText } from "./ui.js";
import { formatModelMetadata } from "./nav.js";

els.btnToggleMode.addEventListener("click", () => {
  switchAppMode(state.appMode === "dataset" ? "interactive" : "dataset");
});

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
  els.interactiveAiMaskOpacity.value = els.aiMaskOpacity.value;
  applyAiMaskOpacityUi();
  saveAiPrefs();
  if (
    datasetAiCtx.getAlphaImage() &&
    (state.aiComposeMode === "overlay" || state.aiComposeMode === "split")
  ) {
    refreshAiPanel(datasetAiCtx);
  }
  if (
    interactiveAiCtx.getAlphaImage() &&
    (state.aiComposeMode === "overlay" || state.aiComposeMode === "split")
  ) {
    refreshAiPanel(interactiveAiCtx);
  }
});

els.interactiveAiMaskOpacity.addEventListener("input", () => {
  state.aiMaskOpacity = Number.parseInt(els.interactiveAiMaskOpacity.value, 10) / 100;
  els.aiMaskOpacity.value = els.interactiveAiMaskOpacity.value;
  applyAiMaskOpacityUi();
  saveAiPrefs();
  if (
    datasetAiCtx.getAlphaImage() &&
    (state.aiComposeMode === "overlay" || state.aiComposeMode === "split")
  ) {
    refreshAiPanel(datasetAiCtx);
  }
  if (
    interactiveAiCtx.getAlphaImage() &&
    (state.aiComposeMode === "overlay" || state.aiComposeMode === "split")
  ) {
    refreshAiPanel(interactiveAiCtx);
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
  if (state.appMode === "interactive") {
    if (event.key === "p" || event.key === "P") {
      event.preventDefault();
      if (state.interactive.point) {
        runInteractiveAiPrediction(state.interactive.point);
      }
    }
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

els.btnCopyJson.addEventListener("click", () => {
  copyText(els.annotationJson.textContent, "annotation JSON");
});

els.btnCopyConfig.addEventListener("click", () => {
  copyText(state.configYaml, "dataset creation config");
});

els.btnCopyTrainingConfig.addEventListener("click", () => {
  copyText(state.trainingConfigYaml, "training / inference config");
});

for (const [button, label] of [
  [els.infoAnnotationPath, "annotation file"],
  [els.infoConfigPath, "dataset creation config"],
  [els.infoTrainingConfigPath, "training / inference config"],
]) {
  button.addEventListener("click", () => {
    const path = button.dataset.path;
    if (!path) {
      return;
    }
    copyText(path, `${label} path`);
  });
}

els.aiModelLabel.addEventListener("click", () => {
  if (!state.modelLoaded) {
    return;
  }
  showErrorDialog(
    "Model checkpoint metadata",
    formatModelMetadata(state.modelMetadata, state.aiMaskThreshold),
  );
});

els.interactiveAiModelLabel.addEventListener("click", () => {
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

bindAiPanelButtons(datasetAiCtx, els.aiFormatButtons, (button) => {
  state.aiOutputFormat = button.dataset.aiFormat;
});
bindAiPanelButtons(datasetAiCtx, els.aiModeButtons, (button) => {
  state.aiDisplayMode = button.dataset.aiMode;
});
bindAiPanelButtons(datasetAiCtx, els.aiComposeButtons, (button) => {
  state.aiComposeMode = button.dataset.aiCompose;
});
bindAiPanelButtons(datasetAiCtx, els.aiMaskBgButtons, (button) => {
  state.aiMaskBlackBackground = button.dataset.aiMaskBg === "black";
});

bindAiPanelButtons(interactiveAiCtx, els.interactiveAiFormatButtons, (button) => {
  state.aiOutputFormat = button.dataset.interactiveAiFormat;
});
bindAiPanelButtons(interactiveAiCtx, els.interactiveAiModeButtons, (button) => {
  state.aiDisplayMode = button.dataset.interactiveAiMode;
});
bindAiPanelButtons(interactiveAiCtx, els.interactiveAiComposeButtons, (button) => {
  state.aiComposeMode = button.dataset.interactiveAiCompose;
});
bindAiPanelButtons(interactiveAiCtx, els.interactiveAiMaskBgButtons, (button) => {
  state.aiMaskBlackBackground = button.dataset.interactiveAiMaskBg === "black";
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
