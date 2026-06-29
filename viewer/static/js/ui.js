import { els, state } from "./shared.js";
import { applyLayout } from "./layout.js";

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

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    setStatus(`Copied ${label}`);
  } catch {
    setStatus(`Could not copy ${label}`);
  }
}

function setPathInfoButton(button, path, label) {
  if (!button) {
    return;
  }
  if (path) {
    button.disabled = false;
    button.title = path;
    button.dataset.path = path;
    button.setAttribute("aria-label", `Copy ${label} path`);
  } else {
    button.disabled = true;
    button.title = `${label} path unavailable`;
    delete button.dataset.path;
    button.setAttribute("aria-label", `${label} path unavailable`);
  }
}

function updateBottomPanelPaths(meta) {
  setPathInfoButton(els.infoAnnotationPath, meta.annotations_path, "annotation file");
  setPathInfoButton(els.infoConfigPath, meta.config_path, "dataset creation config");
  setPathInfoButton(
    els.infoTrainingConfigPath,
    meta.training_config?.path ?? null,
    "training / inference config",
  );
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
export {
  setStatus,
  showErrorDialog,
  hideErrorDialog,
  reportError,
  setPathInfoButton,
  updateBottomPanelPaths,
  updateTrainingConfigUi,
  copyText,
};
