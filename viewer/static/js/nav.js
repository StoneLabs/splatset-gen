import { state, els } from "./shared.js";

function updateNavButtons() {
  const atStart = state.index <= 0;
  const atEnd = state.index >= state.total - 1;
  els.btnFirst.disabled = atStart || state.loading;
  els.btnPrev.disabled = atStart || state.loading;
  els.btnNext.disabled = atEnd || state.loading;
  els.btnLast.disabled = atEnd || state.loading;
  els.btnRunAi.disabled = state.loading || state.total === 0;
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
  const labels = [els.aiModelLabel, els.interactiveAiModelLabel];
  if (state.modelLoaded) {
    const name = model.checkpoint.split(/[/\\]/).pop();
    const text = `${name} · ep ${model.epoch}`;
    for (const label of labels) {
      label.textContent = text;
      label.classList.remove("muted-chip");
      label.disabled = false;
    }
  } else {
    for (const label of labels) {
      label.textContent = "no model";
      label.classList.add("muted-chip");
      label.disabled = true;
    }
  }
  updateNavButtons();
}

export { updateNavButtons, setModelUi, formatModelMetadata };
