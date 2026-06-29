import { STORAGE_KEYS, LAYOUT_LIMITS, state, els } from "./shared.js";
import { clamp } from "./utils.js";
import {
  AI_FORMATS,
  AI_DISPLAY_MODES,
  AI_COMPOSE_MODES,
  AI_MASK_BACKGROUNDS,
} from "./constants.js";

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

  const appMode = localStorage.getItem(STORAGE_KEYS.appMode);
  if (appMode === "dataset" || appMode === "interactive") {
    state.appMode = appMode;
  }
  const interactiveCol1 = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.interactiveCol1));
  if (!Number.isNaN(interactiveCol1)) {
    state.interactiveCol1 = clamp(interactiveCol1, LAYOUT_LIMITS.min, 1 - LAYOUT_LIMITS.min);
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
  if (state.appMode === "interactive") {
    const panels = getRowPanels(els.interactiveRowTop);
    if (panels.length >= 2) {
      panels[0].style.flex = `${state.interactiveCol1} 1 0%`;
      panels[1].style.flex = `${1 - state.interactiveCol1} 1 0%`;
    }
    return;
  }

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
  for (const panel of [
    els.renderPanel,
    els.maskPanel,
    els.aiPanel,
    els.interactiveRenderPanel,
    els.interactiveAiPanel,
  ]) {
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
  for (const [slider, label] of [
    [els.aiMaskOpacity, els.aiMaskOpacityLabel],
    [els.interactiveAiMaskOpacity, els.interactiveAiMaskOpacityLabel],
  ]) {
    slider.value = String(percent);
    label.textContent = `${percent}%`;
  }
}
export {
  getRowPanels,
  getVerticalSplitterWidth,
  loadLayoutPrefs,
  normalizeBottomColumns,
  normalizeTopColumns,
  saveLayoutPrefs,
  loadAiPrefs,
  saveAiPrefs,
  applyLayout,
  applyFitMode,
  applyMaskOpacityUi,
  applyAiMaskOpacityUi,
};
