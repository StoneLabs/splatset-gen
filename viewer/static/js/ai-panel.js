import { state, els, datasetAiCtx, interactiveAiCtx } from "./shared.js";
import { AI_COMPARE_COLORS } from "./constants.js";
import { formatMetric, readErrorResponse, syncSegmentedButtons } from "./utils.js";
import { setStatus, reportError } from "./ui.js";
import { saveAiPrefs, applyAiMaskOpacityUi } from "./layout.js";
import { updateNavButtons } from "./nav.js";
import {
  imageDataUsesAlphaChannel,
  readAlphaPlane,
  setCanvasPixel,
  blitImageData,
  composeRgbOverlay,
  renderSideBySideCanvas,
  renderMaskOverlayOnCanvas,
} from "./canvas.js";
import { renderInteractiveImageWithCross } from "./render.js";

function updateAiMetricsUi(ctx = datasetAiCtx) {
  const refs = ctx.refs;
  const metrics = ctx.getMetrics();
  if (!metrics) {
    refs.metricsBar().hidden = true;
    refs.f1Label().textContent = "F1 —";
    return;
  }

  const { softF1, binF1 } = metrics;
  refs.f1Label().innerHTML =
    `soft F1 <strong>${formatMetric(softF1)}</strong> · bin F1 ${formatMetric(binF1)}`;
  refs.metricsBar().hidden = false;
}

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

function getAiMaskSource(ctx) {
  const alphaImage = ctx.getAlphaImage();
  if (!alphaImage) {
    return null;
  }
  if (state.aiOutputFormat === "alpha") {
    return alphaImage;
  }
  return thresholdMaskCanvas(alphaImage, state.aiMaskThreshold);
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

function buildCompareImageData(ctx, { blackBackground, opacity, overlayStyle = false }) {
  const pred = readAlphaPlane(ctx.getAlphaImage());
  const gtImage = ctx.getMaskImage();
  if (!gtImage) {
    return null;
  }
  const gt = readAlphaPlane(gtImage);
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

function renderAiComposedView(canvas, ctx, { drawMask, drawOverlay }) {
  const rgbImage = ctx.getRgbImage();
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

function renderAiRawView(canvas, ctx) {
  const maskSource = getAiMaskSource(ctx);
  const rgbImage = ctx.getRgbImage();
  if (!maskSource) {
    return;
  }

  renderAiComposedView(canvas, ctx, {
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

function renderAiCompareView(canvas, ctx) {
  if (!ctx.getMaskImage()) {
    return;
  }

  renderAiComposedView(canvas, ctx, {
    drawMask(target, { blackBackground, opacity }) {
      const imageData = buildCompareImageData(ctx, {
        blackBackground,
        opacity,
        overlayStyle: false,
      });
      if (imageData) {
        blitImageData(target, imageData);
      }
    },
    drawOverlay(target, { opacity }) {
      const imageData = buildCompareImageData(ctx, {
        blackBackground: false,
        opacity,
        overlayStyle: true,
      });
      if (imageData) {
        blitImageData(target, imageData);
      }
    },
  });
}

function renderAiPanelView(ctx = datasetAiCtx) {
  const refs = ctx.refs;
  if (!ctx.getAlphaImage()) {
    setAiContentVisibility(ctx, { showImage: false, showCanvas: false });
    updateAiControlUi(ctx);
    return;
  }

  updateAiControlUi(ctx);

  if (state.aiDisplayMode === "compare" && ctx.hasCompare) {
    renderAiCompareView(refs.predictionCanvas(), ctx);
  } else {
    renderAiRawView(refs.predictionCanvas(), ctx);
  }

  setAiContentVisibility(ctx, { showImage: false, showCanvas: true });
}

function bindAiPanelButtons(ctx, buttons, onSelect) {
  for (const button of buttons) {
    button.addEventListener("click", () => {
      if (!ctx.getAlphaImage()) {
        return;
      }
      onSelect(button);
      saveAiPrefs();
      renderAiPanelView(ctx);
    });
  }
}

function updateAiControlUi(ctx = datasetAiCtx) {
  const refs = ctx.refs;
  const hasPrediction = Boolean(ctx.getAlphaImage());
  refs.panelControls().hidden = !hasPrediction;

  syncSegmentedButtons(
    refs.formatButtons(),
    (button) =>
      button.dataset.aiFormat === state.aiOutputFormat ||
      button.dataset.interactiveAiFormat === state.aiOutputFormat,
    !hasPrediction,
  );
  syncSegmentedButtons(
    refs.modeButtons(),
    (button) =>
      button.dataset.aiMode === state.aiDisplayMode ||
      button.dataset.interactiveAiMode === state.aiDisplayMode,
    !hasPrediction || !ctx.hasCompare,
  );

  refs.composeControls().hidden = !hasPrediction;
  syncSegmentedButtons(
    refs.composeButtons(),
    (button) =>
      button.dataset.aiCompose === state.aiComposeMode ||
      button.dataset.interactiveAiCompose === state.aiComposeMode,
    !hasPrediction,
  );

  const showAiOverlayOpacity =
    hasPrediction && (state.aiComposeMode === "overlay" || state.aiComposeMode === "split");
  refs.overlayOpacity().hidden = !showAiOverlayOpacity;

  const showMaskBackgroundToggle = hasPrediction && state.aiComposeMode !== "overlay";
  refs.maskBgControls().hidden = !showMaskBackgroundToggle;
  syncSegmentedButtons(
    refs.maskBgButtons(),
    (button) =>
      button.dataset.aiMaskBg === (state.aiMaskBlackBackground ? "black" : "transparent") ||
      button.dataset.interactiveAiMaskBg === (state.aiMaskBlackBackground ? "black" : "transparent"),
    !hasPrediction,
  );
}

function setAiContentVisibility(ctx, { showImage, showCanvas }) {
  const refs = ctx.refs;
  refs.predictionImage().hidden = !showImage;
  refs.predictionCanvas().hidden = !showCanvas;
}

function clearAiPrediction(ctx = datasetAiCtx) {
  const refs = ctx.refs;
  const predictionUrl = ctx.getPredictionUrl();
  if (predictionUrl) {
    URL.revokeObjectURL(predictionUrl);
    ctx.setPredictionUrl(null);
  }
  ctx.setAlphaImage(null);
  ctx.setMetrics(null);
  refs.predictionImage().removeAttribute("src");
  setAiContentVisibility(ctx, { showImage: false, showCanvas: false });
  if (ctx.id === "interactive") {
    refs.placeholder().hidden = !state.interactive.rgbImage;
  } else {
    refs.placeholder().hidden = false;
  }
  updateAiMetricsUi(ctx);
  renderAiPanelView(ctx);
}

function isImageBlob(blob) {
  return !blob.type || blob.type.startsWith("image/");
}

async function applyAiPredictionFromBlob(ctx, response, blob) {
  const refs = ctx.refs;
  if (!isImageBlob(blob)) {
    throw new Error("Server returned a non-image response for prediction.");
  }

  const predictionUrl = ctx.getPredictionUrl();
  if (predictionUrl) {
    URL.revokeObjectURL(predictionUrl);
    ctx.setPredictionUrl(null);
  }
  ctx.setAlphaImage(null);
  ctx.setMetrics(null);
  refs.predictionImage().removeAttribute("src");

  const objectUrl = URL.createObjectURL(blob);
  const predictionImage = refs.predictionImage();
  const loadedImage = await new Promise((resolve, reject) => {
    const onLoad = () => {
      cleanup();
      resolve(predictionImage);
    };
    const onError = () => {
      cleanup();
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Failed to decode AI prediction image."));
    };
    const cleanup = () => {
      predictionImage.removeEventListener("load", onLoad);
      predictionImage.removeEventListener("error", onError);
    };
    predictionImage.addEventListener("load", onLoad);
    predictionImage.addEventListener("error", onError);
    ctx.setPredictionUrl(objectUrl);
    predictionImage.src = objectUrl;
    if (predictionImage.complete) {
      onLoad();
    }
  });

  const thresholdHeader = response.headers.get("X-AI-Threshold");
  const parsedThreshold = Number.parseFloat(thresholdHeader ?? "");
  if (!Number.isNaN(parsedThreshold)) {
    state.aiMaskThreshold = parsedThreshold;
  }

  const softF1 = Number.parseFloat(response.headers.get("X-AI-Soft-F1") ?? "");
  const binF1 = Number.parseFloat(response.headers.get("X-AI-Bin-F1") ?? "");
  if (!Number.isNaN(softF1) || !Number.isNaN(binF1)) {
    ctx.setMetrics({ softF1, binF1 });
  } else {
    ctx.setMetrics(null);
  }

  ctx.setAlphaImage(loadedImage);
  if (ctx.id === "dataset") {
    state.aiAutoRun = true;
    saveAiPrefs();
  }
  applyAiMaskOpacityUi();
  refs.placeholder().hidden = true;
  updateAiMetricsUi(ctx);
  renderAiPanelView(ctx);
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
      "No model loaded. Pass --model with a checkpoint (.pth).",
    );
    return;
  }

  if (manageLoading) {
    state.loading = true;
    updateNavButtons();
  }
  setStatus("Running AI prediction…");

  const sampleIndex = state.index;
  const generation = ++state.aiPredictionGeneration;

  try {
    const response = await fetch(`/api/predict/index/${sampleIndex}`);
    if (!response.ok) {
      throw new Error(await readErrorResponse(response));
    }
    if (generation !== state.aiPredictionGeneration || sampleIndex !== state.index) {
      return;
    }
    const blob = await response.blob();
    if (generation !== state.aiPredictionGeneration || sampleIndex !== state.index) {
      return;
    }
    await applyAiPredictionFromBlob(datasetAiCtx, response, blob);
    if (generation !== state.aiPredictionGeneration || sampleIndex !== state.index) {
      return;
    }
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

async function runInteractiveAiPrediction(point) {
  const { rgbImage, imageFile } = state.interactive;
  if (!rgbImage || !imageFile) {
    return;
  }
  if (!state.modelLoaded) {
    reportError(
      "AI unavailable",
      "No model loaded. Pass --model with a checkpoint (.pth).",
    );
    return;
  }

  state.interactive.point = point;
  els.interactiveClickLabel.textContent = `point [${point[0]}, ${point[1]}]`;
  renderInteractiveImageWithCross(rgbImage, point);

  const generation = ++state.interactive.predictionGeneration;
  state.interactive.loading = true;
  setStatus("Running AI prediction…");

  try {
    const formData = new FormData();
    formData.append("image", imageFile, imageFile.name || "upload.png");
    formData.append("x", String(point[0]));
    formData.append("y", String(point[1]));

    const response = await fetch("/api/predict/interactive", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readErrorResponse(response));
    }
    if (generation !== state.interactive.predictionGeneration) {
      return;
    }
    const blob = await response.blob();
    if (generation !== state.interactive.predictionGeneration) {
      return;
    }
    await applyAiPredictionFromBlob(interactiveAiCtx, response, blob);
    if (generation !== state.interactive.predictionGeneration) {
      return;
    }
    setStatus(`AI prediction ready · point [${point[0]}, ${point[1]}]`);
  } catch (error) {
    if (generation === state.interactive.predictionGeneration) {
      reportError("AI prediction failed", error.message, error);
    }
  } finally {
    if (generation === state.interactive.predictionGeneration) {
      state.interactive.loading = false;
    }
  }
}

function refreshAiPanel(ctx = datasetAiCtx) {
  if (!ctx.getAlphaImage()) {
    return;
  }
  renderAiPanelView(ctx);
}
export {
  updateAiMetricsUi,
  thresholdMaskCanvas,
  getAiMaskSource,
  renderAiMaskPreview,
  classifyComparePixel,
  fillCompareBlackBackground,
  writeComparePixel,
  buildCompareImageData,
  renderAiComposedView,
  renderAiRawView,
  renderAiCompareView,
  renderAiPanelView,
  bindAiPanelButtons,
  updateAiControlUi,
  setAiContentVisibility,
  clearAiPrediction,
  applyAiPredictionFromBlob,
  runAiPrediction,
  runInteractiveAiPrediction,
  refreshAiPanel,
};
