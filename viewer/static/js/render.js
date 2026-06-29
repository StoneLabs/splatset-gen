import { state, els, panelZoom } from "./shared.js";
import { clamp } from "./utils.js";
import { renderMaskOverlayOnCanvas, renderSideBySideCanvas } from "./canvas.js";

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

async function renderImageWithCross(rgbImage, point, canvas = els.renderCanvas) {
  const ctx = canvas.getContext("2d");

  canvas.width = rgbImage.naturalWidth;
  canvas.height = rgbImage.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(rgbImage, 0, 0);

  const [x, y] = point;
  drawCross(ctx, x, y, canvas.width, canvas.height);
}

function renderInteractiveImageWithCross(rgbImage, point) {
  return renderImageWithCross(rgbImage, point, els.interactiveRenderCanvas);
}

function containRect(boxW, boxH, contentW, contentH) {
  const contentAspect = contentW / contentH;
  const boxAspect = boxW / boxH;
  let displayW;
  let displayH;
  if (contentAspect > boxAspect) {
    displayW = boxW;
    displayH = boxW / contentAspect;
  } else {
    displayH = boxH;
    displayW = boxH * contentAspect;
  }
  return {
    left: (boxW - displayW) / 2,
    top: (boxH - displayH) / 2,
    width: displayW,
    height: displayH,
  };
}

function displayedImageRectInContent(canvas, fitToPanel = state.fitToPanel) {
  const pixelW = canvas.width;
  const pixelH = canvas.height;
  if (pixelW <= 0 || pixelH <= 0) {
    return null;
  }

  const viewport = canvas.closest("[data-zoom-panel]");
  if (!viewport) {
    return null;
  }

  const contentW = viewport.clientWidth;
  const contentH = viewport.clientHeight;
  if (contentW <= 0 || contentH <= 0) {
    return null;
  }

  if (fitToPanel) {
    return containRect(contentW, contentH, pixelW, pixelH);
  }

  return {
    left: (contentW - pixelW) / 2,
    top: (contentH - pixelH) / 2,
    width: pixelW,
    height: pixelH,
  };
}

function clientToCanvasPixels(event, canvas, { fitToPanel = state.fitToPanel } = {}) {
  const pixelW = canvas.width;
  const pixelH = canvas.height;
  if (pixelW <= 0 || pixelH <= 0) {
    return null;
  }

  const viewport = canvas.closest("[data-zoom-panel]");
  if (!viewport) {
    return null;
  }

  const viewportRect = viewport.getBoundingClientRect();
  const vx = event.clientX - viewportRect.left;
  const vy = event.clientY - viewportRect.top;

  const panelId = viewport.dataset.zoomPanel;
  const zoom = panelZoom[panelId] ?? { scale: 1, panX: 0, panY: 0 };
  const cx = (vx - zoom.panX) / zoom.scale;
  const cy = (vy - zoom.panY) / zoom.scale;

  const imageRect = displayedImageRectInContent(canvas, fitToPanel);
  if (!imageRect) {
    return null;
  }

  const relX = cx - imageRect.left;
  const relY = cy - imageRect.top;
  if (
    relX < 0 ||
    relY < 0 ||
    relX > imageRect.width ||
    relY > imageRect.height
  ) {
    return null;
  }

  const x = Math.round((relX / imageRect.width) * pixelW);
  const y = Math.round((relY / imageRect.height) * pixelH);
  return [
    clamp(x, 0, pixelW - 1),
    clamp(y, 0, pixelH - 1),
  ];
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

function updateGtControlUi() {
  const hasSample = Boolean(state.imageCache);
  for (const button of els.gtViewButtons) {
    const active = button.dataset.gtView === state.gtViewMode;
    button.classList.toggle("active", active);
    button.disabled = !hasSample;
  }
  els.gtOverlayOpacity.hidden = !hasSample || (state.gtViewMode !== "overlay" && state.gtViewMode !== "split");
}

function setGtContentVisibility(mode) {
  els.maskImage.hidden = mode !== "mask";
  els.maskOverlayCanvas.hidden = mode !== "overlay";
  els.maskSplitCanvas.hidden = mode !== "split";
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
export {
  drawCross,
  loadImage,
  renderImageWithCross,
  renderInteractiveImageWithCross,
  containRect,
  displayedImageRectInContent,
  clientToCanvasPixels,
  mediaCacheToken,
  mediaUrl,
  loadSampleImages,
  refreshGtPanel,
  updateGtControlUi,
  setGtContentVisibility,
  renderGtSplitView,
  renderGtPanelView,
};
