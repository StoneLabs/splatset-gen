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

function setAiContentVisibility(ctx, { showImage, showCanvas }) {
  const refs = ctx.refs;
  refs.predictionImage().hidden = !showImage;
  refs.predictionCanvas().hidden = !showCanvas;
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
export {
  imageDataUsesAlphaChannel,
  maskWeightFromPixel,
  readAlphaPlane,
  setCanvasPixel,
  blitImageData,
  composeRgbOverlay,
  renderSideBySideCanvas,
  renderMaskOverlayOnCanvas,
};
