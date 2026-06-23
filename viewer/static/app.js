const state = {
  total: 0,
  index: 0,
  configYaml: "",
  loading: false,
};

const els = {
  datasetPath: document.getElementById("dataset-path"),
  sampleIndex: document.getElementById("sample-index"),
  sampleId: document.getElementById("sample-id"),
  sampleCount: document.getElementById("sample-count"),
  statusText: document.getElementById("status-text"),
  clickLabel: document.getElementById("click-label"),
  objectLabel: document.getElementById("object-label"),
  renderCanvas: document.getElementById("render-canvas"),
  maskImage: document.getElementById("mask-image"),
  annotationJson: document.getElementById("annotation-json"),
  configYaml: document.getElementById("config-yaml"),
  btnFirst: document.getElementById("btn-first"),
  btnPrev: document.getElementById("btn-prev"),
  btnNext: document.getElementById("btn-next"),
  btnLast: document.getElementById("btn-last"),
  btnCopyJson: document.getElementById("btn-copy-json"),
  btnCopyConfig: document.getElementById("btn-copy-config"),
};

function setStatus(text) {
  els.statusText.textContent = text;
}

function updateNavButtons() {
  const atStart = state.index <= 0;
  const atEnd = state.index >= state.total - 1;
  els.btnFirst.disabled = atStart || state.loading;
  els.btnPrev.disabled = atStart || state.loading;
  els.btnNext.disabled = atEnd || state.loading;
  els.btnLast.disabled = atEnd || state.loading;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

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

async function renderImageWithCross(imagePath, point) {
  const url = `/media/${imagePath}?t=${encodeURIComponent(state.index)}`;
  const img = await loadImage(url);
  const canvas = els.renderCanvas;
  const ctx = canvas.getContext("2d");

  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  const [x, y] = point;
  drawCross(ctx, x, y, canvas.width, canvas.height);
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
    const payload = await fetchJson(`/api/sample/index/${clamped}`);
    const record = payload.record;
    state.index = payload.index;

    els.sampleIndex.value = String(state.index);
    els.sampleId.value = record.id;
    els.sampleCount.textContent = `/ ${state.total.toLocaleString()}`;

    const [x, y] = record.point;
    els.clickLabel.textContent = `point [${x}, ${y}]`;
    els.objectLabel.textContent = `object ${record.object_id} / ${record.num_objects}`;

    await renderImageWithCross(record.image, record.point);
    els.maskImage.src = `/media/${record.mask}?t=${state.index}`;
    els.annotationJson.textContent = JSON.stringify(record, null, 2);

    setStatus(`Sample ${state.index + 1} · id ${record.id}`);
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${error.message}`);
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

async function init() {
  try {
    const meta = await fetchJson("/api/meta");
    state.total = meta.count;
    els.datasetPath.textContent = meta.dataset_dir;
    els.sampleIndex.max = Math.max(0, state.total - 1);
    els.sampleCount.textContent = `/ ${state.total.toLocaleString()}`;

    const config = await fetchJson("/api/config");
    state.configYaml = config.yaml;
    els.configYaml.textContent = state.configYaml;

    if (state.total === 0) {
      setStatus("Dataset is empty");
      updateNavButtons();
      return;
    }

    await showSample(0);
  } catch (error) {
    console.error(error);
    setStatus(`Failed to initialize: ${error.message}`);
  }
}

els.btnFirst.addEventListener("click", () => showSample(0));
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

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) {
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
  }
});

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    setStatus(`Copied ${label}`);
  } catch {
    setStatus(`Could not copy ${label}`);
  }
}

els.btnCopyJson.addEventListener("click", () => {
  copyText(els.annotationJson.textContent, "annotation JSON");
});

els.btnCopyConfig.addEventListener("click", () => {
  copyText(state.configYaml, "config.yaml");
});

init();
