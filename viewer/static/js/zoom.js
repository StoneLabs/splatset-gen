import { panelZoom, ZOOM_LIMITS } from "./shared.js";
import { clamp } from "./utils.js";

function resetPanelZoom(panelId) {
  panelZoom[panelId] = { scale: 1, panX: 0, panY: 0 };
  applyPanelZoom(panelId);
}

function resetAllPanelZoom() {
  for (const panelId of Object.keys(panelZoom)) {
    resetPanelZoom(panelId);
  }
}

function applyPanelZoom(panelId) {
  const content = document.querySelector(`[data-zoom-content="${panelId}"]`);
  if (!content) {
    return;
  }
  const { scale, panX, panY } = panelZoom[panelId];
  content.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
}

function setupPanelZoom(viewport, panelId, { deferDrag = false } = {}) {
  let dragging = false;
  let pending = false;
  let startX = 0;
  let startY = 0;
  let startPanX = 0;
  let startPanY = 0;
  const dragThreshold = deferDrag ? 5 : 0;

  viewport.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const zoom = panelZoom[panelId];
      const rect = viewport.getBoundingClientRect();
      const cursorX = event.clientX - rect.left;
      const cursorY = event.clientY - rect.top;
      const factor = Math.exp(-event.deltaY * 0.0015);
      const nextScale = clamp(zoom.scale * factor, ZOOM_LIMITS.min, ZOOM_LIMITS.max);
      const ratio = nextScale / zoom.scale;
      zoom.panX = cursorX - ratio * (cursorX - zoom.panX);
      zoom.panY = cursorY - ratio * (cursorY - zoom.panY);
      zoom.scale = nextScale;
      applyPanelZoom(panelId);
    },
    { passive: false },
  );

  viewport.addEventListener("mousedown", (event) => {
    if (event.button !== 0) {
      return;
    }
    pending = true;
    startX = event.clientX;
    startY = event.clientY;
    startPanX = panelZoom[panelId].panX;
    startPanY = panelZoom[panelId].panY;
    if (!deferDrag) {
      dragging = true;
      viewport.classList.add("dragging");
      event.preventDefault();
    }
  });

  window.addEventListener("mousemove", (event) => {
    if (pending && !dragging) {
      const distance = Math.hypot(event.clientX - startX, event.clientY - startY);
      if (distance > dragThreshold) {
        dragging = true;
        viewport.classList.add("dragging");
      }
    }
    if (!dragging) {
      return;
    }
    panelZoom[panelId].panX = startPanX + (event.clientX - startX);
    panelZoom[panelId].panY = startPanY + (event.clientY - startY);
    applyPanelZoom(panelId);
  });

  window.addEventListener("mouseup", () => {
    if (!pending && !dragging) {
      return;
    }
    pending = false;
    dragging = false;
    viewport.classList.remove("dragging");
  });

  viewport.addEventListener("dblclick", () => {
    resetPanelZoom(panelId);
  });
}

function setupPanelZoomControls() {
  for (const viewport of document.querySelectorAll("[data-zoom-panel]")) {
    const panelId = viewport.dataset.zoomPanel;
    setupPanelZoom(viewport, panelId, { deferDrag: panelId === "interactive" });
  }
}
export {
  resetPanelZoom,
  resetAllPanelZoom,
  applyPanelZoom,
  setupPanelZoom,
  setupPanelZoomControls,
};
