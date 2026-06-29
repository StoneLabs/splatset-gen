function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(3);
}

async function readErrorResponse(response) {
  const text = await response.text();
  try {
    const payload = JSON.parse(text);
    if (typeof payload.error === "string" && payload.error) {
      return payload.error;
    }
  } catch {
    // Not JSON — fall through to HTML/plain parsing.
  }

  const htmlMatch = text.match(/<p>([^<]+)<\/p>/i);
  if (htmlMatch?.[1]) {
    return htmlMatch[1].trim();
  }

  const trimmed = text.trim();
  if (trimmed) {
    return trimmed.slice(0, 500);
  }
  return `Request failed: ${response.status}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function syncSegmentedButtons(buttons, isActive, disabled) {
  for (const button of buttons) {
    button.classList.toggle("active", isActive(button));
    button.disabled = disabled;
  }
}
export { clamp, formatMetric, readErrorResponse, fetchJson, syncSegmentedButtons };
