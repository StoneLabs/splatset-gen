const AI_FORMATS = new Set(["alpha", "binary"]);
const AI_DISPLAY_MODES = new Set(["raw", "compare"]);
const AI_COMPOSE_MODES = new Set(["mask", "overlay", "split"]);
const AI_MASK_BACKGROUNDS = new Set(["transparent", "black"]);

const AI_COMPARE_COLORS = {
  tp: [56, 203, 92],
  fp: [235, 64, 64],
  fn: [255, 255, 255],
};
export {
  AI_FORMATS,
  AI_DISPLAY_MODES,
  AI_COMPOSE_MODES,
  AI_MASK_BACKGROUNDS,
  AI_COMPARE_COLORS,
};
