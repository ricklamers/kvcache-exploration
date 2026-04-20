/**
 * VS Code / Cursor dark palette with NVIDIA green (#76B900) as the accent.
 * Exposed as CSS custom props via styles.css; imported here too so the
 * canvas-drawing components (charts, heatmap) can reach the palette
 * directly without going through CSS variables.
 */
export const palette = {
  bg: "#1e1e1e",
  panel: "#252526",
  panelDeep: "#1f1f20",
  border: "#3c3c3c",
  borderSoft: "#2b2b2b",

  fg: "#d4d4d4",
  fgMuted: "#858585",
  fgSubtle: "#6a6a6a",

  accent: "#76b900", // NVIDIA green — cache-on, primary CTA, fills
  accentRgb: "118, 185, 0", // for rgba() use in canvas heatmap
  accentDim: "#4a7400", // darker variant for backgrounds / status bar
  warn: "#dcdcaa", // KV cache near capacity
  error: "#f48771", // quadratic bend highlight, cache overflow
  info: "#569cd6", // status / info line
  magenta: "#c586c0",

  seriesParams: "#569cd6",
  seriesKvCache: "#76b900",
  seriesActivations: "#dcdcaa",
  seriesDriver: "#3a3a3a",
  seriesLatencyCacheOn: "#76b900",
  seriesLatencyCacheOff: "#f48771",
} as const;

export const font = {
  ui: `"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`,
  mono: `"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace`,
} as const;
