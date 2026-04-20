import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";
import { palette } from "../theme";
import { useStore } from "../store";

type WindowMode = "skipFirst" | "all";

export function LatencyChart() {
  const el = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  // The first token is prefill + MPS-kernel warmup and is 5–20× slower
  // than steady-state decode. Default to skipping it so the y-axis fits
  // the actual decode shape; toggle to "all" to see the prefill spike.
  const [windowMode, setWindowMode] = useState<WindowMode>("skipFirst");

  useEffect(() => {
    if (!el.current) return;
    const opts: uPlot.Options = {
      width: el.current.clientWidth,
      height: el.current.clientHeight,
      padding: [8, 8, 4, 4],
      // Built-in legend renders as a sibling div inside the chart root
      // and breaks the parent's flex sizing — we ship a small inline
      // legend in the card header instead.
      legend: { show: false },
      cursor: { drag: { x: true, y: false } },
      scales: {
        x: { time: false },
        y: { auto: true, distr: 1 },
      },
      axes: [
        {
          stroke: palette.fgMuted,
          grid: { stroke: palette.borderSoft, width: 1 },
          label: "token #",
          labelSize: 18,
          labelFont: "11px 'JetBrains Mono'",
        },
        {
          stroke: palette.fgMuted,
          grid: { stroke: palette.borderSoft, width: 1 },
          values: (_, splits) => splits.map((v) => `${v.toFixed(0)} ms`),
          size: 60,
        },
      ],
      series: [
        {},
        { label: "current run", stroke: palette.seriesLatencyCacheOn, width: 1.75, points: { size: 3 } },
        { label: "previous run", stroke: palette.fgSubtle, width: 1, dash: [3, 3], points: { show: false } },
      ],
    };
    plot.current = new uPlot(opts, [[], [], []], el.current);
    const ro = new ResizeObserver(() => {
      if (plot.current && el.current) plot.current.setSize({ width: el.current.clientWidth, height: el.current.clientHeight });
    });
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
  }, []);

  const currentRun = useStore((s) => s.currentRun);
  const previousRun = useStore((s) => s.previousRun);

  useEffect(() => {
    if (!plot.current) return;
    const maxLen = Math.max(currentRun?.tokens.length ?? 0, previousRun?.tokens.length ?? 0);
    // Skip the first token (token #0) when in "skipFirst" mode. We keep
    // the original token-index on the x-axis so the labels stay accurate
    // and the rest of the curve doesn't get re-numbered.
    const start = windowMode === "skipFirst" && maxLen > 1 ? 1 : 0;
    const length = maxLen - start;
    const x = Array.from({ length }, (_, i) => i + start);
    const cur = x.map((i) => currentRun?.tokens[i]?.step_ms ?? null);
    const prev = x.map((i) => previousRun?.tokens[i]?.step_ms ?? null);

    // Flag cache state with color by swapping the series stroke.
    const series = plot.current.series;
    if (series[1] && currentRun) {
      const color = currentRun.use_cache ? palette.seriesLatencyCacheOn : palette.seriesLatencyCacheOff;
      plot.current.delSeries(1);
      plot.current.addSeries(
        { label: currentRun.use_cache ? "current (cache on)" : "current (cache off)", stroke: color, width: 1.75, points: { size: 3 } },
        1,
      );
    }
    plot.current.setData([x, cur as number[], prev as number[]]);
  }, [currentRun, previousRun, windowMode]);

  const totalTokens = Math.max(
    currentRun?.tokens.length ?? 0,
    previousRun?.tokens.length ?? 0,
  );
  // What does the first token's latency look like relative to the rest?
  // Showing it lets the user know whether toggling will actually matter.
  const firstMs = currentRun?.tokens[0]?.step_ms;

  return (
    <div className="card" style={{ flex: 1, minHeight: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <h3 style={{ margin: 0 }}>
          per-token decode latency
          {firstMs !== undefined && (
            <span style={{ marginLeft: 8, color: "var(--fg-subtle)", fontSize: 10, fontFamily: "var(--font-mono)" }}>
              prefill+warmup: {firstMs.toFixed(0)} ms
            </span>
          )}
        </h3>
        <div style={{ display: "inline-flex", border: "1px solid var(--border-soft)" }}>
          <button
            type="button"
            onClick={() => setWindowMode("skipFirst")}
            style={{
              padding: "2px 8px",
              fontSize: 11,
              border: "none",
              background: windowMode === "skipFirst" ? "var(--accent-dim)" : "transparent",
              color: windowMode === "skipFirst" ? "#0c0c0c" : "var(--fg-muted)",
              fontWeight: windowMode === "skipFirst" ? 600 : 400,
            }}
            title="hide token #0 (prefill+warmup) so the y-axis fits the steady-state decode"
          >
            skip 1st
          </button>
          <button
            type="button"
            onClick={() => setWindowMode("all")}
            style={{
              padding: "2px 8px",
              fontSize: 11,
              border: "none",
              background: windowMode === "all" ? "var(--accent-dim)" : "transparent",
              color: windowMode === "all" ? "#0c0c0c" : "var(--fg-muted)",
              fontWeight: windowMode === "all" ? 600 : 400,
            }}
            title="show every token from the current run (prefill spike included)"
          >
            all
          </button>
        </div>
      </div>
      <div className="chart-legend">
        <Swatch color={currentRun?.use_cache === false ? palette.seriesLatencyCacheOff : palette.seriesLatencyCacheOn} />
        current{currentRun ? (currentRun.use_cache ? " (cache on)" : " (cache off)") : ""}
        <Swatch color={palette.fgSubtle} />previous
        <span style={{ marginLeft: 14, color: "var(--fg-subtle)" }}>{totalTokens} tokens</span>
      </div>
      <div ref={el} style={{ flex: 1, minHeight: 0, position: "relative" }} />
    </div>
  );
}

function Swatch({ color }: { color: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        background: color,
        marginRight: 4,
        marginLeft: 8,
        verticalAlign: "middle",
      }}
    />
  );
}
