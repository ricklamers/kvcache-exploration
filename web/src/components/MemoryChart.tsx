import { useEffect, useRef } from "react";
import uPlot from "uplot";
import { palette } from "../theme";
import { useStore } from "../store";

const MB = 1024 * 1024;

export function MemoryChart() {
  const el = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);

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
          values: (_, splits) => splits.map((v) => `${v.toFixed(0)}s`),
        },
        {
          stroke: palette.fgMuted,
          grid: { stroke: palette.borderSoft, width: 1 },
          values: (_, splits) => splits.map((v) => `${(v / MB).toFixed(0)} MB`),
          size: 80,
        },
      ],
      series: [
        {},
        { label: "params", stroke: palette.seriesParams, fill: palette.seriesParams + "44", width: 1.5 },
        { label: "kv cache", stroke: palette.seriesKvCache, fill: palette.seriesKvCache + "66", width: 1.5 },
        { label: "activations", stroke: palette.seriesActivations, fill: palette.seriesActivations + "44", width: 1.5 },
        { label: "driver", stroke: palette.fgSubtle, width: 1, dash: [4, 3] },
      ],
    };
    plot.current = new uPlot(opts, [[], [], [], [], []], el.current);
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

  const telemetry = useStore((s) => s.telemetry);

  useEffect(() => {
    if (!plot.current || telemetry.length === 0) return;
    const t = telemetry.map((s) => s.t_ms / 1000);
    const params = telemetry.map((s) => s.params_bytes);
    const kv = telemetry.map((s) => s.params_bytes + s.kv_cache_bytes_allocated);
    const act = telemetry.map((s) => s.params_bytes + s.kv_cache_bytes_allocated + s.activations_bytes);
    const drv = telemetry.map((s) => s.driver_bytes);
    plot.current.setData([t, params, kv, act, drv]);
  }, [telemetry]);

  return (
    <div className="card" style={{ flex: 1, minHeight: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <h3 style={{ margin: 0 }}>memory over time (stacked)</h3>
        <div className="chart-legend">
          <Swatch color={palette.seriesParams} />params
          <Swatch color={palette.seriesKvCache} />kv cache
          <Swatch color={palette.seriesActivations} />activations
          <Swatch color={palette.fgSubtle} />driver
        </div>
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
