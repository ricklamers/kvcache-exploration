import { useStore } from "../store";
import { KVCacheBar } from "./KVCacheBar";
import { KVLayerHeatmap } from "./KVLayerHeatmap";
import { LatencyChart } from "./LatencyChart";
import { MemoryChart } from "./MemoryChart";

const MB = 1024 * 1024;
const GB = 1024 * MB;

function fmt(n: number): string {
  if (n >= GB) return `${(n / GB).toFixed(2)} GB`;
  if (n >= MB) return `${(n / MB).toFixed(1)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
}

export function Telemetry() {
  const latest = useStore((s) => s.latest);
  const breakdown = latest?.params_breakdown ?? {};
  const currentRun = useStore((s) => s.currentRun);
  const tokensPerSec =
    currentRun && currentRun.tokens.length > 1
      ? (currentRun.tokens.length /
          ((performance.now() - currentRun.startedAt) / 1000)
        ).toFixed(1)
      : "—";

  return (
    <div className="pane">
      <div className="pane-header">
        <span>TELEMETRY</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
          {latest ? `${fmt(latest.driver_bytes)} driver · ${fmt(latest.current_allocated_bytes)} live` : "…"}
        </span>
      </div>
      <div className="telemetry">
        <div className="card">
          <h3>parameters · 1.2 GB in bf16</h3>
          <div className="card-grid">
            {["embeddings", "attention", "mlp", "norms", "lm_head"].map((k) => (
              <div key={k}>
                <div className="k">{k}</div>
                <div className="v">{fmt(breakdown[k] ?? 0)}</div>
              </div>
            ))}
            <div>
              <div className="k">total params</div>
              <div className="v accent">{fmt(breakdown.total ?? 0)}</div>
            </div>
            <div>
              <div className="k">activations</div>
              <div className="v">{fmt(latest?.activations_bytes ?? 0)}</div>
            </div>
            <div>
              <div className="k">tokens/s</div>
              <div className="v accent">{tokensPerSec}</div>
            </div>
          </div>
        </div>

        <KVCacheBar />
        <KVLayerHeatmap />
        <MemoryChart />
        <LatencyChart />
      </div>
    </div>
  );
}
