import { useStore } from "../store";
import { palette } from "../theme";

export function KVLayerHeatmap() {
  const latest = useStore((s) => s.latest);
  const info = useStore((s) => s.modelInfo);

  const numLayers = info?.num_layers ?? 28;
  const per = latest?.kv_per_layer_bytes_used ?? [];
  const alloc = latest?.kv_cache_bytes_allocated ?? 0;
  const perLayerCapacity = alloc / Math.max(1, numLayers);

  return (
    <div className="card">
      <h3>per-layer KV fill · {numLayers} blocks</h3>
      <div
        className="kv-heatmap"
        style={{ gridTemplateColumns: `repeat(${numLayers}, 1fr)` }}
      >
        {Array.from({ length: numLayers }, (_, i) => {
          const bytes = per[i] ?? 0;
          const pct = perLayerCapacity > 0 ? bytes / perLayerCapacity : 0;
          return (
            <div
              key={i}
              className="cell"
              title={`layer ${i}: ${(pct * 100).toFixed(1)}%`}
              style={{
                background: `rgba(${palette.accentRgb}, ${Math.min(1, pct)})`,
                borderColor: pct > 0 ? palette.accent : palette.borderSoft,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
