import { useStore } from "../store";

const MB = 1024 * 1024;
const GB = 1024 * MB;

function fmtBytes(n: number): string {
  if (n >= GB) return `${(n / GB).toFixed(2)} GB`;
  if (n >= MB) return `${(n / MB).toFixed(1)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
}

export function KVCacheBar() {
  const latest = useStore((s) => s.latest);

  const len = latest?.kv_cache_length ?? 0;
  const cap = latest?.kv_cache_capacity ?? 0;
  const alloc = latest?.kv_cache_bytes_allocated ?? 0;
  const used = latest?.kv_cache_bytes_used ?? 0;
  const pct = cap > 0 ? len / cap : 0;
  const pctAlloc = alloc > 0 ? used / alloc : 0;
  const warn = pct > 0.75;
  const full = pct > 0.95;

  return (
    <div className="card">
      <h3>KV cache</h3>
      <div className="card-grid">
        <div>
          <div className="k">tokens</div>
          <div className="v">
            {len.toLocaleString()} / {cap.toLocaleString()}
          </div>
        </div>
        <div>
          <div className="k">bytes used</div>
          <div className={`v ${full ? "error" : warn ? "warn" : "accent"}`}>
            {fmtBytes(used)} / {fmtBytes(alloc)}
          </div>
        </div>
        <div>
          <div className="k">fill</div>
          <div className={`v ${full ? "error" : warn ? "warn" : "accent"}`}>
            {(pct * 100).toFixed(1)}%
          </div>
        </div>
      </div>
      <div className="kv-bar-outer">
        <div
          className={`kv-bar-inner ${full ? "full" : warn ? "warn" : ""}`}
          style={{ width: `${pctAlloc * 100}%` }}
        />
      </div>
    </div>
  );
}
