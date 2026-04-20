import { useStore } from "../store";

const MB = 1024 * 1024;
const GB = 1024 * MB;

function fmt(n: number): string {
  if (n >= GB) return `${(n / GB).toFixed(2)} GB`;
  if (n >= MB) return `${(n / MB).toFixed(1)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
}

export function StatusBar() {
  const chatConn = useStore((s) => s.wsConnected);
  const telConn = useStore((s) => s.telemetryConnected);
  const info = useStore((s) => s.modelInfo);
  const latest = useStore((s) => s.latest);
  const currentRun = useStore((s) => s.currentRun);
  const disconnected = !chatConn || !telConn;
  const tokensPerSec =
    currentRun && currentRun.tokens.length > 1
      ? (currentRun.tokens.length /
          ((performance.now() - currentRun.startedAt) / 1000)
        ).toFixed(1)
      : "—";

  return (
    <div className={`status-bar${disconnected ? " disconnected" : ""}`}>
      <span>
        <span className="lbl">●</span>
        {disconnected ? "disconnected" : "connected"}
      </span>
      <span className="sep">│</span>
      <span>{info ? `${info.model_name}` : "—"}</span>
      <span className="sep">│</span>
      <span>{info ? `${info.device}/${info.dtype}` : "—"}</span>
      <span className="sep">│</span>
      <span>
        <span className="lbl">kv/tok</span>
        {info ? `${info.bytes_per_kv_token} B` : "—"}
      </span>
      <span className="sep">│</span>
      <span>
        <span className="lbl">driver</span>
        {latest ? fmt(latest.driver_bytes) : "—"}
      </span>
      <span className="sep">│</span>
      <span>
        <span className="lbl">tok/s</span>
        {tokensPerSec}
      </span>
      <span style={{ flex: 1 }} />
      <span>{currentRun?.finish_reason ? `finish: ${currentRun.finish_reason}` : ""}</span>
    </div>
  );
}
