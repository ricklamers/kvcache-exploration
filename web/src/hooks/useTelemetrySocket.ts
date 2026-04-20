import { useEffect } from "react";
import { useStore } from "../store";
import type { TelemetryMsg } from "../types/ws";

export function useTelemetrySocket() {
  useEffect(() => {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/telemetry`;
    const ws = new WebSocket(url);
    ws.onopen = () => useStore.setState({ telemetryConnected: true });
    ws.onclose = () => useStore.setState({ telemetryConnected: false });
    ws.onerror = () => useStore.setState({ telemetryConnected: false });
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as TelemetryMsg;
      if (msg.type === "info") {
        useStore.getState().setModelInfo(msg);
      } else if (msg.type === "mem") {
        useStore.getState().pushTelemetry(msg);
      }
    };
    return () => ws.close();
  }, []);
}
