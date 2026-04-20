import { useCallback, useEffect, useRef } from "react";
import { useStore } from "../store";
import type { ChatMsg, GenerateRequest } from "../types/ws";

export interface SendOpts {
  use_cache: boolean;
  enable_thinking: boolean;
  max_new_tokens: number;
  max_seq_len: number;
  temperature: number;
  top_k: number;
  top_p: number;
  min_p: number;
  seed: number | null;
}

export function useChatSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const store = useStore.getState; // avoid re-render loop

  useEffect(() => {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/chat`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => useStore.setState({ wsConnected: true });
    ws.onclose = () => useStore.setState({ wsConnected: false });
    ws.onerror = () => useStore.setState({ wsConnected: false });
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as ChatMsg;
      const s = store();
      switch (msg.type) {
        case "started":
          s.startRun(msg.use_cache, msg.max_seq_len);
          break;
        case "token":
          s.appendAssistantToken(msg);
          s.appendRunToken({
            id: msg.token_id,
            text: msg.text,
            step_ms: msg.step_ms,
            seq_len: msg.seq_len,
            kv_length: msg.kv_length,
          });
          break;
        case "done":
          s.finalizeAssistant();
          s.endRun(msg.total_ms, msg.finish_reason);
          break;
        case "error":
          s.setError(msg.message);
          break;
      }
    };
    return () => ws.close();
  }, [store]);

  const send = useCallback(
    (messages: GenerateRequest["messages"], opts: SendOpts) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const req: GenerateRequest = { type: "generate", messages, ...opts };
      ws.send(JSON.stringify(req));
    },
    [],
  );

  const cancel = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "cancel" }));
    }
  }, []);

  return { send, cancel };
}
