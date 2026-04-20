import { create } from "zustand";
import type {
  ChatMessage,
  GenToken,
  ModelInfo,
  TelemetrySample,
} from "./types/ws";

export interface TokenRecord {
  id: number;
  text: string;
  step_ms: number;
  seq_len: number;
  kv_length: number;
}

export interface RunRecord {
  startedAt: number;
  use_cache: boolean;
  max_seq_len: number;
  tokens: TokenRecord[];
  done: boolean;
  total_ms?: number;
  finish_reason?: string;
}

interface Store {
  // Connection
  wsConnected: boolean;
  telemetryConnected: boolean;
  setWsConnected: (v: boolean) => void;
  setTelemetryConnected: (v: boolean) => void;

  // Model info (from /ws/telemetry on connect)
  modelInfo: ModelInfo | null;
  setModelInfo: (m: ModelInfo) => void;

  // Telemetry stream (capped ring)
  telemetry: TelemetrySample[];
  pushTelemetry: (s: TelemetrySample) => void;

  // Latest sample (fast access for cards / heatmap)
  latest: TelemetrySample | null;

  // Chat state
  transcript: ChatMessage[];
  setTranscript: (m: ChatMessage[]) => void;
  appendAssistantToken: (t: GenToken) => void;
  finalizeAssistant: () => void;

  // Current-run state (tokens with latency, and the prior run for overlay)
  currentRun: RunRecord | null;
  previousRun: RunRecord | null;
  startRun: (use_cache: boolean, max_seq_len: number) => void;
  appendRunToken: (t: TokenRecord) => void;
  endRun: (total_ms: number, finish_reason: string) => void;
  resetRuns: () => void;

  // Global error line
  errorMessage: string | null;
  setError: (s: string | null) => void;
}

const TELEMETRY_CAP = 2000; // ~100s at 20 Hz

export const useStore = create<Store>((set) => ({
  wsConnected: false,
  telemetryConnected: false,
  setWsConnected: (v) => set({ wsConnected: v }),
  setTelemetryConnected: (v) => set({ telemetryConnected: v }),

  modelInfo: null,
  setModelInfo: (m) => set({ modelInfo: m }),

  telemetry: [],
  latest: null,
  pushTelemetry: (s) =>
    set((st) => {
      const next = st.telemetry.length >= TELEMETRY_CAP ? st.telemetry.slice(1) : st.telemetry;
      return { telemetry: [...next, s], latest: s };
    }),

  transcript: [],
  setTranscript: (m) => set({ transcript: m }),
  appendAssistantToken: (t) =>
    set((st) => {
      const tr = [...st.transcript];
      const last = tr[tr.length - 1];
      if (last && last.role === "assistant" && !last.content.endsWith("<|im_end|>")) {
        tr[tr.length - 1] = { ...last, content: last.content + t.text };
      } else {
        tr.push({ role: "assistant", content: t.text });
      }
      return { transcript: tr };
    }),
  finalizeAssistant: () =>
    set((st) => {
      const tr = st.transcript.map((m) =>
        m.role === "assistant"
          ? { ...m, content: m.content.replaceAll("<|im_end|>", "").trimEnd() }
          : m,
      );
      return { transcript: tr };
    }),

  currentRun: null,
  previousRun: null,
  startRun: (use_cache, max_seq_len) =>
    set((st) => ({
      previousRun: st.currentRun,
      currentRun: {
        startedAt: performance.now(),
        use_cache,
        max_seq_len,
        tokens: [],
        done: false,
      },
    })),
  appendRunToken: (t) =>
    set((st) => {
      if (!st.currentRun) return st;
      return {
        currentRun: { ...st.currentRun, tokens: [...st.currentRun.tokens, t] },
      };
    }),
  endRun: (total_ms, finish_reason) =>
    set((st) => {
      if (!st.currentRun) return st;
      return {
        currentRun: { ...st.currentRun, done: true, total_ms, finish_reason },
      };
    }),
  resetRuns: () => set({ currentRun: null, previousRun: null }),

  errorMessage: null,
  setError: (s) => set({ errorMessage: s }),
}));
