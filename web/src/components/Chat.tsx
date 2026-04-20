import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { useChatSocket } from "../hooks/useChatSocket";
import { useTokenCount } from "../hooks/useTokenCount";
import { useStore } from "../store";

export function Chat() {
  const transcript = useStore((s) => s.transcript);
  const setTranscript = useStore((s) => s.setTranscript);
  const currentRun = useStore((s) => s.currentRun);
  const modelInfo = useStore((s) => s.modelInfo);

  const { send, cancel } = useChatSocket();
  const [input, setInput] = useState("");
  const [useCache, setUseCache] = useState(true);
  // Thinking mode on by default (Qwen3's reasoning path). Sampling preset
  // is initialized to match.
  const [enableThinking, setEnableThinking] = useState(true);
  const [maxSeqLen, setMaxSeqLen] = useState(32768);
  // null = "auto, fill the remaining context". A number = user-overridden.
  const [maxNewTokensOverride, setMaxNewTokensOverride] = useState<number | null>(null);

  // Sampling: default to Qwen thinking preset (matches the initial toggle state).
  const [temperature, setTemperature] = useState(0.6);
  const [topP, setTopP] = useState(0.95);
  const topK = 20;
  const minP = 0.0;

  // Live prompt-token count based on the current transcript + draft input,
  // rendered through our Jinja chat template on the server.
  const { promptTokens, maxNewAvailable } = useTokenCount(
    transcript,
    input,
    enableThinking,
  );
  // The effective ceiling: don't exceed what's left in the KV cache budget
  // OR the model's native context, whichever is smaller.
  const ceiling = Math.max(
    1,
    Math.min(
      maxSeqLen - promptTokens,
      maxNewAvailable || modelInfo?.max_position_embeddings || 32768,
    ),
  );
  const effectiveMaxNewTokens =
    maxNewTokensOverride === null ? ceiling : Math.min(maxNewTokensOverride, ceiling);

  // Keep transcript scrolled to the bottom as tokens stream in.
  const bodyRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [transcript, currentRun?.tokens.length]);

  // Adjust the default max_seq_len once we know what the model supports.
  useEffect(() => {
    if (modelInfo) setMaxSeqLen((prev) => Math.min(prev, modelInfo.max_position_embeddings));
  }, [modelInfo]);

  const running = currentRun !== null && !currentRun.done;

  function onSubmit() {
    const text = input.trim();
    if (!text || running) return;
    const msgs = [...transcript, { role: "user" as const, content: text }];
    setTranscript(msgs);
    setInput("");
    send(msgs, {
      use_cache: useCache,
      enable_thinking: enableThinking,
      max_new_tokens: effectiveMaxNewTokens,
      max_seq_len: maxSeqLen,
      temperature,
      top_k: topK,
      top_p: topP,
      min_p: minP,
      seed: null,
    });
    // Focus is restored via the effect below once `running` flips back
    // to false (the textarea is disabled mid-run, so an immediate focus
    // call wouldn't stick).
  }

  // Whenever a generation finishes (running: true → false), put focus
  // back in the composer so the user can immediately type a follow-up.
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) {
      inputRef.current?.focus();
    }
    wasRunning.current = running;
  }, [running]);

  function onThinkingChange(v: boolean) {
    setEnableThinking(v);
    if (v) {
      setTemperature(0.6);
      setTopP(0.95);
    } else {
      setTemperature(0.7);
      setTopP(0.8);
    }
  }

  const [restartInFlight, setRestartInFlight] = useState(false);
  const resetRuns = useStore((s) => s.resetRuns);
  async function onRestart() {
    setRestartInFlight(true);
    try {
      // Tell the server to cancel, drop the cache, flush MPS.
      cancel();
      await fetch("/api/restart", { method: "POST" });
    } catch {
      /* ignore — still clear the local state */
    } finally {
      // Reset the local view regardless. This clears the in-flight token
      // stream and the latency-overlay comparison so the user can start
      // from a clean slate.
      setTranscript([]);
      resetRuns();
      setRestartInFlight(false);
    }
  }

  return (
    <div className="pane">
      <div className="pane-header">
        <span>CHAT</span>
        <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
            {modelInfo ? `${modelInfo.model_name} · ${modelInfo.device} · ${modelInfo.dtype}` : "connecting…"}
          </span>
          <button
            type="button"
            onClick={onRestart}
            disabled={restartInFlight}
            title="cancel any in-flight generation, drop KV cache, flush MPS pool, clear transcript"
            style={{ padding: "2px 8px", fontSize: 11 }}
          >
            {restartInFlight ? "restarting…" : "restart engine"}
          </button>
        </span>
      </div>
      <div className="pane-body" ref={bodyRef}>
        <div className="chat-transcript">
          {transcript.length === 0 && (
            <div className="placeholder">
              Try a long prompt with KV cache OFF to watch the quadratic bend.
            </div>
          )}
          {transcript.map((m, i) => (
            <div className={`msg ${m.role}`} key={i}>
              <div className="msg-role">{m.role}</div>
              <div className="msg-body">
                {m.role === "assistant" ? <ReactMarkdown>{m.content}</ReactMarkdown> : m.content}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="chat-composer">
        <div className="chat-controls">
          <label>
            <input
              type="checkbox"
              checked={useCache}
              onChange={(e) => setUseCache(e.target.checked)}
              disabled={running}
            />
            KV cache
          </label>
          <label>
            <input
              type="checkbox"
              checked={enableThinking}
              onChange={(e) => onThinkingChange(e.target.checked)}
              disabled={running}
            />
            thinking mode
          </label>
          <label title="Defaults to max_seq_len − prompt tokens. Edit to override.">
            max_new_tokens
            <input
              type="number"
              min={1}
              max={modelInfo?.max_position_embeddings ?? 32768}
              value={effectiveMaxNewTokens}
              onChange={(e) => {
                const v = Number(e.target.value);
                setMaxNewTokensOverride(Number.isFinite(v) && v > 0 ? v : null);
              }}
              disabled={running}
            />
            {maxNewTokensOverride !== null && (
              <button
                type="button"
                onClick={() => setMaxNewTokensOverride(null)}
                style={{ padding: "2px 6px", marginLeft: 4 }}
                title="reset to auto"
                disabled={running}
              >
                auto
              </button>
            )}
          </label>
          <span
            style={{ fontFamily: "var(--font-mono)", color: "var(--fg-subtle)", fontSize: 11 }}
            title="prompt tokens after chat template (live)"
          >
            prompt: {promptTokens.toLocaleString()} tok
          </span>
          <label>
            max_seq_len
            <input
              type="number"
              min={128}
              max={modelInfo?.max_position_embeddings ?? 32768}
              step={128}
              value={maxSeqLen}
              onChange={(e) => setMaxSeqLen(Math.max(128, Number(e.target.value) || 128))}
              disabled={running || !useCache}
            />
          </label>
          <label>
            T
            <input
              type="number"
              step={0.05}
              min={0}
              max={2}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              disabled={running}
              style={{ width: 70 }}
            />
          </label>
          <label>
            top_p
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={topP}
              onChange={(e) => setTopP(Number(e.target.value))}
              disabled={running}
              style={{ width: 70 }}
            />
          </label>
        </div>
        <div className="chat-input-row">
          <textarea
            ref={inputRef}
            className="chat-input"
            autoFocus
            placeholder="Ask something…  (Cmd/Ctrl+Enter to send)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                onSubmit();
              }
            }}
            disabled={running}
          />
          {running ? (
            <button
              className="stop"
              onClick={cancel}
              title="stop inference (finishes the current token, then halts)"
            >
              stop
            </button>
          ) : (
            <button className="primary" onClick={onSubmit} disabled={!input.trim()}>
              send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
