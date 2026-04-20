import { useEffect, useState } from "react";
import type { ChatMessage } from "../types/ws";

interface TokenizeResponse {
  prompt_tokens: number;
  prompt_chars: number;
  max_new_tokens_available: number;
}

/**
 * Live-tokenize the current transcript + any draft user message.
 * Debounced so we don't POST on every keystroke. Returns both the prompt
 * token count and how many generation tokens fit in the full context
 * window (max_position_embeddings − prompt_tokens).
 */
export function useTokenCount(
  messages: ChatMessage[],
  draft: string,
  enableThinking: boolean,
): { promptTokens: number; maxNewAvailable: number } {
  const [state, setState] = useState({ promptTokens: 0, maxNewAvailable: 0 });

  useEffect(() => {
    let cancelled = false;
    const handle = setTimeout(() => {
      const payload = {
        messages: draft.trim()
          ? [...messages, { role: "user" as const, content: draft }]
          : messages,
        enable_thinking: enableThinking,
        add_generation_prompt: true,
      };
      fetch("/api/tokenize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
        .then((r) => r.json() as Promise<TokenizeResponse>)
        .then((d) => {
          if (cancelled) return;
          setState({
            promptTokens: d.prompt_tokens,
            maxNewAvailable: d.max_new_tokens_available,
          });
        })
        .catch(() => {
          /* server not ready, ignore */
        });
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [messages, draft, enableThinking]);

  return state;
}
