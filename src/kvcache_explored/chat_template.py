"""Render Qwen3 chat prompts via the official Jinja template.

The template itself lives in ``templates/qwen3_chat.jinja`` (copied verbatim
from the tokenizer's ``chat_template`` field). That file is the single source
of truth — there is no hand-written string formatting of ``<|im_start|>``
anywhere in this codebase.

Parity with ``tokenizer.apply_chat_template`` is asserted by the verifier.
"""

from __future__ import annotations

from importlib.resources import files
from typing import TypedDict

from jinja2.sandbox import ImmutableSandboxedEnvironment


class ChatMessage(TypedDict):
    role: str
    content: str


# Match Hugging Face's sandbox so missing attrs (e.g. `message.tool_calls` on
# a plain user message) evaluate to undefined-but-falsy, the behavior the
# Qwen3 template is authored against.
_ENV = ImmutableSandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    autoescape=False,
)

# HF tokenizers' chat_template renderer exposes a `raise_exception` global; the
# Qwen3 template uses it for error branches we don't hit in practice, but we
# still register it so any future template tweak keeps working.
_ENV.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))

_TEMPLATE_SRC = (files("kvcache_explored") / "templates" / "qwen3_chat.jinja").read_text()
_TEMPLATE = _ENV.from_string(_TEMPLATE_SRC)


def render(
    messages: list[ChatMessage],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    tools: list[dict] | None = None,
) -> str:
    """Render a conversation to the exact prompt Qwen3 expects."""
    return _TEMPLATE.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        tools=tools,
    )
