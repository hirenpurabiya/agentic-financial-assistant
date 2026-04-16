"""
Agentic Financial Assistant — Gradio UI for HuggingFace Spaces.

Conversational interface over the multi agent graph. Shows which agents
were dispatched for each query and visualizes the LangGraph architecture.
"""

from __future__ import annotations

import time
import uuid

import gradio as gr
from langchain_core.messages import HumanMessage

from src.config import logger
from src.graph import graph


# --- Abuse protection ---

MAX_INPUT_CHARS = 500
CALL_DELAY = 2

# Voice-specific limits: stricter because STT + LLM + TTS = 3 Gemini calls per turn.
VOICE_MAX_SECONDS = 5
VOICE_MAX_PER_HOUR = 10  # per thread/session
VOICE_CALL_DELAY = 3


_last_call = {"ts": 0.0}
_last_voice_call = {"ts": 0.0}
_voice_buckets: dict[str, list[float]] = {}


def _rate_limit() -> None:
    now = time.time()
    elapsed = now - _last_call["ts"]
    if elapsed < CALL_DELAY:
        time.sleep(CALL_DELAY - elapsed)
    _last_call["ts"] = time.time()


def _rate_limit_voice(thread_id: str) -> tuple[bool, str]:
    """Enforce per-session voice quota. Returns (allowed, reason)."""
    now = time.time()

    # Global pacing between voice calls (coarse anti-burst).
    elapsed = now - _last_voice_call["ts"]
    if elapsed < VOICE_CALL_DELAY:
        time.sleep(VOICE_CALL_DELAY - elapsed)

    bucket = _voice_buckets.setdefault(thread_id, [])
    cutoff = time.time() - 3600
    bucket[:] = [t for t in bucket if t > cutoff]
    if len(bucket) >= VOICE_MAX_PER_HOUR:
        return False, (
            f"Voice limit reached ({VOICE_MAX_PER_HOUR}/hour). "
            "Try text chat or try again later."
        )

    bucket.append(time.time())
    _last_voice_call["ts"] = time.time()
    return True, ""


# --- Graph visualization ---


def _get_graph_image() -> str | None:
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        path = "graph.png"
        with open(path, "wb") as f:
            f.write(png_bytes)
        return path
    except Exception as exc:
        logger.warning(f"Could not render graph image: {exc}")
        return None


# --- Chat handler ---


def chat(message: str, history: list, thread_id: str):
    if not message or not message.strip():
        return history, thread_id, ""

    message = message.strip()
    if len(message) > MAX_INPUT_CHARS:
        message = message[:MAX_INPUT_CHARS]

    if not thread_id:
        thread_id = str(uuid.uuid4())

    _rate_limit()

    config = {"configurable": {"thread_id": thread_id}}
    history = history + [{"role": "user", "content": message}]

    try:
        result = graph.invoke(
            {
                "user_query": message,
                "messages": [HumanMessage(content=message)],
            },
            config=config,
        )
    except Exception as exc:
        logger.exception("Graph invocation failed")
        history.append(
            {"role": "assistant", "content": f"Error processing request: {exc}"}
        )
        return history, thread_id, ""

    tasks = result.get("tasks", []) or []
    logger.info(
        "chat routed to: %s",
        [getattr(t, "agent", "?") for t in tasks] or "direct",
    )

    # Defensive: Gemini can return content as a list of blocks.
    # src/nodes.py normalizes this, but coerce again here as a backstop.
    raw = result.get("final_answer") or ""
    if isinstance(raw, list):
        raw = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in raw
        )
    answer = str(raw).strip() or "The agents returned no answer. Try rephrasing."

    history.append({"role": "assistant", "content": answer})
    return history, thread_id, ""


def reset_conversation():
    return [], str(uuid.uuid4()), ""


# --- Voice handler ---


def _final_answer_to_string(result: dict) -> str:
    """Coerce final_answer to a plain string (mirrors chat() backstop)."""
    raw = result.get("final_answer") or ""
    if isinstance(raw, list):
        raw = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in raw
        )
    return str(raw).strip()


def voice_chat(audio_path: str | None, history: list, thread_id: str):
    """STT -> LangGraph -> TTS. Pushes the user transcript and agent answer
    into the shared chatbot history and autoplays the TTS output.

    Returns (history, thread_id, audio_out, mic_clear=None).
    """
    # Lazy import so a voice-only import error doesn't block app startup.
    from src.voice import transcribe, synthesize

    history = history or []

    if not thread_id:
        thread_id = str(uuid.uuid4())

    if not audio_path:
        history = history + [
            {"role": "assistant", "content": "No audio captured. Record, then stop — the mic sends on stop."}
        ]
        return history, thread_id, None, None

    ok, reason = _rate_limit_voice(thread_id)
    if not ok:
        history = history + [{"role": "assistant", "content": reason}]
        return history, thread_id, None, None

    try:
        with open(audio_path, "rb") as f:
            wav_bytes = f.read()
    except Exception as exc:
        logger.exception("Could not read mic capture")
        history = history + [
            {"role": "assistant", "content": f"Could not read audio: {exc}"}
        ]
        return history, thread_id, None, None

    # Enforce 5-second cap server-side. Gradio's filepath output is WAV by
    # default so stdlib wave handles it. If a different format sneaks in, we
    # fall back to a size heuristic rather than failing.
    try:
        import wave
        with wave.open(audio_path, "rb") as w:
            duration = w.getnframes() / float(w.getframerate())
        if duration > VOICE_MAX_SECONDS + 0.5:
            history = history + [
                {
                    "role": "assistant",
                    "content": f"Clip was {duration:.1f}s. Please keep it under {VOICE_MAX_SECONDS}s.",
                }
            ]
            return history, thread_id, None, None
    except Exception:
        # Rough fallback: 16kHz mono 16-bit ~= 32KB/s; 5s ~= 160KB.
        # Be generous (500KB) to avoid false positives on other encodings.
        if len(wav_bytes) > 500_000:
            history = history + [
                {
                    "role": "assistant",
                    "content": f"Audio too long. Please keep it under {VOICE_MAX_SECONDS}s.",
                }
            ]
            return history, thread_id, None, None

    # 1. STT
    try:
        transcript = transcribe(wav_bytes)
    except Exception as exc:
        history = history + [
            {"role": "assistant", "content": f"Voice transcription failed: {exc}. Try typing."}
        ]
        return history, thread_id, None, None
    if len(transcript) < 2:
        history = history + [
            {"role": "assistant", "content": "I didn't catch that. Please try again."}
        ]
        return history, thread_id, None, None

    # Render the user's voice turn in the chat history.
    history = history + [{"role": "user", "content": transcript}]

    # 2. LLM via existing graph (same thread_id so text + voice share memory).
    try:
        result = graph.invoke(
            {
                "user_query": transcript,
                "messages": [HumanMessage(content=transcript)],
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        logger.exception("Graph invocation failed (voice)")
        history = history + [
            {"role": "assistant", "content": f"Error processing request: {exc}"}
        ]
        return history, thread_id, None, None

    tasks = result.get("tasks", []) or []
    logger.info(
        "voice_chat routed to: %s",
        [getattr(t, "agent", "?") for t in tasks] or "direct",
    )
    answer = _final_answer_to_string(result)
    if not answer:
        history = history + [
            {"role": "assistant", "content": "The agents returned no answer. Try rephrasing."}
        ]
        return history, thread_id, None, None

    # 3. TTS (graceful fallback: text remains even if TTS fails).
    # Gemini preview TTS can 500 transiently. synthesize() retries; if it still
    # fails, append a brief note so the user sees why audio did not play and
    # knows a retry will probably succeed, rather than assuming voice is broken.
    tts_note = ""
    try:
        sr, audio = synthesize(answer)
        audio_out = (sr, audio)
    except Exception:
        logger.exception("TTS failed after retries")
        audio_out = None
        tts_note = (
            "\n\n_Voice playback is temporarily unavailable. "
            "Try the mic again in a moment._"
        )

    history = history + [{"role": "assistant", "content": answer + tts_note}]

    # Return mic_clear=None so the Audio component resets for the next recording.
    return history, thread_id, audio_out, None


# --- UI theme and styling ---

THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "SF Mono", "monospace"],
)

CUSTOM_CSS = """
footer { display: none !important; }
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 16px 32px !important;
}
@media (min-width: 1280px) {
    .gradio-container { padding: 16px 64px !important; }
}
@media (min-width: 1600px) {
    .gradio-container { padding: 16px 96px !important; }
}
#title-row h1 {
    background: linear-gradient(90deg, #059669 0%, #0d9488 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
/* Subtitle: use Gradio theme vars so it stays legible in light AND dark mode.
   --body-text-color-subdued is the theme's secondary-text token and flips
   automatically with the active theme. */
#subtitle {
    color: var(--body-text-color-subdued, #475569);
    font-size: 1rem;
    margin-top: 0;
    line-height: 1.55;
}
/* Defense in depth: hard-code safe colors per mode if the CSS var is ever
   unavailable (e.g. custom theme override). */
html:not(.dark) #subtitle { color: #334155; }
html.dark #subtitle { color: #cbd5e1; }
/* Disclaimer: visible but understated. Slightly muted, italic, smaller than
   the subtitle, with a thin left border so it reads as an advisory note
   rather than primary copy. */
#disclaimer {
    margin: 0.5rem 0 0 0;
    padding: 0.5rem 0.85rem;
    font-size: 0.88rem;
    font-style: italic;
    line-height: 1.5;
    border-left: 3px solid var(--border-color-accent, #94a3b8);
    color: var(--body-text-color-subdued, #475569);
}
html:not(.dark) #disclaimer { color: #475569; border-left-color: #94a3b8; }
html.dark #disclaimer { color: #94a3b8; border-left-color: #475569; }
#chatbox { border-radius: 14px; }
.gr-button-primary {
    background: linear-gradient(90deg, #059669 0%, #0d9488 100%) !important;
    border: none !important;
}

/* Mic recording pulse: target Gradio's active-recording state */
#voice-mic.recording,
#voice-mic:has(button[aria-label*="Stop" i]) {
    animation: mic-pulse 1.4s ease-in-out infinite;
    border-radius: 12px;
}
@keyframes mic-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.55); }
    50%      { box-shadow: 0 0 0 8px rgba(5, 150, 105, 0); }
}
#voice-mic { border-radius: 12px; }

/* Mic device selector (Speaker/Mic dropdowns) must not truncate labels */
#voice-mic select,
#voice-mic .device-select,
#voice-mic [data-testid*="device" i] select {
    min-width: 260px !important;
    max-width: 100% !important;
}

/* Hide share/duplicate/trash icons on Audio and Image components.
   Gradio 6 ships these as built-in actions; we don't want "Share" leaking
   out (it doesn't work offline) and the tiny trash/duplicate icons add noise. */
#voice-out button[aria-label*="Share" i],
#voice-out button[aria-label*="Duplicate" i],
#voice-out button[title*="Share" i],
#voice-out button[title*="Duplicate" i],
#voice-mic button[aria-label*="Share" i],
#voice-mic button[title*="Share" i],
#architecture-image button[aria-label*="Share" i],
#architecture-image button[title*="Share" i],
.gradio-container button[aria-label*="Share" i],
.gradio-container button[title*="Share" i] {
    display: none !important;
}

/* Architecture diagram: left align and scale gracefully in fullscreen */
#architecture-image { text-align: left !important; }
#architecture-image > div { justify-content: flex-start !important; }
#architecture-image img { max-width: 720px; height: auto; }
:fullscreen #architecture-image,
:fullscreen #architecture-image > div,
:fullscreen #architecture-image img,
.fullscreen #architecture-image img {
    max-width: 95vw !important;
    max-height: 90vh !important;
    width: auto !important;
    height: auto !important;
}
"""

HEADER_MD = """
<div id="title-row">

# Agentic Financial Assistant

</div>

<p id="subtitle">
A conversational, voice enabled multi agent system built on LangGraph. An
Orchestrator agent classifies each query and dispatches specialist agents in
parallel across live market data, live news, and a curated financial knowledge
base. A Synthesizer merges their outputs into one coherent answer grounded in
real world data.
</p>

<p id="disclaimer">
This application is an educational demo, not financial advice. Always do your
own research and consult a qualified professional before making any investment
decision.
</p>
"""

ARCHITECTURE_MD = """
### How it works

A single AI assistant cannot be great at everything. This system uses a team of
specialists. One lead agent reads your question, decides which specialists
should answer, and runs them at the same time. A final agent stitches the
answers together into one clean reply.

### The team

- **Market Agent** pulls live stock prices, valuations, and trading volume from yfinance.
- **Research Agent** searches the live web for news, earnings, and analyst commentary via Tavily.
- **Advisory Agent** answers educational questions using a curated knowledge base of investing and tax concepts, retrieved with embeddings from a vector database (ChromaDB).

### How a question flows

1. The **Orchestrator** reads the question and returns a structured plan (which agents, what to ask each).
2. Selected agents run **in parallel**, each with its own tools.
3. The **Synthesizer** merges the outputs into one answer.
4. State is checkpointed per conversation thread, so follow-ups have memory.

### Routing in practice

- "How is AAPL performing today?" → Market only
- "Latest news on Amazon and AWS?" → Research only
- "What is a Roth IRA?" → Advisory only
- "How is AAPL performing and what are analysts saying?" → Market and Research in parallel

### Stack

LangGraph (graph and state), Google Gemini 2.5 Flash (reasoning), Google
embeddings and ChromaDB (retrieval), yfinance (markets), Tavily (web search).
Voice uses Gemini 2.5 Flash for speech to text and gemini-2.5-flash-preview-tts
(Kore voice) for text to speech.

The architecture is **model agnostic**. The LLM and embedding layers are
swappable, so the same system can run on OpenAI, Anthropic, or AWS Bedrock
with a config change.
"""

ABOUT_MD = """
### What this is

A production style multi agent assistant for financial questions.
Ask about a stock, today's market news, or core investing concepts, and the
right specialist answers. Works in both text and voice.

### Why it matters

- **One interface, many capabilities.** Live market data, live news, and
  curated financial education in a single conversation.
- **Faster answers.** When a question needs two perspectives, the agents run
  in parallel instead of one after the other.
- **Grounded in real data.** Every number and headline comes from a live API
  or a curated knowledge base. No synthetic content.
- **Portable.** The design is model agnostic, so it can be deployed on any
  major LLM provider without rewriting the application. The LLM layer can
  swap to OpenAI, Anthropic, or Amazon Bedrock; the voice layer can swap to
  Amazon Transcribe and Polly, or OpenAI Whisper and TTS.

### What it demonstrates, technically

- Multi agent orchestration on **LangGraph**, with a classifier that dispatches
  specialists and a synthesizer that merges their outputs.
- **Parallel agent execution**, tool calling, and structured output with
  Pydantic for deterministic routing.
- **RAG** over a curated financial knowledge base using Google embeddings and
  ChromaDB.
- **Conversation memory** via checkpointing per thread.
- **Voice** with Gemini 2.5 Flash for STT and TTS, integrated into the same
  chat surface.

### Data sources

- **Live stock data**: yfinance
- **Live web news**: Tavily
- **Financial education corpus**: 16 original entries written for this project

### Disclaimer

This application is an educational demo, not financial advice. The Advisory
Agent also appends a brief disclaimer to every response that touches on
personal financial decisions. Always do your own research and consult a
qualified professional before making any investment decision.
"""

EXAMPLES = [
    "How is AMZN performing today?",
    "How is NVDA doing today?",
    "What is a Roth IRA and how is it different from a Traditional IRA?",
    "What is the latest news on Amazon and AWS?",
    "How is AAPL performing and what are analysts saying about it?",
    "Explain dollar cost averaging with an example",
    "Compare the market cap of AMZN, AAPL, and MSFT",
]

graph_image = _get_graph_image()

with gr.Blocks(title="Agentic Financial Assistant") as app:
    gr.Markdown(HEADER_MD)

    thread_state = gr.State(value=str(uuid.uuid4()))

    with gr.Tabs():
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                label="Conversation",
                elem_id="chatbox",
                height=520,
                placeholder=(
                    "Ask in text or voice. Click the mic to speak (up to 5 seconds)."
                ),
                layout="bubble",
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Type a question, or use the mic below to speak...",
                    lines=2,
                    scale=5,
                    max_length=MAX_INPUT_CHARS,
                )
                with gr.Column(scale=1, min_width=130):
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("New chat", variant="secondary")

            with gr.Row():
                voice_mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label=f"Voice (up to {VOICE_MAX_SECONDS}s — click record, speak, then stop to send)",
                    scale=3,
                    elem_id="voice-mic",
                )
                voice_audio = gr.Audio(
                    label="Spoken answer",
                    autoplay=True,
                    type="numpy",
                    interactive=False,
                    scale=2,
                    elem_id="voice-out",
                )

            gr.Examples(
                examples=EXAMPLES,
                inputs=msg,
                label="Example questions",
            )

        with gr.Tab("Architecture"):
            if graph_image:
                gr.Image(
                    value=graph_image,
                    show_label=False,
                    container=False,
                    elem_id="architecture-image",
                )
            else:
                gr.Markdown("_Graph visualization unavailable in this environment._")
            gr.Markdown(ARCHITECTURE_MD)

        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

    submit_btn.click(
        fn=chat,
        inputs=[msg, chatbot, thread_state],
        outputs=[chatbot, thread_state, msg],
    )
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, thread_state],
        outputs=[chatbot, thread_state, msg],
    )
    clear_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[chatbot, thread_state, msg],
    )

    # Auto-send when the user stops recording. Fixes the "Send voice" race
    # where users clicked send before stopping the mic (empty audio_path).
    voice_mic.stop_recording(
        fn=voice_chat,
        inputs=[voice_mic, chatbot, thread_state],
        outputs=[chatbot, thread_state, voice_audio, voice_mic],
    )


if __name__ == "__main__":
    app.launch(theme=THEME, css=CUSTOM_CSS)
