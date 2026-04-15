"""
Generate a polished architecture diagram for the README and portfolio.

Writes two images:
  - graph.png          auto-generated from LangGraph (used in Gradio UI)
  - graph_annotated.png annotated mermaid with data sources per agent
"""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import quote

import httpx

from src.graph import graph


REPO_ROOT = Path(__file__).resolve().parent.parent


ANNOTATED_MERMAID = """flowchart TD
    user([User Query]) --> orch
    orch{{Orchestrator<br/>Gemini structured output}}
    orch -. Send() .-> market
    orch -. Send() .-> research
    orch -. Send() .-> advisory
    market[Market Agent<br/>yfinance live prices]
    research[Research Agent<br/>Tavily live news]
    advisory[Advisory Agent<br/>ChromaDB RAG]
    market --> synth
    research --> synth
    advisory --> synth
    synth{{Synthesizer<br/>merge parallel outputs}}
    synth --> answer([Final Answer])
    classDef orch fill:#dcfce7,stroke:#059669,stroke-width:2px,color:#064e3b
    classDef agent fill:#ecfdf5,stroke:#10b981,color:#065f46
    classDef synth fill:#ccfbf1,stroke:#0d9488,stroke-width:2px,color:#134e4a
    classDef io fill:#f1f5f9,stroke:#475569,color:#0f172a
    class orch orch
    class market,research,advisory agent
    class synth synth
    class user,answer io
"""


def write_langgraph_png() -> None:
    png = graph.get_graph().draw_mermaid_png()
    (REPO_ROOT / "graph.png").write_bytes(png)
    print("wrote graph.png from LangGraph")


def write_annotated_png() -> None:
    """Render the annotated mermaid via mermaid.ink."""
    encoded = base64.urlsafe_b64encode(ANNOTATED_MERMAID.encode()).decode()
    url = f"https://mermaid.ink/img/{encoded}?type=png"
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    (REPO_ROOT / "graph_annotated.png").write_bytes(resp.content)
    print("wrote graph_annotated.png from mermaid.ink")


if __name__ == "__main__":
    write_langgraph_png()
    try:
        write_annotated_png()
    except Exception as exc:
        print(f"annotated graph skipped: {exc}")
