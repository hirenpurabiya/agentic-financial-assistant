"""
CLI entry point for the Agentic Financial Assistant.

Usage:
    python -m src.main                    # interactive REPL
    python -m src.main --query "..."      # single query mode
"""

from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import HumanMessage

from .config import logger
from .graph import graph


class FinancialAssistant:
    """Thin wrapper around the compiled LangGraph graph."""

    def __init__(self, thread_id: str | None = None):
        self.thread_id = thread_id or str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}

    def ask(self, query: str) -> str:
        result = graph.invoke(
            {
                "user_query": query,
                "messages": [HumanMessage(content=query)],
            },
            config=self.config,
        )
        return result.get("final_answer", "")


def _repl() -> None:
    assistant = FinancialAssistant()
    print("Agentic Financial Assistant. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        try:
            answer = assistant.ask(query)
        except Exception as exc:
            logger.exception("Query failed")
            print(f"Error: {exc}\n")
            continue
        print(f"\nAssistant: {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Financial Assistant")
    parser.add_argument("--query", help="Run a single query and exit.")
    args = parser.parse_args()

    if args.query:
        assistant = FinancialAssistant()
        print(assistant.ask(args.query))
    else:
        _repl()


if __name__ == "__main__":
    main()
