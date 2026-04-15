"""
LangGraph wiring for the Agentic Financial Assistant.

Orchestrator classifies the query, Send() fans out to 0 or more agents in
parallel, all agents converge on the Synthesizer, MemorySaver keeps
conversation state per thread id.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    advisory_agent,
    market_agent,
    orchestrator,
    research_agent,
    route_to_agents,
    synthesizer,
)
from .state import FinancialState


def build_graph():
    """Compile the financial assistant graph with an in memory checkpointer."""
    builder = StateGraph(FinancialState)

    builder.add_node("orchestrator", orchestrator)
    builder.add_node("market_agent", market_agent)
    builder.add_node("research_agent", research_agent)
    builder.add_node("advisory_agent", advisory_agent)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_to_agents,
        ["market_agent", "research_agent", "advisory_agent", "synthesizer"],
    )
    builder.add_edge("market_agent", "synthesizer")
    builder.add_edge("research_agent", "synthesizer")
    builder.add_edge("advisory_agent", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
