"""
State definitions for the Agentic Financial Assistant graph.

FinancialState flows through every node in the LangGraph StateGraph.
Pydantic models give us structured output for deterministic classification
and task dispatch from the Orchestrator.
"""

from typing import Annotated, Literal
from operator import add

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --- Pydantic models for structured output ---


AgentName = Literal["market", "research", "advisory"]


class AgentTask(BaseModel):
    """A single task the Orchestrator dispatches to one specialized agent."""

    agent: AgentName = Field(
        description="Which specialized agent should handle this task."
    )
    task: str = Field(
        description="Self contained task description for the agent to act on."
    )


class ClassificationResult(BaseModel):
    """Orchestrator output. Lists zero or more agents to run in parallel."""

    tasks: list[AgentTask] = Field(
        default_factory=list,
        description=(
            "Tasks to dispatch. Empty list means the Orchestrator will answer "
            "the query directly without any specialized agent."
        ),
    )
    reasoning: str = Field(
        description="One sentence on why this routing was chosen."
    )
    direct_response: str = Field(
        default="",
        description=(
            "If tasks is empty, this is the Orchestrator's direct answer "
            "(for greetings, chit chat, or queries outside finance scope)."
        ),
    )


# --- Graph state ---


def merge_agent_results(
    left: dict[str, str] | None, right: dict[str, str] | None
) -> dict[str, str]:
    """Reducer that merges parallel agent outputs into a single dict."""
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


class FinancialState(TypedDict, total=False):
    """State that flows through the Orchestrator, agents, and Synthesizer."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    tasks: list[AgentTask]
    requires_synthesis: bool
    agent_results: Annotated[dict[str, str], merge_agent_results]
    final_answer: str
