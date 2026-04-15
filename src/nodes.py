"""
Graph nodes: Orchestrator, three specialized agents, and the Synthesizer.

The Orchestrator uses structured output to classify the query and dispatches
to agents in parallel via Send(). Each agent runs its own model plus tools
loop. The Synthesizer merges parallel outputs into a single coherent answer.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Send

from .config import llm, logger
from .state import AgentTask, ClassificationResult, FinancialState
from .tools import get_market_data, search_financial_news, search_knowledge_base


# --- Content normalization ---


def _content_to_text(content) -> str:
    """Normalize AIMessage.content (str or list of blocks) to a plain string.

    Newer Gemini / langchain-google-genai versions can return content as a list
    of blocks like [{"type": "text", "text": "..."}]. Downstream code expects
    a string, so collapse blocks to plain text here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


# --- Orchestrator ---

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator of a multi agent financial assistant.

You route user queries to specialized agents. There are three agents:

1. market: answers questions about specific stock prices, valuations, market cap, \
P/E ratios, volume, and recent trading activity. Has access to live market data via yfinance.

2. research: answers questions requiring current news, earnings reports, macro \
events, analyst commentary, or any real time financial information. Has access to live web search.

3. advisory: answers conceptual and educational questions about investing, \
retirement accounts, tax treatment, portfolio strategy, and financial planning. \
Has access to a curated financial knowledge base.

Rules:
- If the query involves multiple intents, dispatch to multiple agents in parallel.
- If the query is a simple greeting, thanks, or clearly outside finance scope, \
return an empty tasks list and put your direct answer in direct_response.
- Each task description must be self contained (the agent will not see conversation history).
- Prefer the minimal set of agents that can answer the query.
"""


def orchestrator(state: FinancialState) -> dict:
    """Classify the user query and produce a list of AgentTasks."""
    user_query = state.get("user_query", "")
    history = state.get("messages", [])

    classifier = llm.with_structured_output(ClassificationResult)
    context_messages = [SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)]
    context_messages.extend(history[-6:])
    context_messages.append(HumanMessage(content=user_query))

    result: ClassificationResult = classifier.invoke(context_messages)
    logger.info(
        f"Orchestrator routed to {[t.agent for t in result.tasks] or 'direct'}: "
        f"{result.reasoning}"
    )

    tasks = result.tasks
    requires_synthesis = len(tasks) > 1

    # Reset final_answer every turn. MemorySaver persists state across turns,
    # so without this the synthesizer's `if state.get("final_answer")` check
    # would short-circuit to the previous turn's answer instead of re-running.
    update: dict = {
        "tasks": tasks,
        "requires_synthesis": requires_synthesis,
        "agent_results": {},
        "final_answer": "",
    }
    if not tasks:
        update["final_answer"] = result.direct_response or (
            "I am a financial assistant. Ask me about stocks, market news, "
            "or investing concepts."
        )
    return update


def route_to_agents(state: FinancialState):
    """Fan out to the agents selected by the Orchestrator."""
    tasks = state.get("tasks", [])
    if not tasks:
        return "synthesizer"
    return [
        Send(f"{task.agent}_agent", {"user_query": task.task})
        for task in tasks
    ]


# --- Agent subgraph runner ---


def _run_agent_loop(
    agent_name: str,
    system_prompt: str,
    tools: list,
    task: str,
    max_iters: int = 4,
) -> str:
    """Run a model plus tools loop for one agent and return its final text."""
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task),
    ]

    for _ in range(max_iters):
        ai_msg: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return _content_to_text(ai_msg.content)

        for call in ai_msg.tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_fn = tool_map.get(name)
            if tool_fn is None:
                output = f"Tool {name} not available."
            else:
                try:
                    output = tool_fn.invoke(args)
                except Exception as exc:
                    logger.exception(f"{agent_name} tool {name} failed")
                    output = f"Tool error: {exc}"
            messages.append(
                ToolMessage(content=str(output), tool_call_id=call["id"])
            )

    final = llm_with_tools.invoke(messages)
    return _content_to_text(final.content)


# --- Market Agent ---

MARKET_AGENT_PROMPT = """You are the Market Agent. You answer questions about \
specific stocks using live data from yfinance.

- Always call get_market_data for the tickers in the question before answering.
- If the user names a company without a ticker, infer the ticker (Apple -> AAPL, \
Tesla -> TSLA, Nvidia -> NVDA, Microsoft -> MSFT, Alphabet -> GOOGL, Amazon -> AMZN, \
Meta -> META).
- Present numbers clearly. Call out notable moves or valuations.
- Be concise. Two to four short paragraphs max.
"""


def market_agent(state: dict) -> dict:
    task = state["user_query"]
    logger.info(f"Market Agent running: {task}")
    answer = _run_agent_loop(
        agent_name="market",
        system_prompt=MARKET_AGENT_PROMPT,
        tools=[get_market_data],
        task=task,
    )
    return {"agent_results": {"market": answer}}


# --- Research Agent ---

RESEARCH_AGENT_PROMPT = """You are the Research Agent. You answer questions \
about current financial news, earnings, and macro events using live web search.

- Always call search_financial_news for the topic in the question before answering.
- Cite the most relevant sources inline like (Source: <url>) when making specific claims.
- Summarize the key points. Do not copy long passages from the sources.
- Be concise. Two to four short paragraphs max.
"""


def research_agent(state: dict) -> dict:
    task = state["user_query"]
    logger.info(f"Research Agent running: {task}")
    answer = _run_agent_loop(
        agent_name="research",
        system_prompt=RESEARCH_AGENT_PROMPT,
        tools=[search_financial_news],
        task=task,
    )
    return {"agent_results": {"research": answer}}


# --- Advisory Agent ---

ADVISORY_AGENT_PROMPT = """You are the Advisory Agent. You answer educational \
questions about investing, retirement accounts, taxes, and portfolio strategy.

- Always call search_knowledge_base first to ground your answer in the curated corpus.
- Explain concepts clearly with concrete examples.
- Add a brief disclaimer when the question touches on personal financial decisions: \
"This is educational information, not personalized financial advice."
- Be concise. Two to four short paragraphs max.
"""


def advisory_agent(state: dict) -> dict:
    task = state["user_query"]
    logger.info(f"Advisory Agent running: {task}")
    answer = _run_agent_loop(
        agent_name="advisory",
        system_prompt=ADVISORY_AGENT_PROMPT,
        tools=[search_knowledge_base],
        task=task,
    )
    return {"agent_results": {"advisory": answer}}


# --- Synthesizer ---

SYNTHESIZER_PROMPT = """You are the Synthesizer. You merge outputs from multiple \
specialized agents into one coherent answer for the user.

- Combine the agent outputs into a single response with natural flow.
- Do not label sections with agent names.
- Preserve all specific numbers, facts, and citations from the agents.
- Be concise. Match the shortest reasonable length that covers everything.
"""


def synthesizer(state: FinancialState) -> dict:
    results = state.get("agent_results", {}) or {}
    user_query = state.get("user_query", "")

    if state.get("final_answer"):
        final = state["final_answer"]
    elif not results:
        final = "I could not process that query. Please try rephrasing."
    elif len(results) == 1:
        final = next(iter(results.values()))
    else:
        parts = "\n\n".join(
            f"[{name} agent]\n{_content_to_text(content)}"
            for name, content in results.items()
        )
        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(
                    content=f"User query: {user_query}\n\nAgent outputs:\n{parts}"
                ),
            ]
        )
        final = _content_to_text(response.content) or "\n\n".join(
            _content_to_text(v) for v in results.values()
        )

    # Belt and suspenders: guarantee final is a plain string downstream.
    final = _content_to_text(final) if not isinstance(final, str) else final

    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
    }
