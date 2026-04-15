"""
Tools for the three specialized agents.

- get_market_data: real stock data via yfinance (Market Agent)
- search_financial_news: live web search via Tavily (Research Agent)
- search_knowledge_base: semantic search over our RAG corpus (Advisory Agent)
"""

from __future__ import annotations

import yfinance as yf
from langchain_core.tools import tool
from tavily import TavilyClient

from .config import TAVILY_API_KEY, logger
from .rag import knowledge_vectorstore


_tavily = TavilyClient(api_key=TAVILY_API_KEY)


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.2f}"


@tool
def get_market_data(symbol: str) -> str:
    """Fetch live market data for a stock ticker.

    Returns current price, day change, volume, market cap, P/E ratio, and
    52 week range. Use this for any question about a specific stock's price,
    valuation, or recent trading activity. Symbol should be the ticker like
    AAPL, MSFT, TSLA, NVDA.
    """
    symbol = symbol.strip().upper()
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        if price is None:
            return f"Could not fetch live data for {symbol}. Verify the ticker is valid."

        prev_close = info.get("previousClose") or price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        name = info.get("longName") or info.get("shortName") or symbol
        sector = info.get("sector") or "n/a"
        market_cap = _format_number(info.get("marketCap"))
        pe_ratio = info.get("trailingPE")
        pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "n/a"
        volume = info.get("volume") or info.get("regularMarketVolume")
        volume_str = f"{volume:,}" if isinstance(volume, (int, float)) else "n/a"
        week_high = info.get("fiftyTwoWeekHigh")
        week_low = info.get("fiftyTwoWeekLow")
        range_str = (
            f"${week_low:.2f} to ${week_high:.2f}"
            if week_low is not None and week_high is not None
            else "n/a"
        )

        return (
            f"{name} ({symbol})\n"
            f"Sector: {sector}\n"
            f"Price: ${price:,.2f} ({change:+.2f}, {change_pct:+.2f}%)\n"
            f"Market Cap: {market_cap}\n"
            f"P/E Ratio: {pe_str}\n"
            f"Volume: {volume_str}\n"
            f"52 Week Range: {range_str}"
        )
    except Exception as exc:
        logger.exception(f"yfinance lookup failed for {symbol}")
        return f"Error fetching market data for {symbol}: {exc}"


@tool
def search_financial_news(query: str) -> str:
    """Search the live web for recent financial news and analysis.

    Use for questions about market events, company news, earnings, macro
    commentary, or anything that requires current information beyond static
    knowledge. Returns the top results with titles, snippets, and source URLs.
    """
    try:
        result = _tavily.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )
        lines: list[str] = []
        if result.get("answer"):
            lines.append(f"Summary: {result['answer']}\n")
        for i, item in enumerate(result.get("results", []), 1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            snippet = (item.get("content") or "").strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."
            lines.append(f"[{i}] {title}\n{snippet}\nSource: {url}")
        return "\n\n".join(lines) if lines else "No results found."
    except Exception as exc:
        logger.exception("Tavily search failed")
        return f"Error running web search: {exc}"


@tool
def search_knowledge_base(query: str) -> str:
    """Semantic search over our curated financial education knowledge base.

    Use for questions about investment concepts, retirement accounts, tax
    treatment, portfolio strategy, and other educational topics. Returns the
    most relevant entries from the knowledge base.
    """
    try:
        docs = knowledge_vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No relevant entries found in the knowledge base."
        return "\n\n---\n\n".join(
            f"Title: {d.metadata.get('title', 'Untitled')}\n"
            f"Category: {d.metadata.get('category', 'general')}\n\n"
            f"{d.page_content}"
            for d in docs
        )
    except Exception as exc:
        logger.exception("Knowledge base search failed")
        return f"Error searching knowledge base: {exc}"
