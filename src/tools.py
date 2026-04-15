"""
Tools for the three specialized agents.

- get_market_data: real stock data via yfinance (Market Agent)
- search_financial_news: live web search via Tavily (Research Agent)
- search_knowledge_base: semantic search over our RAG corpus (Advisory Agent)
"""

from __future__ import annotations

import time

import yfinance as yf
from langchain_core.tools import tool
from tavily import TavilyClient

from .config import TAVILY_API_KEY, logger
from .rag import knowledge_vectorstore


_tavily = TavilyClient(api_key=TAVILY_API_KEY)


# --- yfinance hardening ---
#
# Yahoo Finance rate-limits aggressively, especially from shared IP pools like
# Hugging Face Spaces. Two defenses:
#   1. curl_cffi session that impersonates a real Chrome browser. Default
#      python-requests user-agents are throttled hard; Chrome TLS fingerprint
#      is not. yfinance >= 0.2.50 accepts a session via the session= kwarg.
#   2. A small in-memory TTL cache so back-to-back questions about the same
#      ticker do not re-hit Yahoo at all.
#
# If curl_cffi is not installed (local dev without the extra), fall back to a
# plain session. yfinance still works, it just gets rate-limited faster.

try:
    from curl_cffi import requests as curl_requests

    _yf_session = curl_requests.Session(impersonate="chrome")
except Exception:  # pragma: no cover
    logger.warning("curl_cffi unavailable; yfinance falling back to default session")
    _yf_session = None


_MARKET_CACHE_TTL_SECONDS = 300  # 5 minutes is plenty for conversational use
_market_cache: dict[str, tuple[float, str]] = {}


def _cache_get(symbol: str) -> str | None:
    entry = _market_cache.get(symbol)
    if entry is None:
        return None
    ts, value = entry
    if (time.time() - ts) > _MARKET_CACHE_TTL_SECONDS:
        _market_cache.pop(symbol, None)
        return None
    return value


def _cache_set(symbol: str, value: str) -> None:
    _market_cache[symbol] = (time.time(), value)


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


def _ticker(symbol: str) -> yf.Ticker:
    """Build a yfinance Ticker with our impersonating session, if available."""
    if _yf_session is not None:
        return yf.Ticker(symbol, session=_yf_session)
    return yf.Ticker(symbol)


def _fetch_market_data(symbol: str) -> str:
    """Core lookup with retry-on-rate-limit. Raises on unrecoverable failure."""
    last_err: Exception | None = None
    # Try up to 3 times with gentle backoff. Yahoo's rate limits usually clear
    # within a couple of seconds once impersonation is in play.
    for attempt in range(3):
        try:
            ticker = _ticker(symbol)

            # fast_info is a lighter endpoint than .info and survives throttling
            # better. Use it for price/volume/52w range, fall back to .info for
            # the richer metadata (sector, market cap, P/E, company name).
            fast = getattr(ticker, "fast_info", None)

            price = None
            prev_close = None
            volume = None
            week_high = None
            week_low = None
            if fast is not None:
                # fast_info fields vary by yfinance version; guard every access.
                price = getattr(fast, "last_price", None) or getattr(fast, "lastPrice", None)
                prev_close = getattr(fast, "previous_close", None) or getattr(fast, "previousClose", None)
                volume = getattr(fast, "last_volume", None) or getattr(fast, "lastVolume", None)
                week_high = getattr(fast, "year_high", None) or getattr(fast, "yearHigh", None)
                week_low = getattr(fast, "year_low", None) or getattr(fast, "yearLow", None)

            info: dict = {}
            try:
                info = ticker.info or {}
            except Exception as info_exc:
                # .info is the most throttled call; keep going with fast_info only.
                logger.warning(f"yfinance .info unavailable for {symbol}: {info_exc}")

            price = price or info.get("currentPrice") or info.get("regularMarketPrice")
            if price is None:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])

            if price is None:
                raise RuntimeError("no price data returned")

            prev_close = prev_close or info.get("previousClose") or price
            change = price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0.0

            name = info.get("longName") or info.get("shortName") or symbol
            sector = info.get("sector") or "n/a"
            market_cap = _format_number(info.get("marketCap"))
            pe_ratio = info.get("trailingPE")
            pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "n/a"
            volume = volume or info.get("volume") or info.get("regularMarketVolume")
            volume_str = f"{int(volume):,}" if isinstance(volume, (int, float)) else "n/a"
            week_high = week_high or info.get("fiftyTwoWeekHigh")
            week_low = week_low or info.get("fiftyTwoWeekLow")
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
            last_err = exc
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg or "too many" in msg:
                # Exponential backoff: 1s, 2s, 4s.
                time.sleep(2 ** attempt)
                continue
            # Non-rate-limit error: do not retry.
            break

    raise RuntimeError(f"yfinance failed for {symbol}: {last_err}")


@tool
def get_market_data(symbol: str) -> str:
    """Fetch live market data for a stock ticker.

    Returns current price, day change, volume, market cap, P/E ratio, and
    52 week range. Use this for any question about a specific stock's price,
    valuation, or recent trading activity. Symbol should be the ticker like
    AAPL, MSFT, TSLA, NVDA.
    """
    symbol = symbol.strip().upper()

    cached = _cache_get(symbol)
    if cached is not None:
        logger.info(f"market cache hit for {symbol}")
        return cached

    try:
        result = _fetch_market_data(symbol)
        _cache_set(symbol, result)
        return result
    except Exception as exc:
        logger.exception(f"yfinance lookup failed for {symbol}")
        return (
            f"Live market data for {symbol} is temporarily unavailable "
            f"(upstream rate limit or transient error). Try again in a minute, "
            f"or ask about a different ticker. Detail: {exc}"
        )


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
