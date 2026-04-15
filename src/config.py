"""
Configuration module for Agentic Financial Assistant.

Loads environment variables, validates API keys, and initializes
the LLM, embeddings, and logger. All other modules import from here.
"""

import logging
import os
import sys

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()


# --- Logger setup ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("financial-assistant")


# --- API key validation ---

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("your_"):
    logger.error("Missing GOOGLE_API_KEY. Copy .env.example to .env and add your key.")
    sys.exit(1)

if not TAVILY_API_KEY or TAVILY_API_KEY.startswith("your_"):
    logger.error("Missing TAVILY_API_KEY. Copy .env.example to .env and add your key.")
    sys.exit(1)


# --- Model initialization ---

LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY,
    max_retries=3,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)

logger.info(f"Initialized LLM: {LLM_MODEL}")
logger.info(f"Initialized embeddings: {EMBEDDING_MODEL}")
