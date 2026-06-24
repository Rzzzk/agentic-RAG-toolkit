"""
pdf_rag_bot.py
--------------
Concrete chatbot implementation: a PDF-RAG bot powered by Gemini + ChromaDB.
This mirrors tests/chat_bot.py but wraps it behind the ChatbotInterface contract
so the API server stays completely ignorant of LangGraph internals.
"""
import os
import sys
import yaml
from typing import Any, Dict
from langchain_core.messages import HumanMessage

# Allow imports from the project root (src.*)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.flows.rag_graph import build_rag_graph
from api.chatbot_interface import ChatbotInterface


class PdfRagBot(ChatbotInterface):
    """
    A Retrieval-Augmented Generation chatbot that answers questions
    about a PDF corpus stored in ChromaDB, using Gemini as the LLM.

    Config is loaded from a YAML file (same format as tests/chat_bot.py).
    The compiled LangGraph engine is cached on first use.
    """

    def __init__(self, config_path: str = "configs/free_pdf_bot.yaml"):
        self._config_path = config_path
        self._engine = None  # Lazy-loaded on first chat call

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_engine(self):
        """Compile the LangGraph once and cache it."""
        if self._engine is None:
            print(f"[PdfRagBot] Compiling graph from: {self._config_path}")
            with open(self._config_path, "r") as f:
                config = yaml.safe_load(f)
            self._engine = build_rag_graph(config)
            print("[PdfRagBot] Graph compiled successfully.")
        return self._engine

    # ------------------------------------------------------------------
    # ChatbotInterface implementation
    # ------------------------------------------------------------------

    def create_initial_state(self) -> Dict[str, Any]:
        """Fresh state for every new user session."""
        return {
            "messages": [],
            "retrieved_context": [],
            "token_usage": {},
            "embedding_details": {},
            "current_agent": "system",
            "errors": [],
        }

    def chat(self, state: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """
        Append the user message to state, invoke the graph, and return
        a uniform response dict that the API server can serialise directly.
        """
        engine = self._get_engine()

        # Append user turn — same pattern as tests/chat_bot.py
        state["messages"].append(HumanMessage(content=user_message))

        # Run the LangGraph engine (blocking — caller wraps in asyncio.to_thread)
        result_state = engine.invoke(state)

        # Extract the last AI message
        reply = result_state["messages"][-1].content
        token_usage = result_state.get("token_usage", {})

        return {
            "state": result_state,
            "reply": reply,
            "tokens": {
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
            },
            "chunks": len(result_state.get("retrieved_context", [])),
            "agent": result_state.get("current_agent", "generator"),
        }
