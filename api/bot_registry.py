"""
bot_registry.py
---------------
Central registry that maps bot IDs (strings) to ChatbotInterface instances.
The server only talks to this module — it never imports a concrete bot directly.

To add a new chatbot:
  1. Create a new class implementing ChatbotInterface (e.g., api/my_new_bot.py).
  2. Register it here with a unique ID string.
  3. Add that ID to the UI's AVAILABLE_BOTS list in ui/ui.py (or via env var).
"""
from typing import Dict
from api.chatbot_interface import ChatbotInterface
from api.pdf_rag_bot import PdfRagBot

# ---------------------------------------------------------------------------
# Bot catalogue — add entries here to expose more bots through the API
# ---------------------------------------------------------------------------
_BOT_CATALOGUE: Dict[str, ChatbotInterface] = {
    "free_pdf_bot": PdfRagBot(config_path="configs/free_pdf_bot.yaml"),
    # "my_other_bot": MyOtherBot(config_path="configs/my_other_bot.yaml"),
}

_DEFAULT_BOT_ID = "free_pdf_bot"


def get_bot(bot_id: str) -> ChatbotInterface:
    """
    Return the registered bot for the given ID.
    Falls back to the default bot and logs a warning if the ID is unknown.
    """
    if bot_id not in _BOT_CATALOGUE:
        print(
            f"[Registry] WARNING: bot_id '{bot_id}' not found. "
            f"Falling back to '{_DEFAULT_BOT_ID}'."
        )
        bot_id = _DEFAULT_BOT_ID
    return _BOT_CATALOGUE[bot_id]


def available_bot_ids() -> list[str]:
    """Return all registered bot IDs (useful for health/info endpoints)."""
    return list(_BOT_CATALOGUE.keys())
