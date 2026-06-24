"""
chatbot_interface.py
--------------------
Abstract contract that every chatbot must implement.
The API server only knows about this interface — never about specific bots.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class ChatbotInterface(ABC):
    """
    Every chatbot plugged into this API must implement these two methods.
    The server calls them; it never cares what is underneath.
    """

    @abstractmethod
    def create_initial_state(self) -> Dict[str, Any]:
        """
        Return a fresh, empty conversation state for a new session.
        Each bot defines its own state schema here.
        """
        ...

    @abstractmethod
    def chat(self, state: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """
        Process one user turn.

        Args:
            state:        The current session state (returned from the previous call).
            user_message: The raw text the user typed.

        Returns:
            A dict with at least these keys so the server can build a uniform response:
              - state   : Dict  — the updated session state to persist
              - reply   : str   — the bot's text reply to show the user
              - tokens  : dict  — {"prompt_tokens": int, "completion_tokens": int}
              - chunks  : int   — number of context chunks retrieved (0 if none)
              - agent   : str   — name of the active node/agent for diagnostics
        """
        ...
