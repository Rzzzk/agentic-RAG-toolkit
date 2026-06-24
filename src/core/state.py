from typing import Annotated, TypedDict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The central state payload for the Agentic Toolkit.
    Every node in your LangGraph will read from and write to this dictionary.
    """
    # The conversation history. 'add_messages' ensures new messages are appended, not overwritten.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # RAG payload: Documents retrieved from whichever Vector DB you configure
    retrieved_context: list[str]
    
    # Granular metrics for downstream UI visibility
    token_usage: dict[str, int]         # e.g., {"prompt_tokens": 120, "completion_tokens": 45}
    embedding_details: dict[str, Any]   # To expose vector shapes, distances, or raw embeddings
    
    # Graph routing metadata
    current_agent: str                  # Tracks which agent/node currently holds the baton
    errors: list[str]                   # Graceful error handling across nodes