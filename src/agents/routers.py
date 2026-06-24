from typing import Literal
from src.core.state import AgentState

def route_question(state: AgentState) -> Literal["retrieve", "generate"]:
    """
    A conditional edge that decides the next step in the graph.
    Returns the string name of the node that LangGraph should route to next.
    """
    latest_message = state["messages"][-1].content.lower()
    
    # Simple heuristic routing:
    # If the user is just saying hello, skip the database search to save tokens and time.
    greetings = ["hi", "hello", "hey", "who are you"]
    if any(latest_message.startswith(greet) for greet in greetings):
        return "generate"
    
    # For all other questions, assume we need to search our knowledge base first
    return "retrieve"