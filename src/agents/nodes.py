from typing import Any, Dict
from src.core.state import AgentState

def retrieve_node(state: AgentState, retriever: Any) -> Dict[str, Any]:
    """
    Node responsible for fetching context from the vector database.
    """
    # Grab the text of the latest user message
    latest_message = state["messages"][-1].content
    
    # Perform the search using the injected retriever
    # This returns our rich dictionary: text, distance, and embeddings
    results = retriever.search(query=latest_message, top_k=3)
    
    # Extract just the text chunks for the LLM prompt
    context_texts = [res["content"] for res in results]
    
    # Return the dictionary that will UPDATE the AgentState
    return {
        "retrieved_context": context_texts,
        # Pushing the deep metadata into the state for your frontend UI to consume
        "embedding_details": {"latest_search": results},
        "current_agent": "retriever"
    }

def generate_node(state: AgentState, llm: Any) -> Dict[str, Any]:
    """
    Node responsible for generating the final response using the LLM.
    """
    messages = state.get("messages", [])
    context = state.get("retrieved_context", [])
    
    # If we have retrieved context, we format a RAG system prompt
    if context:
        sys_prompt = (
            "You are a helpful AI assistant. Use the following context to answer the user's question:\n\n" 
            + "\n---\n".join(context)
        )
        formatted_messages = [{"role": "system", "content": sys_prompt}] + messages
    else:
        # No context needed (e.g., casual chat)
        formatted_messages = messages
        
    # Call our strict LLM wrapper interface
    response_data = llm.generate(formatted_messages)
    
    # Update the state with the new AI message and the exact token metrics
    return {
        "messages": [response_data["message"]],
        "token_usage": response_data["token_usage"],
        "current_agent": "generator"
    }