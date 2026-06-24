import os
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.flows.rag_graph import build_rag_graph

def main():
    # 1. Load secure API keys from .env
    load_dotenv()
    
    # 2. Load the configuration profile
    config_path = "configs/main.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # 3. Build the LangGraph Engine dynamically
    print("Compiling Agentic Graph...")
    app = build_rag_graph(config)
    print("Graph compiled successfully. Type 'exit' or 'quit' to stop.\n")

    # 4. Initialize the state payload
    # This dictionary will accumulate messages and deep metadata over the conversation
    state = {
        "messages": [],
        "retrieved_context": [],
        "token_usage": {},
        "embedding_details": {},
        "current_agent": "system",
        "errors": []
    }

    # 5. Interactive Chat Loop
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Append the new user message to the state
        state["messages"].append(HumanMessage(content=user_input))

        # Invoke the graph
        # The graph will route the question, retrieve context if necessary, and generate a response
        result_state = app.invoke(state)
        
        # Update our local state with the graph's output
        state = result_state

        # Extract the latest AI message
        ai_response = state["messages"][-1].content
        print(f"\nAI: {ai_response}\n")
        
        # Print the metadata we captured for debugging (your future UI will use this)
        tokens = state.get("token_usage", {})
        print(f"[Debug] Tokens Used: Prompt: {tokens.get('prompt_tokens', 0)} | Completion: {tokens.get('completion_tokens', 0)}")
        print("-" * 50)

if __name__ == "__main__":
    main()