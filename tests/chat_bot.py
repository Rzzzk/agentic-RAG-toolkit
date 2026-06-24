import os
import sys
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Ensure Python can find the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.flows.rag_graph import build_rag_graph

def main():
    load_dotenv()
    
    # 1. Load your specific free tier configuration
    config_path = "configs/free_pdf_bot.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # 2. Build the graph (Factory does the heavy lifting here)
    print("Compiling Chatbot Graph...")
    app = build_rag_graph(config)
    print("Graph compiled successfully!\n")
    print("==================================================")
    print("🤖 PDF Chatbot Online. Type 'exit' to quit.")
    print("==================================================\n")

    # 3. Initialize state
    state = {
        "messages": [],
        "retrieved_context": [],
        "token_usage": {},
        "embedding_details": {},
        "current_agent": "system",
        "errors": []
    }

    # 4. The Chat Loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Shutting down...")
            break
        if not user_input.strip():
            continue

        state["messages"].append(HumanMessage(content=user_input))

        # Stream the execution to catch the response
        try:
            result_state = app.invoke(state)
            state = result_state
            
            ai_response = state["messages"][-1].content
            print(f"\n🤖 Bot: {ai_response}\n")
            
            # Optional debug info (remove or comment out for a cleaner UI)
            tokens = state.get("token_usage", {})
            ctx_length = len(state.get("retrieved_context", []))
            print(f"  [Debug] Tokens: Prompt {tokens.get('prompt_tokens', 0)} | Context Chunks Used: {ctx_length}")
            print("-" * 50)
            
        except Exception as e:
            print(f"\n❌ Error during execution: {e}\n")

if __name__ == "__main__":
    main()