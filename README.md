# 🚀 Agentic RAG Toolkit

A modular, configuration-driven toolkit for building Retrieval-Augmented Generation (RAG) and Agentic AI systems. Built on top of **LangGraph**, **LangChain**, and **uv**, this template uses SOLID principles to completely decouple your AI logic from your infrastructure.

Want to swap from OpenAI to Anthropic? Or from local ChromaDB to Cloud Qdrant? Just change one line in a YAML file. No code rewrites required. **"Here is the model, here is the router, boom, done."**

## ✨ Core Capabilities

* **🔌 Plug-and-Play Architecture:** Abstract Base Classes (ABCs) enforce strict contracts. Swap LLMs, Embeddings, and Vector Databases instantly via YAML configs.
* **🧠 Dynamic Component Factory:** Zero hardcoded dependencies. The system dynamically imports and instantiates components at runtime based on your configuration.
* **📊 Granular State Management:** Deep UI-ready metadata. The LangGraph state automatically tracks token consumption, retrieval distance scores, and raw embeddings for downstream frontend visualization.
* **⚡ 100% Free-Tier Ready:** Comes out-of-the-box with wrappers for Google's free Gemini 2.5 Flash and Gemini Embeddings.

---

## 🚀 Quick Start

### 1. Installation
This project uses [uv](https://github.com/astral-sh/uv) for blazing-fast dependency management. Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone [https://github.com/yourusername/my-agentic-toolkit.git](https://github.com/yourusername/my-agentic-toolkit.git)
cd my-agentic-toolkit

# Install dependencies via uv
uv sync

```

### 2. Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
GOOGLE_API_KEY="your_gemini_api_key"
# OPENAI_API_KEY="your_openai_api_key" (If using OpenAI)
LANGCHAIN_TRACING_V2="false"

```

### 3. Run the Default Bot

Put a `sample.pdf` in the root folder, then run:

```bash
# Ingest the PDF into the local Chroma vector database
uv run python -m tests.ingest_pdf

# Start chatting!
uv run python -m tests.chat_bot

```

---

## 🛠️ How to Build Your Own Chatbot

The beauty of this toolkit is that you don't need to touch the core orchestration code to build something entirely new. Here is how you customize it.

### Step 1: Create a Custom Component

Want to use a new Vector Database (like Pinecone) or a local LLM (like Ollama)? Just create a Python class that inherits from our core interfaces (`BaseLLM` or `BaseVectorDB`).

```python
# src/components/llms/ollama_wrapper.py
from src.core.base_llm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model_name="llama3"):
        # Setup your local connection here
        pass

    def generate(self, messages, **kwargs):
        # Call the model and return the strict dictionary format
        return {
            "message": response_message,
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }

```

### Step 2: Update the YAML Configuration

You don't need to write `if/else` statements. Just point the factory to your new class path in your config file (e.g., `configs/my_custom_bot.yaml`):

```yaml
llm:
  class_path: "src.components.llms.ollama_wrapper.OllamaLLM"
  init_args:
    model_name: "llama3"

vector_db:
  class_path: "src.components.retrievers.chroma_wrapper.ChromaDBWrapper"
  init_args:
    collection_name: "my_docs"

```

### Step 3: Build a Custom Graph (Optional)

If your chatbot needs a more complex workflow (e.g., adding a web search tool or a grading node), you can easily wire up a new graph while utilizing the exact same dynamic factory.

```python
# src/flows/my_custom_graph.py
from langgraph.graph import StateGraph
from src.core.state import AgentState
from src.utils.factory import ComponentFactory

def build_custom_graph(config):
    # 1. Load components dynamically
    llm = ComponentFactory.create_instance(config["llm"])
    db = ComponentFactory.create_instance(config["vector_db"])
    
    # 2. Wire your custom LangGraph nodes
    workflow = StateGraph(AgentState)
    # ... add custom nodes and routing edges ...
    
    return workflow.compile()

```

## 🤝 Contributing

Contributions are welcome! Please ensure that any new LLM or Database wrapper strictly adheres to the Abstract Base Classes defined in `src/core/`.

---