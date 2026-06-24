# 🚀 Agentic RAG Toolkit

A modular, configuration-driven toolkit for building Retrieval-Augmented Generation (RAG) and Agentic AI systems. Built on **LangGraph**, **LangChain**, and **uv**, this template uses SOLID principles to completely decouple your AI logic from your infrastructure.

Want to swap from OpenAI to Anthropic? Or from ChromaDB to Qdrant? Change one line in a YAML file. No code rewrites required. **"Here is the model, here is the router, boom, done."**

---

## ✨ Core Capabilities

| Feature | Description |
|---|---|
| 🔌 **Plug-and-Play Architecture** | Abstract Base Classes enforce strict contracts. Swap LLMs, Embedders, and Vector DBs instantly via YAML. |
| 🧠 **Dynamic Component Factory** | Zero hardcoded dependencies — components are imported and instantiated at runtime. |
| 🌐 **REST API Layer** | FastAPI server that exposes any registered chatbot through a single `/chat` endpoint. |
| 🖥️ **Reflex UI** | A beautiful, config-driven chat frontend that works with any API endpoint. |
| 📊 **Granular State Tracking** | Token usage, retrieval scores, and active node diagnostics streamed to the UI. |
| ⚡ **100% Free-Tier Ready** | Ships with wrappers for Google Gemini 2.5 Flash + Gemini Embeddings. |

---

## 📁 Project Structure

```
my_agentic_toolkit/
│
├── api/                        # API layer — the only public surface
│   ├── __init__.py
│   ├── chatbot_interface.py    # Abstract contract every chatbot must implement
│   ├── pdf_rag_bot.py          # Concrete bot: PDF RAG with Gemini + ChromaDB
│   ├── bot_registry.py         # Maps bot IDs → instances (edit this to add bots)
│   └── server.py               # FastAPI app — zero AI imports, pure routing
│
├── ui/
│   └── ui.py                   # Reflex frontend — configured via env vars
│
├── src/                        # Core AI toolkit (don't need to touch for new bots)
│   ├── agents/                 # LangGraph nodes and routers
│   ├── components/             # LLM & VectorDB wrappers
│   ├── core/                   # ABCs and shared state schema
│   ├── flows/                  # Graph builders (e.g. build_rag_graph)
│   └── utils/                  # ComponentFactory
│
├── configs/                    # YAML configs, one per chatbot
│   └── free_pdf_bot.yaml
│
├── tests/
│   ├── ingest_pdf.py           # Populate ChromaDB from a PDF
│   └── chat_bot.py             # Terminal chatbot (no API needed)
│
├── .env                        # API keys (never commit this)
├── rxconfig.py                 # Reflex app config
└── pyproject.toml
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/my-agentic-toolkit.git
cd my-agentic-toolkit

# Install all dependencies (uses uv)
uv sync
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your_gemini_api_key"
# OPENAI_API_KEY="your_openai_api_key"
LANGCHAIN_TRACING_V2="false"
```

### 3. Ingest a PDF (one-time setup)

Drop a `sample.pdf` in the project root, then run:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -m tests.ingest_pdf
```

### 4. Test in the terminal (no API needed)

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -m tests.chat_bot
```

---

## 🌐 Running the API + UI

The system runs as **two separate processes** — start both from the project root.

### Terminal 1 — Start the FastAPI backend

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Verify it's running — open http://localhost:8000 in your browser. You should see:

```json
{ "status": "ok", "available_bots": ["free_pdf_bot"] }
```

### Terminal 2 — Start the Reflex UI

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run reflex run
```

Open http://localhost:3000 — the chat interface will load and connect to the API automatically.

### Point the UI at a different server

```bash
API_BASE_URL=http://192.168.1.100:8000 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run reflex run
```

### Quick API test with curl

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-1","bot_id":"free_pdf_bot","message":"What is this document about?"}'
```

---

## 🛠️ How to Build & Deploy a New Chatbot

Follow these 4 steps to add a completely new chatbot and expose it through the API and UI.

---

### Step 1 — Create a YAML config for your bot

Each chatbot gets its own config file in `configs/`. The factory reads `class_path` and `init_args` to instantiate the components dynamically.

```yaml
# configs/my_new_bot.yaml

llm:
  class_path: "src.components.llms.gemini_wrapper.GeminiLLM"
  init_args:
    model_name: "gemini-2.5-flash"
    temperature: 0.3

vector_db:
  class_path: "src.components.retrievers.gemini_chroma_wrapper.GeminiChromaDBWrapper"
  init_args:
    collection_name: "my_new_collection"
    persist_directory: "./chroma_data"
```

> To use a **custom LLM or Vector DB**, create a wrapper class in `src/components/` that inherits from `BaseLLM` or `BaseVectorDB` (see Step 1b below), then point `class_path` to it.

---

### Step 1b — (Optional) Create a custom component wrapper

Only needed if you're using a provider not already in `src/components/`.

```python
# src/components/llms/ollama_wrapper.py
from src.core.base_llm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model_name="llama3"):
        # initialise your connection here
        pass

    def generate(self, messages, **kwargs):
        # must return this exact dict shape
        return {
            "message": ai_message,           # langchain AIMessage
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        }
```

---

### Step 2 — Create a bot class implementing `ChatbotInterface`

This is the adapter between your LangGraph graph and the API server. It follows the same pattern as `tests/chat_bot.py`.

```python
# api/my_new_bot.py
import os, sys, yaml
from typing import Any, Dict
from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.flows.rag_graph import build_rag_graph   # or your custom graph builder
from api.chatbot_interface import ChatbotInterface

class MyNewBot(ChatbotInterface):

    def __init__(self, config_path: str = "configs/my_new_bot.yaml"):
        self._config_path = config_path
        self._engine = None  # compiled lazily on first message

    def _get_engine(self):
        if self._engine is None:
            with open(self._config_path) as f:
                config = yaml.safe_load(f)
            self._engine = build_rag_graph(config)
        return self._engine

    def create_initial_state(self) -> Dict[str, Any]:
        return {
            "messages": [],
            "retrieved_context": [],
            "token_usage": {},
            "embedding_details": {},
            "current_agent": "system",
            "errors": [],
        }

    def chat(self, state: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        engine = self._get_engine()
        state["messages"].append(HumanMessage(content=user_message))
        result_state = engine.invoke(state)

        reply = result_state["messages"][-1].content
        usage  = result_state.get("token_usage", {})

        return {
            "state":  result_state,
            "reply":  reply,
            "tokens": {
                "prompt_tokens":     usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            "chunks": len(result_state.get("retrieved_context", [])),
            "agent":  result_state.get("current_agent", "generator"),
        }
```

---

### Step 3 — Register the bot in the registry

Open `api/bot_registry.py` and add **two lines**:

```python
# api/bot_registry.py

from api.chatbot_interface import ChatbotInterface
from api.pdf_rag_bot import PdfRagBot
from api.my_new_bot import MyNewBot          # ← import your bot

_BOT_CATALOGUE: Dict[str, ChatbotInterface] = {
    "free_pdf_bot": PdfRagBot(config_path="configs/free_pdf_bot.yaml"),
    "my_new_bot":   MyNewBot(config_path="configs/my_new_bot.yaml"),   # ← register it
}
```

That's it for the API side. Restart the server and verify:

```bash
curl http://localhost:8000/
# → {"status":"ok","available_bots":["free_pdf_bot","my_new_bot"]}
```

---

### Step 4 — Make the UI use the new bot

The UI **auto-discovers** registered bots from the API's health endpoint on page load — no code change is needed. Just refresh http://localhost:3000 and select your new bot from the dropdown.

#### Option A — Auto-discovery (default)

Restart Reflex. The dropdown will include every bot listed by `GET /`.

#### Option B — Hard-code a list via env var

```bash
AVAILABLE_BOTS="my_new_bot,free_pdf_bot" DEFAULT_BOT="my_new_bot" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run reflex run
```

#### Option C — Point the UI at a completely different API server

```bash
API_BASE_URL=http://production-server:8000 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run reflex run
```

| Environment Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | Base URL of the FastAPI backend |
| `AVAILABLE_BOTS` | *(fetched from API)* | Comma-separated list of bot IDs to show |
| `DEFAULT_BOT` | first bot in list | Which bot is selected on load |

---

## 🔌 API Reference

### `GET /`
Health check — returns all registered bots.

```json
{ "status": "ok", "available_bots": ["free_pdf_bot"] }
```

### `POST /chat`
Send a message and receive a reply.

**Request body:**
```json
{
  "session_id": "unique-user-session-id",
  "bot_id": "free_pdf_bot",
  "message": "What is this document about?"
}
```

**Response:**
```json
{
  "reply": "The document is about...",
  "tokens": { "prompt_tokens": 512, "completion_tokens": 128 },
  "chunks": 3,
  "agent": "generator"
}
```

> Session state (conversation history) is stored in-memory on the server. Each unique `session_id` gets its own isolated conversation. Use a new `session_id` to start a fresh chat.

---

## 🤝 Contributing

Contributions are welcome! Please ensure:
- New LLM or VectorDB wrappers inherit from the ABCs in `src/core/`.
- New chatbots implement `ChatbotInterface` in `api/chatbot_interface.py`.
- No AI-specific imports belong in `api/server.py` — everything goes through the registry.

---