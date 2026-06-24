"""
server.py
---------
FastAPI application — the ONLY public surface of the API layer.
It knows nothing about LangGraph, Gemini, or ChromaDB.
All AI logic is accessed through the bot_registry abstraction.
"""
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# The server only imports from the api package — never from src directly
from api.bot_registry import get_bot, available_bot_ids

load_dotenv()

app = FastAPI(
    title="Agentic RAG API",
    description="Unified endpoint for pluggable LangGraph chatbots.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> conversation state dict
SESSIONS: dict = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatPayload(BaseModel):
    session_id: str
    bot_id: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    """Quick health check — also lists registered bots."""
    return {
        "status": "ok",
        "available_bots": available_bot_ids(),
    }


@app.post("/chat")
async def chat_endpoint(payload: ChatPayload):
    """
    Universal chat endpoint.

    The server:
      1. Looks up the requested bot via the registry.
      2. Retrieves (or creates) the session state for this user.
      3. Delegates the actual AI work to the bot's chat() method.
      4. Persists the updated state and returns a clean JSON response.
    """
    print(
        f"[API] bot='{payload.bot_id}'  session='{payload.session_id}'  "
        f"msg='{payload.message[:60]}...'"
    )

    bot = get_bot(payload.bot_id)

    # Create a fresh state for new sessions
    if payload.session_id not in SESSIONS:
        SESSIONS[payload.session_id] = bot.create_initial_state()

    current_state = SESSIONS[payload.session_id]

    try:
        # Run blocking AI work in a thread pool so the event loop stays free
        result = await asyncio.to_thread(bot.chat, current_state, payload.message)

        # Persist updated state (the bot returns it inside result["state"])
        SESSIONS[payload.session_id] = result["state"]

        return {
            "reply": result["reply"],
            "tokens": result["tokens"],
            "chunks": result["chunks"],
            "agent": result["agent"],
        }

    except Exception as exc:
        print(f"[API ERROR] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))