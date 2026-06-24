"""
ui/ui.py
--------
Reflex frontend — a configurable chat UI that talks to ANY chatbot API endpoint.

Configuration (all via environment variables or the constants below):
  API_BASE_URL   – base URL of the FastAPI server  (default: http://localhost:8000)
  AVAILABLE_BOTS – comma-separated list of bot IDs  (default: fetched from /  endpoint)
  DEFAULT_BOT    – which bot to select on load       (default: first in list)

The UI is intentionally unaware of LangGraph, Gemini, or any RAG internals.
It only speaks JSON with the /chat endpoint.
"""
import os
import uuid
import httpx
import reflex as rx

# ---------------------------------------------------------------------------
# UI-level configuration — change endpoint here or via env vars
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
CHAT_ENDPOINT: str = f"{API_BASE_URL}/chat"
INFO_ENDPOINT: str = f"{API_BASE_URL}/"

# If you want to hard-code a fixed list of bots instead of fetching from the API,
# set the env var: AVAILABLE_BOTS="free_pdf_bot,my_other_bot"
_env_bots = os.getenv("AVAILABLE_BOTS", "")
STATIC_BOT_LIST: list[str] = [b.strip() for b in _env_bots.split(",") if b.strip()]

DEFAULT_BOT: str = os.getenv("DEFAULT_BOT", STATIC_BOT_LIST[0] if STATIC_BOT_LIST else "free_pdf_bot")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ChatState(rx.State):
    # Conversation
    chat_history: list[dict[str, str]] = []
    current_input: str = ""
    is_processing: bool = False

    # Bot selection
    available_bots: list[str] = STATIC_BOT_LIST or [DEFAULT_BOT]
    selected_bot: str = DEFAULT_BOT
    session_id: str = str(uuid.uuid4())

    # Diagnostics from the API response
    token_metrics: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    context_chunks_used: int = 0
    current_agent: str = "Idle"

    # Config display
    api_url: str = CHAT_ENDPOINT
    error_message: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @rx.event
    async def on_load(self):
        """
        On page load, fetch the list of available bots from the API's health
        endpoint so the UI stays in sync with whatever bots are registered.
        Only runs if no static list was configured via env var.
        """
        if STATIC_BOT_LIST:
            return  # Static config wins; skip the network call

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(INFO_ENDPOINT)
                resp.raise_for_status()
                data = resp.json()
                bots = data.get("available_bots", [DEFAULT_BOT])
                self.available_bots = bots
                if self.selected_bot not in bots and bots:
                    self.selected_bot = bots[0]
        except Exception:
            # If the server isn't reachable yet, keep defaults silently
            pass

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------
    def set_current_input(self, text: str):
        self.current_input = text

    def change_bot(self, bot_name: str):
        """Reset conversation when the user picks a different bot."""
        self.selected_bot = bot_name
        self.chat_history = []
        self.session_id = str(uuid.uuid4())
        self.token_metrics = {"prompt_tokens": 0, "completion_tokens": 0}
        self.context_chunks_used = 0
        self.current_agent = "Idle"
        self.error_message = ""

    @rx.event(background=True)
    async def process_message(self):
        """Send the user's message to the API and update the UI with the response."""
        if not self.current_input.strip():
            return

        user_text = self.current_input

        async with self:
            self.chat_history.append({"role": "user", "content": user_text})
            self.current_input = ""
            self.is_processing = True
            self.current_agent = "Waiting for API…"
            self.error_message = ""
        yield

        payload = {
            "session_id": self.session_id,
            "bot_id": self.selected_bot,
            "message": user_text,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                data = response.json()

            async with self:
                self.chat_history.append({"role": "bot", "content": data["reply"]})
                self.token_metrics = data.get("tokens", {"prompt_tokens": 0, "completion_tokens": 0})
                self.context_chunks_used = data.get("chunks", 0)
                self.current_agent = data.get("agent", "generator")
                self.is_processing = False

        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response else str(exc)
            async with self:
                self.chat_history.append({"role": "bot", "content": f"❌ API Error {exc.response.status_code}: {detail}"})
                self.error_message = detail
                self.is_processing = False

        except Exception as exc:
            async with self:
                self.chat_history.append({"role": "bot", "content": f"❌ Connection error: {str(exc)}"})
                self.error_message = str(exc)
                self.is_processing = False
        yield


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def message_bubble(message: dict) -> rx.Component:
    is_user = message["role"] == "user"
    return rx.box(
        rx.markdown(message["content"]),
        background=rx.cond(
            is_user,
            "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "rgba(255,255,255,0.07)",
        ),
        color=rx.cond(is_user, "white", "rgba(255,255,255,0.9)"),
        padding="0.85em 1.1em",
        border_radius=rx.cond(is_user, "18px 18px 4px 18px", "18px 18px 18px 4px"),
        margin_bottom="0.75em",
        align_self=rx.cond(is_user, "flex-end", "flex-start"),
        max_width="78%",
        box_shadow=rx.cond(
            is_user,
            "0 4px 15px rgba(102,126,234,0.4)",
            "0 2px 8px rgba(0,0,0,0.3)",
        ),
        border=rx.cond(is_user, "none", "1px solid rgba(255,255,255,0.1)"),
    )


def sidebar() -> rx.Component:
    return rx.box(
        rx.vstack(
            # Header
            rx.vstack(
                rx.text("🤖", font_size="2rem"),
                rx.heading("Agent Hub", size="5", color="white", font_weight="700"),
                rx.text("Powered by Agentic RAG", size="1", color="rgba(255,255,255,0.5)"),
                align="center",
                spacing="1",
                padding_bottom="1.5em",
            ),
            rx.divider(border_color="rgba(255,255,255,0.1)"),

            # Bot selector
            rx.vstack(
                rx.text("Active Bot", size="1", color="rgba(255,255,255,0.5)", font_weight="600", letter_spacing="0.1em"),
                rx.select(
                    ChatState.available_bots,
                    value=ChatState.selected_bot,
                    on_change=ChatState.change_bot,
                    width="100%",
                    color_scheme="violet",
                ),
                spacing="2",
                width="100%",
                padding_top="1em",
            ),
            rx.divider(border_color="rgba(255,255,255,0.1)", margin_y="1em"),

            # Session info
            rx.vstack(
                rx.text("Connection", size="1", color="rgba(255,255,255,0.5)", font_weight="600", letter_spacing="0.1em"),
                rx.text(ChatState.api_url, size="1", color="rgba(255,255,255,0.4)", word_break="break-all"),
                rx.hstack(
                    rx.box(width="8px", height="8px", border_radius="50%", background="#4ade80"),
                    rx.text(f"Session …{ChatState.session_id[-6:]}", size="1", color="rgba(255,255,255,0.5)"),
                    spacing="2",
                    align="center",
                ),
                spacing="2",
                width="100%",
                padding_top="0.5em",
            ),
            rx.divider(border_color="rgba(255,255,255,0.1)", margin_y="1em"),

            # Diagnostics
            rx.vstack(
                rx.text("Diagnostics", size="1", color="rgba(255,255,255,0.5)", font_weight="600", letter_spacing="0.1em"),
                rx.hstack(
                    rx.text("Node:", size="1", color="rgba(255,255,255,0.4)"),
                    rx.text(ChatState.current_agent, size="1", color="#a78bfa", font_weight="600"),
                    justify="between",
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Context chunks:", size="1", color="rgba(255,255,255,0.4)"),
                    rx.text(ChatState.context_chunks_used, size="1", color="#60a5fa"),
                    justify="between",
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Prompt tokens:", size="1", color="rgba(255,255,255,0.4)"),
                    rx.text(ChatState.token_metrics["prompt_tokens"], size="1", color="#34d399"),
                    justify="between",
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Completion tokens:", size="1", color="rgba(255,255,255,0.4)"),
                    rx.text(ChatState.token_metrics["completion_tokens"], size="1", color="#34d399"),
                    justify="between",
                    width="100%",
                ),
                spacing="2",
                width="100%",
                padding_top="0.5em",
            ),

            # Error banner
            rx.cond(
                ChatState.error_message != "",
                rx.box(
                    rx.text(f"⚠ {ChatState.error_message}", size="1", color="#fca5a5"),
                    background="rgba(239,68,68,0.15)",
                    border="1px solid rgba(239,68,68,0.4)",
                    border_radius="8px",
                    padding="0.75em",
                    margin_top="1em",
                    width="100%",
                ),
            ),

            spacing="0",
            width="100%",
            height="100%",
            padding="1.5em",
            align_items="start",
        ),
        width="280px",
        min_width="280px",
        height="100vh",
        background="rgba(255,255,255,0.04)",
        border_right="1px solid rgba(255,255,255,0.08)",
        backdrop_filter="blur(12px)",
        overflow_y="auto",
    )


def chat_area() -> rx.Component:
    return rx.vstack(
        # Top bar
        rx.hstack(
            rx.vstack(
                rx.heading("Universal Agent UI", size="6", color="white"),
                rx.text(
                    f"Chatting with: ",
                    rx.text.span(ChatState.selected_bot, color="#a78bfa", font_weight="700"),
                    size="2",
                    color="rgba(255,255,255,0.5)",
                ),
                spacing="0",
                align_items="start",
            ),
            rx.spacer(),
            rx.cond(
                ChatState.is_processing,
                rx.badge("● Processing", color_scheme="violet", variant="soft"),
                rx.badge("● Ready", color_scheme="green", variant="soft"),
            ),
            width="100%",
            padding="1.5em 2em",
            border_bottom="1px solid rgba(255,255,255,0.08)",
        ),

        # Messages
        rx.box(
            rx.cond(
                ChatState.chat_history.length() == 0,
                rx.vstack(
                    rx.text("💬", font_size="3rem"),
                    rx.text("Start a conversation", size="4", color="rgba(255,255,255,0.3)", font_weight="600"),
                    rx.text("Ask anything about your documents", size="2", color="rgba(255,255,255,0.2)"),
                    align="center",
                    spacing="2",
                    position="absolute",
                    top="50%",
                    left="50%",
                    transform="translate(-50%, -50%)",
                ),
            ),
            rx.foreach(ChatState.chat_history, message_bubble),
            position="relative",
            flex="1",
            overflow_y="auto",
            width="100%",
            padding="1.5em 2em",
            display="flex",
            flex_direction="column",
        ),

        # Input bar
        rx.box(
            rx.hstack(
                rx.text_area(
                    placeholder="Ask your agent a question… (Shift+Enter for new line)",
                    value=ChatState.current_input,
                    on_change=ChatState.set_current_input,
                    on_key_down=rx.call_script(
                        "if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();document.getElementById('send_btn').click();}"
                    ),
                    disabled=ChatState.is_processing,
                    flex="1",
                    rows="3",
                    resize="vertical",
                    background="rgba(255,255,255,0.07)",
                    border="1px solid rgba(255,255,255,0.15)",
                    color="white",
                    _placeholder={"color": "rgba(255,255,255,0.3)"},
                    _focus={"border_color": "#7c3aed", "box_shadow": "0 0 0 2px rgba(124,58,237,0.3)", "outline": "none"},
                    border_radius="12px",
                    padding="0.75em 1em",
                    font_size="0.95rem",
                    line_height="1.5",
                    min_height="72px",
                ),
                rx.button(
                    rx.cond(ChatState.is_processing, rx.spinner(size="3"), rx.text("Send ↵")),
                    id="send_btn",
                    on_click=ChatState.process_message,
                    disabled=ChatState.is_processing,
                    background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    color="white",
                    border_radius="12px",
                    padding="0.75em 1.4em",
                    font_weight="600",
                    cursor="pointer",
                    _hover={"opacity": "0.85", "transform": "translateY(-1px)"},
                    transition="all 0.2s ease",
                ),
                width="100%",
                spacing="3",
            ),
            padding="1em 2em 1.5em",
            border_top="1px solid rgba(255,255,255,0.08)",
            background="rgba(0,0,0,0.2)",
            width="100%",
        ),

        width="100%",
        height="100vh",
        spacing="0",
        overflow="hidden",
    )


def index() -> rx.Component:
    return rx.box(
        rx.hstack(
            sidebar(),
            chat_area(),
            spacing="0",
            width="100%",
            height="100vh",
            overflow="hidden",
        ),
        background="linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
        min_height="100vh",
        font_family="'Inter', sans-serif",
        on_mount=ChatState.on_load,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = rx.App(
    stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
)
app.add_page(index, route="/")