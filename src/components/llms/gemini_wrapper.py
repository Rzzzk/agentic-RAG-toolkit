import os
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from src.core.base_llm import BaseLLM

class GeminiLLM(BaseLLM):
    """
    Concrete implementation of BaseLLM for Google's free-tier Gemini models.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.0):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is missing.")
            
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )

    def generate(self, messages: list[BaseMessage], **kwargs) -> Dict[str, Any]:
        response = self.llm.invoke(messages, **kwargs)
        
        # Format metrics standardly for your downstream UI tracking
        return {
            "message": response,
            "token_usage": {
                "prompt_tokens": response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)
            }
        }

print("Free LLM Component successfully registered to contract.")