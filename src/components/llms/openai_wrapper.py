import os
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

# Import the contract we wrote in Step 2
from src.core.base_llm import BaseLLM

class OpenAILLM(BaseLLM):
    """
    Concrete implementation of BaseLLM for OpenAI models (GPT-4o, etc.).
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        # We assume OPENAI_API_KEY is loaded into the environment via .env
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
            
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    def generate(self, messages: list[BaseMessage], **kwargs) -> Dict[str, Any]:
        """
        Invokes the OpenAI model and formats the output to match our strict contract.
        """
        # Call the actual LangChain model
        response = self.llm.invoke(messages, **kwargs)
        
        # Safely extract token metrics from the response metadata
        usage = response.response_metadata.get("token_usage", {})
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
        
        return {
            "message": response,
            "token_usage": token_usage
        }