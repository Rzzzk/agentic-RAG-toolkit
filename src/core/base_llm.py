from abc import ABC, abstractmethod
from typing import Any, Dict
from langchain_core.messages import BaseMessage

class BaseLLM(ABC):
    """
    Abstract Base Class defining the strict contract for all Language Models.
    Whether wrapping OpenAI, Anthropic, or a localized model, it must inherit from this.
    """
    
    @abstractmethod
    def generate(self, messages: list[BaseMessage], **kwargs) -> Dict[str, Any]:
        """
        Processes a list of LangChain messages and returns the model's response.
        
        Args:
            messages: The conversation history or current prompt.
            **kwargs: Dynamic generation parameters (temperature, top_p, etc.) passed via config.
            
        Returns:
            A standardized dictionary containing:
                - "message": The output BaseMessage (AIMessage).
                - "token_usage": A dictionary of token metrics (prompt, completion, total).
        """
        pass