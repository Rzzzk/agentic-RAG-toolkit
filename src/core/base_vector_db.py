from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseVectorDB(ABC):
    """
    The strict contract for all Vector Database wrappers.
    Ensures seamless swapping between local (Chroma) and cloud (Qdrant/Pinecone) stores.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Embeds and stores a list of text chunks.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 4, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant chunks for a given query.
        
        Args:
            query: The user's search string.
            top_k: Number of documents to return.
            **kwargs: DB-specific filters.
            
        Returns:
            A list of dictionaries. Each dictionary MUST contain:
                - "content": The actual text chunk.
                - "distance": The similarity score (e.g., cosine distance).
                - "embedding": The raw vector representation.
                - "metadata": Source document information.
        """
        pass