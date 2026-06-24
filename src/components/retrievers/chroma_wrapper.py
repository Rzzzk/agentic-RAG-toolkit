from typing import Any, Dict, List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Import the contract we wrote in Step 3
from src.core.base_vector_db import BaseVectorDB

class ChromaDBWrapper(BaseVectorDB):
    """
    Concrete implementation of BaseVectorDB using local ChromaDB.
    """
    
    def __init__(self, collection_name: str = "agent_docs", persist_directory: str = "./chroma_data"):
        # Using OpenAI embeddings by default, but this could also be injected via config
        self.embeddings = OpenAIEmbeddings()
        
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """Embeds and saves documents to the local Chroma instance."""
        self.vector_store.add_texts(texts=documents, metadatas=metadatas)

    def search(self, query: str, top_k: int = 4, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieves documents and ensures vectors and distance metrics are included.
        """
        # Use similarity_search_with_score to get the distance metric
        results = self.vector_store.similarity_search_with_score(query, k=top_k, **kwargs)
        
        formatted_results = []
        for doc, score in results:
            # Re-embed the query or fetch the doc embedding if needed for UI visualization
            # For performance, we grab the text and score; if full vectors are needed, 
            # we can call self.embeddings.embed_query(doc.page_content) here.
            
            formatted_results.append({
                "content": doc.page_content,
                "distance": score,
                "metadata": doc.metadata,
                # Embeddings can be costly to return on every search if large, 
                # but we include the hook here for detailed UI rendering.
                "embedding": self.embeddings.embed_query(doc.page_content) 
            })
            
        return formatted_results