import os
from typing import Any, Dict, List
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.base_vector_db import BaseVectorDB

class GeminiChromaDBWrapper(BaseVectorDB):
    """Concrete implementation using 100% free Google Embeddings."""
    def __init__(self, collection_name: str = "pdf_collection", persist_directory: str = "./chroma_data"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing from environment variables.")
            
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        self.vector_store.add_texts(texts=documents, metadatas=metadatas)

    def search(self, query: str, top_k: int = 4, **kwargs) -> List[Dict[str, Any]]:
        results = self.vector_store.similarity_search_with_score(query, k=top_k, **kwargs)
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "distance": score,
                "metadata": doc.metadata,
                "embedding": [] 
            })
        return formatted_results