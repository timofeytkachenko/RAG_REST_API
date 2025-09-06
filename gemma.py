import os
from typing import List

import huggingface_hub
import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

HF_TOKEN = os.getenv("HF_TOKEN", "")


def authenticate_huggingface() -> None:
    """Authenticate with HuggingFace Hub using token from environment."""
    if HF_TOKEN and HF_TOKEN.strip():
        try:
            huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
            print("Successfully authenticated with HuggingFace Hub")
        except Exception as e:
            print(f"Warning: HuggingFace authentication failed: {e}")
            print("Some models may not be accessible without proper authentication")
    else:
        print(
            "Warning: No HF_TOKEN provided. Google Embedding Gemma may not be accessible."
        )


class GemmaEmbeddings(Embeddings):
    """Custom embeddings class for google/embeddinggemma-300m model."""

    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        """Initialize the Google Embedding Gemma model.

        Args:
            model_name: The model name on HuggingFace Hub
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model with authentication."""
        try:
            # Authenticate first
            authenticate_huggingface()

            # Load model
            self.model = SentenceTransformer(self.model_name)
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents using encode_document method.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self._load_model()

        try:
            embeddings = self.model.encode_document(texts)
            # Convert to list format expected by LangChain
            return embeddings.tolist()
        except Exception as e:
            print(f"Error embedding documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using encode_query method.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list
        """
        if self.model is None:
            self._load_model()

        try:
            embedding = self.model.encode_query(text)
            # Convert to list format expected by LangChain
            return embedding.tolist()
        except Exception as e:
            print(f"Error embedding query: {e}")
            raise

    def similarity(
        self, query_embedding: List[float], document_embeddings: List[List[float]]
    ) -> List[float]:
        """Compute similarity scores between query and documents.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors

        Returns:
            List of similarity scores
        """
        if self.model is None:
            self._load_model()

        try:
            # Convert back to numpy arrays
            query_emb = np.array(query_embedding)
            doc_embs = np.array(document_embeddings)

            similarities = self.model.similarity(query_emb, doc_embs)
            return similarities[0].tolist()  # Return first row as list
        except Exception as e:
            print(f"Error computing similarities: {e}")
            raise
