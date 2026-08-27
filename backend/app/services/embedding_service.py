"""Local text embedding service for the HRMS-AI application."""

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate vector embeddings with a local sentence-transformers model."""

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Load the configured embedding model.

        Args:
            model_name: Hugging Face model identifier or local model path.
        """
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        """Generate one embedding vector for each document chunk.

        Args:
            chunks: Document chunks to embed.

        Returns:
            A list of embedding vectors in the same order as ``chunks``.
        """
        if not chunks:
            return []

        embeddings = self._model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for a user query.

        Args:
            query: User query text to embed.

        Returns:
            The query's embedding vector.
        """
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()
