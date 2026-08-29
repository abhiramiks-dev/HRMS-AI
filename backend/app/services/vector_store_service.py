"""Persistent ChromaDB storage and retrieval for document embeddings."""

from pathlib import Path
from typing import TypedDict

import chromadb


class RetrievedChunk(TypedDict):
    """A document chunk returned by a vector similarity search."""

    text: str
    source_document: str
    chunk_index: int
    distance: float


class VectorStoreService:
    """Store and retrieve embedded document chunks in a local ChromaDB store."""

    DEFAULT_COLLECTION_NAME = "hrms_documents"

    def __init__(
        self,
        persist_directory: str | Path = "chroma_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Open a persistent ChromaDB collection.

        Args:
            persist_directory: Directory where ChromaDB stores its local data.
            collection_name: Name of the collection used for document chunks.
        """
        self._client = chromadb.PersistentClient(path=str(Path(persist_directory)))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_document_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source_filename: str,
    ) -> None:
        """Store document chunks, embeddings, and source metadata.

        Args:
            chunks: Text chunks to store.
            embeddings: Embedding vector corresponding to each chunk.
            source_filename: Filename from which the chunks were created.

        Raises:
            ValueError: If the chunk and embedding counts differ.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must contain the same number of items")
        if not chunks:
            return

        chunk_indices = list(range(len(chunks)))
        self._collection.upsert(
            ids=[f"{source_filename}:{index}" for index in chunk_indices],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source_document": source_filename, "chunk_index": index}
                for index in chunk_indices
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3,
    ) -> list[RetrievedChunk]:
        """Find the chunks nearest to a query embedding.

        Args:
            query_embedding: Vector representation of the user's query.
            n_results: Maximum number of chunks to return.

        Returns:
            Retrieved chunks ordered by ChromaDB distance, nearest first.

        Raises:
            ValueError: If ``n_results`` is not positive.
        """
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text": str(document),
                "source_document": str(metadata["source_document"]),
                "chunk_index": int(metadata["chunk_index"]),
                "distance": float(distance),
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]
