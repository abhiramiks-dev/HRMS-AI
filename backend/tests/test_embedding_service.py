"""Tests for the local embedding service."""

from backend.app.services.embedding_service import EmbeddingService


def test_embedding_can_be_generated_for_text() -> None:
    """A query can be converted into a non-empty embedding vector."""
    service = EmbeddingService()

    embedding = service.embed_query("What is the annual leave policy?")

    assert embedding
    assert all(isinstance(value, float) for value in embedding)


def test_multiple_texts_produce_multiple_embeddings() -> None:
    """Each input text produces one embedding."""
    service = EmbeddingService()

    embeddings = service.embed_documents(["Annual leave policy", "Sick leave policy"])

    assert len(embeddings) == 2
    assert all(embedding for embedding in embeddings)


def test_embeddings_have_consistent_vector_dimension() -> None:
    """Document and query embeddings share the model's vector dimension."""
    service = EmbeddingService()

    document_embeddings = service.embed_documents(["Annual leave", "Emergency leave"])
    query_embedding = service.embed_query("How much leave can I take?")

    assert len(document_embeddings[0]) == len(document_embeddings[1])
    assert len(document_embeddings[0]) == len(query_embedding)
