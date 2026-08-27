"""Utilities for loading and chunking HRMS text documents."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS_DIRECTORY = PROJECT_ROOT / "documents"


def load_text_document(filename: str) -> str:
    """Load a UTF-8 text document from the project's ``documents/`` directory.

    Args:
        filename: Name of the document to load, such as ``"leave_policy.txt"``.

    Returns:
        The document contents as a string.

    Raises:
        ValueError: If the filename would access a path outside ``documents/``.
        FileNotFoundError: If the requested document does not exist.
        IsADirectoryError: If the requested path is a directory.
        UnicodeDecodeError: If the document is not valid UTF-8 text.
    """
    document_path = (DOCUMENTS_DIRECTORY / filename).resolve()

    try:
        document_path.relative_to(DOCUMENTS_DIRECTORY.resolve())
    except ValueError as exc:
        raise ValueError("filename must refer to a file inside the documents directory") from exc

    return document_path.read_text(encoding="utf-8")


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split text into fixed-size character chunks with optional overlap.

    Chunks are created sequentially, and each chunk after the first starts
    ``chunk_size - chunk_overlap`` characters after the previous chunk.

    Args:
        text: Text to split.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters shared by adjacent chunks.

    Returns:
        A list of text chunks. Empty input produces an empty list.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or if
            ``chunk_overlap`` is negative or greater than or equal to
            ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1")

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step

    return chunks
