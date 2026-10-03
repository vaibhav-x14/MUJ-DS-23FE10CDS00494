import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from omnirag.core.schemas import Chunk
from config.settings import settings


class SemanticRecursiveChunker:
    """Recursively splits documents on semantic boundaries (paragraphs, sentences)."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_length: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        cfg = settings.retrieval.chunking
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.min_chunk_length = min_chunk_length or cfg.min_chunk_length
        self.separators = separators or cfg.separators

    def _approx_token_count(self, text: str) -> int:
        return len(text.split())

    def split_text(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Splits document text into overlapping semantic chunks."""
        paragraphs = text.split("\n\n")
        raw_chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._approx_token_count(para)
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                raw_chunks.append("\n\n".join(current_chunk))
                # Create overlap by keeping trailing items if available
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            raw_chunks.append("\n\n".join(current_chunk))

        # Build formal Chunk models
        chunks: List[Chunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            cleaned = chunk_text.strip()
            if len(cleaned) < self.min_chunk_length:
                continue

            chunk_id = f"{doc_id}_chunk_{idx + 1}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=cleaned,
                    metadata=metadata or {},
                )
            )

        return chunks

    def chunk_directory(self, docs_dir: Path) -> List[Chunk]:
        """Loads and chunks all text/markdown files in a directory."""
        all_chunks: List[Chunk] = []
        for file_path in docs_dir.glob("*.*"):
            if file_path.suffix.lower() in [".txt", ".md", ".json"]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    doc_id = file_path.stem
                    chunks = self.split_text(content, doc_id=doc_id, metadata={"source_path": str(file_path)})
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"[Warning] Failed to chunk {file_path}: {e}")
        return all_chunks
