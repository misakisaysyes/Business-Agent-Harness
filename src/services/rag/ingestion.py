"""Markdown/TXT 加载、文档入库和增量索引。

Markdown/TXT loading, document ingestion, and incremental indexing.
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from harness.capabilities.rag import (
    DocumentIndexState,
    EmbeddingProvider,
    SourceDocument,
    VectorStore,
)
from harness.logging import AgentLog
from services.rag.splitter import DocumentSplitter

log = AgentLog(__name__)

SUPPORTED_SUFFIXES = frozenset({".docx", ".md", ".txt"})
RESERVED_METADATA = frozenset(
    {
        "document_id",
        "chunk_id",
        "chunk_index",
        "content_hash",
        "embedding_dimension",
        "embedding_model",
        "indexed_at",
        "knowledge_base_id",
        "scope",
        "source",
        "splitter_version",
        "user_id",
    }
)
MetadataAdapter = TypeAdapter(dict[str, JsonValue])


class IndexFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    error: str = Field(min_length=1)


class IndexingReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    indexed: int = 0
    skipped: int = 0
    deleted_chunks: int = 0
    failed: tuple[IndexFailure, ...] = ()


def _parse_markdown(text: str) -> tuple[str, dict[str, JsonValue]]:
    if not text.startswith("---\n"):
        return text, {}
    closing = text.find("\n---\n", 4)
    if closing < 0:
        raise ValueError("Markdown frontmatter is not closed")
    raw_metadata: object = yaml.safe_load(text[4:closing]) or {}
    if not isinstance(raw_metadata, dict):
        raise ValueError("Markdown frontmatter must be a mapping")
    metadata = MetadataAdapter.validate_python(raw_metadata)
    reserved = RESERVED_METADATA.intersection(metadata)
    if reserved:
        raise ValueError(f"frontmatter uses reserved metadata: {', '.join(sorted(reserved))}")
    return text[closing + 5 :], metadata


def _parse_docx(path: Path) -> tuple[str, dict[str, JsonValue]]:
    """Extract body paragraphs and tables while preserving heading structure."""

    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    blocks: list[str] = []
    for element in document.element.body.iterchildren():
        if element.tag == qn("w:p"):
            paragraph = Paragraph(element, document)
            text = paragraph.text.strip()
            if not text:
                continue
            match = re.search(r"heading\s*([1-6])$", paragraph.style.name.casefold())
            blocks.append(f"{'#' * int(match.group(1))} {text}" if match else text)
        elif element.tag == qn("w:tbl"):
            table = Table(element, document)
            for row in table.rows:
                cells = [" ".join(cell.text.split()) for cell in row.cells]
                if any(cells):
                    blocks.append(" | ".join(cells))

    metadata: dict[str, JsonValue] = {}
    title = document.core_properties.title
    if title and title.strip():
        metadata["title"] = title.strip()
    return "\n\n".join(blocks), metadata


def _document_id(
    knowledge_base_id: str,
    scope: Literal["public", "user"],
    user_id: str | None,
    relative_source: str,
) -> str:
    identity = f"{knowledge_base_id}\0{scope}\0{user_id or ''}\0{relative_source}"
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


def load_source_documents(
    source_root: str | Path,
    *,
    knowledge_base_id: str,
    scope: Literal["public", "user"],
    user_id: str | None = None,
) -> tuple[SourceDocument, ...]:
    """安全加载目录内的 Markdown/TXT，并生成稳定技术元数据。"""

    if scope == "user" and not user_id:
        raise ValueError("user scope requires user_id")
    if scope == "public" and user_id is not None:
        raise ValueError("public scope must not include user_id")
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    documents: list[SourceDocument] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        resolved = path.resolve()
        if not path.is_file() or not resolved.is_relative_to(root):
            continue
        if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            continue
        metadata: dict[str, JsonValue] = {}
        if path.suffix.casefold() == ".docx":
            text, metadata = _parse_docx(path)
        else:
            text = path.read_text(encoding="utf-8")
            if path.suffix.casefold() == ".md":
                text, metadata = _parse_markdown(text)
        text = text.strip()
        if not text:
            continue
        source = relative.as_posix()
        # The fingerprint covers both searchable text and frontmatter.  Metadata
        # is persisted on every Chunk and can affect filtering, so a metadata-
        # only change must not be treated as an unchanged document.
        fingerprint = json.dumps(
            {"metadata": metadata, "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        content_hash = hashlib.sha256(fingerprint.encode()).hexdigest()
        document_id = _document_id(knowledge_base_id, scope, user_id, source)
        technical: dict[str, JsonValue] = {
            "document_id": document_id,
            "knowledge_base_id": knowledge_base_id,
            "source": source,
            "scope": scope,
            "content_hash": content_hash,
            "title": metadata.get("title", path.stem),
        }
        if user_id is not None:
            technical["user_id"] = user_id
        documents.append(
            SourceDocument(
                document_id=document_id,
                text=text,
                source=source,
                metadata={**metadata, **technical},
            )
        )
    return tuple(documents)


class IngestionService:
    """可重复执行、按文档原子替换语义的同步索引服务。"""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        splitter: DocumentSplitter | None = None,
        knowledge_base_id: str = "knowledge_assistant",
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.splitter = splitter or DocumentSplitter()
        self.knowledge_base_id = knowledge_base_id

    def index_directory(
        self,
        source_root: str | Path,
        *,
        scope: Literal["public", "user"],
        user_id: str | None = None,
        rebuild: bool = False,
    ) -> IndexingReport:
        documents = load_source_documents(
            source_root,
            knowledge_base_id=self.knowledge_base_id,
            scope=scope,
            user_id=user_id,
        )
        indexed = skipped = deleted_chunks = 0
        failures: list[IndexFailure] = []

        with log.operation(
            "rag.index_directory",
            scope=scope,
            rebuild=rebuild,
            document_count=len(documents),
            embedding_model=self.embeddings.model_name,
            embedding_dimension=self.embeddings.dimension,
        ) as index_outcome:
            for document in documents:
                try:
                    with log.operation(
                        "rag.index_document",
                        document_id=document.document_id,
                        scope=scope,
                        rebuild=rebuild,
                        embedding_model=self.embeddings.model_name,
                        embedding_dimension=self.embeddings.dimension,
                    ) as document_outcome:
                        previous = self.store.get_document_state(document.document_id)
                        content_hash = str(document.metadata["content_hash"])
                        if (
                            not rebuild
                            and previous is not None
                            and previous.content_hash == content_hash
                            and previous.embedding_model == self.embeddings.model_name
                            and previous.embedding_dimension == self.embeddings.dimension
                            and previous.splitter_version == self.splitter.config.version
                        ):
                            skipped += 1
                            document_outcome["status"] = "skipped"
                            continue

                        indexed_at = datetime.now(UTC).isoformat()
                        enriched = document.model_copy(
                            update={
                                "metadata": {
                                    **document.metadata,
                                    "embedding_model": self.embeddings.model_name,
                                    "embedding_dimension": self.embeddings.dimension,
                                    "indexed_at": indexed_at,
                                }
                            }
                        )
                        chunks = self.splitter.split(enriched)
                        if not chunks:
                            raise ValueError("document produced no chunks")
                        vectors = self.embeddings.embed_documents(
                            [chunk.text for chunk in chunks]
                        )
                        if len(vectors) != len(chunks):
                            raise RuntimeError(
                                "embedding provider returned the wrong vector count"
                            )
                        state = DocumentIndexState(
                            document_id=document.document_id,
                            content_hash=content_hash,
                            embedding_model=self.embeddings.model_name,
                            embedding_dimension=self.embeddings.dimension,
                            splitter_version=self.splitter.config.version,
                            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
                        )
                        self.store.replace_document(state, chunks, vectors)
                        indexed += 1
                        removed = (
                            len(set(previous.chunk_ids) - set(state.chunk_ids))
                            if previous is not None
                            else 0
                        )
                        deleted_chunks += removed
                        document_outcome["chunk_count"] = len(chunks)
                        document_outcome["vector_count"] = len(vectors)
                        document_outcome["deleted_chunks"] = removed
                except Exception as error:
                    failures.append(
                        IndexFailure(
                            source=document.source,
                            error=f"{type(error).__name__}: {error}",
                        )
                    )

            index_outcome["indexed_count"] = indexed
            index_outcome["skipped_count"] = skipped
            index_outcome["deleted_chunks"] = deleted_chunks
            index_outcome["failed_count"] = len(failures)
            if failures:
                index_outcome["status"] = "partial_failure"
            return IndexingReport(
                indexed=indexed,
                skipped=skipped,
                deleted_chunks=deleted_chunks,
                failed=tuple(failures),
            )


__all__ = [
    "IndexFailure",
    "IndexingReport",
    "IngestionService",
    "load_source_documents",
]
