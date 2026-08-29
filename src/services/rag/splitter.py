"""文档切分的具体实现。

Document splitting implementations.
"""

import hashlib
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from harness.capabilities.rag import DocumentChunk, SourceDocument

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DEFAULT_SPLITTER_VERSION = "markdown-recursive-v1"


@dataclass(frozen=True, slots=True)
class TextSplitterConfig:
    chunk_size: int = 1_200
    chunk_overlap: int = 150
    version: str = DEFAULT_SPLITTER_VERSION

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk overlap must be non-negative and below chunk size")
        if not self.version:
            raise ValueError("splitter version must not be empty")


class DocumentSplitter:
    """Markdown/DOCX 标题优先、递归字符长度兜底的确定性切分器。"""

    def __init__(self, config: TextSplitterConfig | None = None) -> None:
        self.config = config or TextSplitterConfig()
        self._fallback = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
            keep_separator=True,
            length_function=len,
        )

    def split(self, document: SourceDocument) -> tuple[DocumentChunk, ...]:
        sections = self._sections(document)
        raw_chunks: list[tuple[str, str]] = []
        for section, text in sections:
            for chunk_text in self._fallback.split_text(text):
                normalized = chunk_text.strip()
                if normalized:
                    raw_chunks.append((section, normalized))

        chunks: list[DocumentChunk] = []
        content_hash = str(document.metadata["content_hash"])
        for index, (section, text) in enumerate(raw_chunks):
            digest = hashlib.sha256(
                f"{document.document_id}\0{content_hash}\0{index}\0{text}".encode()
            ).hexdigest()[:32]
            metadata = {
                **document.metadata,
                "source": document.source,
                "section": section,
                "chunk_id": digest,
                "chunk_index": index,
                "splitter_version": self.config.version,
            }
            chunks.append(
                DocumentChunk(
                    document_id=document.document_id,
                    chunk_id=digest,
                    text=text,
                    metadata=metadata,
                )
            )
        return tuple(chunks)

    @staticmethod
    def _sections(document: SourceDocument) -> tuple[tuple[str, str], ...]:
        if not document.source.casefold().endswith((".docx", ".md")):
            title = document.metadata.get("title", document.source)
            return ((str(title), document.text),)

        heading_stack: list[str] = []
        current_lines: list[str] = []
        current_section = str(document.metadata.get("title", document.source))
        sections: list[tuple[str, str]] = []

        def flush() -> None:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append((current_section, text))

        for line in document.text.splitlines():
            match = HEADING_PATTERN.match(line)
            if match is None:
                current_lines.append(line)
                continue
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            del heading_stack[level - 1 :]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(title)
            current_section = " > ".join(part for part in heading_stack if part)
            current_lines = [line]
        flush()
        if not sections:
            sections.append((current_section, document.text))
        return tuple(sections)


__all__ = ["DEFAULT_SPLITTER_VERSION", "DocumentSplitter", "TextSplitterConfig"]
