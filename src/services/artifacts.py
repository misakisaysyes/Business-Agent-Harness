"""报告、Tool 输出和文件 Artifact 的存储。

Storage for reports, tool outputs, and file artifacts.
"""

from dataclasses import dataclass
from pathlib import Path


class InvalidArtifactPathError(ValueError):
    """Artifact 路径逃逸了配置的根目录。

    Raised when an artifact path escapes the configured root directory.
    """


@dataclass(frozen=True)
class ArtifactWrite:
    """一次 Artifact 写入的结果。

    Result of one artifact write.
    """

    path: Path
    overwritten: bool


class ArtifactStore:
    """只允许在固定根目录内写入文本 Artifact。

    Write text artifacts only inside a fixed root directory.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, relative_path: str) -> Path:
        """解析并校验根目录内的相对路径。

        Resolve and validate a relative path within the artifact root.
        """

        requested = Path(relative_path)
        if requested.is_absolute():
            raise InvalidArtifactPathError("artifact path must be relative")

        target = (self.root / requested).resolve()
        if target == self.root or not target.is_relative_to(self.root):
            raise InvalidArtifactPathError("artifact path escapes the configured root")
        return target

    def write_text(
        self,
        relative_path: str,
        content: str,
        overwrite: bool = False,
    ) -> ArtifactWrite:
        """写入 UTF-8 文本；默认使用排他创建避免意外覆盖。

        Write UTF-8 text, using exclusive creation by default to prevent overwrites.
        """

        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()

        with target.open("w" if overwrite else "x", encoding="utf-8") as artifact_file:
            artifact_file.write(content)

        return ArtifactWrite(path=target, overwritten=existed)


__all__ = ["ArtifactStore", "ArtifactWrite", "InvalidArtifactPathError"]
