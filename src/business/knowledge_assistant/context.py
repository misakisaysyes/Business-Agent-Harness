"""知识库和当前任务的安全 Context Provider。

Safe context provider for the knowledge base and current task.
"""

from collections.abc import Sequence
from pathlib import Path

from harness.context import ContextFragment
from harness.messages import MessageRole
from harness.state import AgentState


class KnowledgeAssistantContextProvider:
    """只暴露当前任务和授权知识资料的相对文件名。

    Expose only the current task and relative names of authorized knowledge files.
    """

    name = "knowledge_assistant_context"

    def __init__(
        self,
        knowledge_root: str | Path | Sequence[str | Path],
        max_files: int = 100,
    ) -> None:
        if max_files < 1:
            raise ValueError("max_files must be at least 1")
        roots = (
            (knowledge_root,) if isinstance(knowledge_root, str | Path) else tuple(knowledge_root)
        )
        self._knowledge_roots = tuple(dict.fromkeys(Path(root).resolve() for root in roots))
        if not self._knowledge_roots:
            raise ValueError("at least one knowledge root is required")
        self.max_files = max_files

    def provide(self, state: AgentState) -> tuple[ContextFragment, ...]:
        """返回当前任务及可访问资料名，不读取任何文件正文。

        Return the current task and accessible material names without reading file contents.
        """

        fragments: list[ContextFragment] = []
        current_task = self._current_task(state)
        if current_task is not None:
            fragments.append(
                ContextFragment(
                    key="current_task",
                    title="Current Task",
                    content=current_task,
                    priority=1000,
                )
            )

        files = self._allowed_material_names()
        materials = (
            "\n".join(f"- {name}" for name in files)
            if files
            else "No local knowledge materials are currently available."
        )
        fragments.append(
            ContextFragment(
                key="allowed_local_materials",
                title="Allowed Local Materials",
                content=materials,
                priority=500,
            )
        )
        return tuple(fragments)

    @staticmethod
    def _current_task(state: AgentState) -> str | None:
        for message in reversed(state["messages"]):
            if message.role is MessageRole.USER:
                content = message.content.strip()
                return content or None
        return None

    def _allowed_material_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        for root in self._knowledge_roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                if not path.is_file() or not path.resolve().is_relative_to(root):
                    continue
                names.add(relative.as_posix())
                if len(names) >= self.max_files:
                    return tuple(sorted(names))
        return tuple(sorted(names))


__all__ = ["KnowledgeAssistantContextProvider"]
