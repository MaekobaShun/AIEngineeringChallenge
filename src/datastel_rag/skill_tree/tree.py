"""Load and query a navigable skill tree (folder hierarchy as INDEX)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datastel_rag import config

DEFAULT_TREE_PATH = config.PIPELINE_ROOT / "skill_trees" / "folder_hierarchy_v1.json"


@dataclass
class SkillNode:
    id: str
    title: str
    summary: str = ""
    kind: str = "branch"  # branch | leaf
    children: list[str] | None = None
    rel_path: str | None = None
    code: str | None = None
    phase: str | None = None
    parent_id: str | None = None

    def to_public_dict(self) -> dict:
        d = {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "summary": self.summary,
        }
        if self.code:
            d["code"] = self.code
        if self.phase:
            d["phase"] = self.phase
        if self.rel_path:
            d["rel_path"] = self.rel_path
        if self.parent_id:
            d["parent_id"] = self.parent_id
        if self.kind == "branch":
            d["child_ids"] = list(self.children or [])
        return d


class SkillTree:
    def __init__(self, nodes: dict[str, SkillNode], meta: dict | None = None) -> None:
        self.nodes = nodes
        self.meta = meta or {}

    @classmethod
    def from_dict(cls, raw: dict) -> SkillTree:
        nodes: dict[str, SkillNode] = {}
        for nid, n in raw["nodes"].items():
            nodes[nid] = SkillNode(
                id=n["id"],
                title=n["title"],
                summary=n.get("summary", ""),
                kind=n.get("kind", "branch"),
                children=list(n.get("children") or []),
                rel_path=n.get("rel_path"),
                code=n.get("code"),
                phase=n.get("phase"),
                parent_id=n.get("parent_id"),
            )
        return cls(nodes=nodes, meta=raw.get("meta") or {})

    def get(self, node_id: str) -> SkillNode | None:
        return self.nodes.get(node_id)

    def list_children(self, node_id: str = "root") -> str:
        node = self.nodes.get(node_id)
        if node is None:
            return (
                f"ノードが見つかりません: {node_id!r}。"
                "list_children(node_id='root') からやり直してください。"
            )
        lines = [
            f"node_id={node.id}",
            f"title={node.title}",
            f"kind={node.kind}",
        ]
        if node.summary:
            lines.append(f"summary={node.summary}")
        if node.parent_id:
            lines.append(f"parent_id={node.parent_id}  # 行き止まりならここに戻る")
        if node.kind == "leaf":
            lines.append(f"rel_path={node.rel_path}")
            lines.append("このノードはリーフです。get_document(rel_path=...) で本文を取得してください。")
            return "\n".join(lines)

        children = [self.nodes[cid] for cid in (node.children or []) if cid in self.nodes]
        if not children:
            lines.append("子ノードはありません。parent_id に戻って別の枝を選んでください。")
            return "\n".join(lines)

        lines.append(f"children ({len(children)}):")
        for c in children:
            bits = [f"  - id={c.id}", f"title={c.title}", f"kind={c.kind}"]
            if c.code:
                bits.append(f"code={c.code}")
            if c.phase:
                bits.append(f"phase={c.phase}")
            if c.rel_path:
                bits.append(f"rel_path={c.rel_path}")
            if c.summary:
                bits.append(f"summary={c.summary}")
            lines.append(" | ".join(bits))
        lines.append(
            "次の操作: 関連しそうな child の id を list_children(node_id=...) に渡す。"
            "リーフなら get_document(rel_path=...) を呼ぶ。"
        )
        return "\n".join(lines)


def load_skill_tree(path: Path | None = None) -> SkillTree:
    p = path or DEFAULT_TREE_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    return SkillTree.from_dict(raw)
