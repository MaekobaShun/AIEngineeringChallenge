"""Build a folder-hierarchy skill tree from the share-drive catalog.

Leaf summaries are deterministic structural pointers (sheet/column names,
heading hierarchy, function names, slide count, ...), computed from each
file's already-parsed content -- see pointer_summary.py for why this is
"pointers, not summaries" on purpose. Branch (phase/project/root) nodes keep
generic role descriptions; the real per-file signal lives at the leaves and
list_children already surfaces each child's summary when you visit its
parent, so there's nothing to aggregate upward.

An earlier version hand-wrote summaries for the ~7 files that
questions_valid.csv's hardest questions happened to need, which is exactly
the kind of thing the competition rules single out as disqualifying
("未知の案件・未知の資料が追加されても同じ処理方針で対応できる汎用的な設計
が求められる"): it would have inflated the valid30 score using knowledge of
the eval set, not generalized to test100 or the acceptance-review data. This
mirrors the existing directory layout; it does not embed/cluster like
Corpus2Skill.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from datastel_rag import config
from datastel_rag.catalog.glossary import Glossary, load_or_build_glossary
from datastel_rag.catalog.scanner import Catalog, load_or_build_catalog
from datastel_rag.skill_tree.pointer_summary import pointer_summary
from datastel_rag.skill_tree.tree import DEFAULT_TREE_PATH

_PHASE_HINTS: dict[str, str] = {
    "00.提案": "提案書・参考資料。評価指標や提案スコープの記載が多い。",
    "01.契約": "契約書(CT)。金額・条項・章立て。",
    "02.計画": "スケジュール/計画書(PL)。タスクID・マイルストーン・日程。",
    "03.データ": "学習データ・カラム説明。",
    "04.分析": "分析コード・ノートブック・metrics・図表。",
    "05.会議": "会議録・中間報告資料。",
    "06.報告書": "最終報告。目的/スコープ、残余リスク、結論。",
}


def build_tree(catalog: Catalog, glossary: Glossary, code_by_key: dict[str, str]) -> dict:
    nodes: dict[str, dict] = {}
    root_children: list[str] = []

    nodes["root"] = {
        "id": "root",
        "title": "共有ドライブ / プロジェクト",
        "kind": "branch",
        "summary": (
            "データアステル社の案件ルート。"
            "子は案件ノード（主略称つき）。質問の案件を選んで降りること。"
        ),
        "children": root_children,
        "parent_id": None,
    }

    for proj in sorted(catalog.projects, key=lambda p: code_by_key.get(p.key, p.key)):
        code = code_by_key.get(proj.key, "UNK")
        proj_id = f"proj:{code}"
        root_children.append(proj_id)

        by_phase: dict[str, list] = defaultdict(list)
        for f in proj.files:
            by_phase[f.phase or "other"].append(f)

        phase_ids: list[str] = []
        nodes[proj_id] = {
            "id": proj_id,
            "title": proj.key,
            "kind": "branch",
            "code": code,
            "summary": f"案件 {code}（{proj.key}）。フェーズ(00.提案〜06.報告書)に分かれる。",
            "children": phase_ids,
            "parent_id": "root",
        }

        for phase in sorted(by_phase.keys()):
            phase_id = f"phase:{code}:{phase}"
            phase_ids.append(phase_id)
            file_ids: list[str] = []
            nodes[phase_id] = {
                "id": phase_id,
                "title": phase,
                "kind": "branch",
                "code": code,
                "phase": phase,
                "summary": _PHASE_HINTS.get(phase, f"{code} の {phase} 配下ファイル。"),
                "children": file_ids,
                "parent_id": proj_id,
            }
            for f in sorted(by_phase[phase], key=lambda x: x.rel_path):
                name = Path(f.rel_path).name
                file_id = f"file:{code}:{phase}:{name}"
                if file_id in nodes:
                    file_id = f"file:{code}:{phase}:{name}#{len(file_ids)}"
                file_ids.append(file_id)
                nodes[file_id] = {
                    "id": file_id,
                    "title": name,
                    "kind": "leaf",
                    "code": code,
                    "phase": phase,
                    "rel_path": f.rel_path,
                    "summary": pointer_summary(f, catalog, glossary),
                    "children": [],
                    "parent_id": phase_id,
                }

    return {
        "meta": {
            "version": "folder_hierarchy_v1",
            "source": "share-drive catalog phases (not Corpus2Skill clustering)",
            "note": (
                "Leaf summaries are deterministic structural pointers (sheet/column names, "
                "headings, function names, ...) computed from each file's parsed content -- "
                "not LLM-generated gist summaries, and no per-file/per-question hand-tuning. "
                "Used as a navigation fallback (list_children) alongside BM25 search_documents "
                "in retrieval_mode=hybrid; retrieval_mode=skill_nav disables search entirely "
                "for isolated ablation testing."
            ),
        },
        "nodes": nodes,
    }


def main() -> None:
    catalog = load_or_build_catalog()
    glossary = load_or_build_glossary()
    code_by_key = {p.full_name: p.code for p in glossary.projects}
    tree = build_tree(catalog, glossary, code_by_key)
    out = DEFAULT_TREE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    n_leaf = sum(1 for n in tree["nodes"].values() if n.get("kind") == "leaf")
    print(f"wrote {out} nodes={len(tree['nodes'])} leaves={n_leaf}")


if __name__ == "__main__":
    main()
