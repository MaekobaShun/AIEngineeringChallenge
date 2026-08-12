"""Build a folder-hierarchy skill tree from the share-drive catalog.

MINAMINO (and a few other experiment targets) get richer hand-written
summaries; other nodes get filename/phase heuristics. This mirrors the
existing directory layout -- it does not embed/cluster like Corpus2Skill.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from datastel_rag import config
from datastel_rag.catalog.glossary import load_or_build_glossary
from datastel_rag.catalog.scanner import Catalog, load_or_build_catalog
from datastel_rag.skill_tree.tree import DEFAULT_TREE_PATH

# Hand-enriched leaf summaries for the skill_nav smoke set.
# Keys are path suffixes (matched against rel_path endswith / contains).
_LEAF_HINTS: dict[str, str] = {
    "みなみ野女性医療センター_最終報告.pdf": (
        "最終報告書。プロジェクト目的・スコープ、残余リスクと影響度、成果物の記載あり。"
        "『影響度が最も高い残余リスク』系の質問はここを読む。"
    ),
    "みなみ野女性医療センター/01.契約/契約書.docx": (
        "契約書本文。条項番号（例: 第8条）・秘密保持期間・金額条件など。"
    ),
    "みなみ野女性医療センター/02.計画/スケジュール.xlsx": (
        "プロジェクト計画書(PL)。マイルストーン・タスクID・日程。"
    ),
    "かえで総合病院/00.提案/提案書.pptx": (
        "提案書。重視する評価指標(Recall等)や提案内容の記載。"
    ),
    "白峰信用リスク評価株式会社_最終報告.pptx": (
        "最終報告書。プロジェクト目的とスコープ、API化の分類(対象/対象外)など。"
    ),
    "東都人材プラットフォーム/01.契約/契約書.docx": (
        "契約書(CT)。章立て・本業務の対象データ/前提/制約などの条項。"
    ),
    "青葉与信マネジメント株式会社/02.計画/スケジュール.xlsx": (
        "プロジェクト計画書(PL)。フェーズとタスクID(Txx)の対応。"
    ),
}

_PHASE_HINTS: dict[str, str] = {
    "00.提案": "提案書・参考資料。評価指標や提案スコープの記載が多い。",
    "01.契約": "契約書(CT)。金額・条項・章立て。",
    "02.計画": "スケジュール/計画書(PL)。タスクID・マイルストーン・日程。",
    "03.データ": "学習データ・カラム説明。",
    "04.分析": "分析コード・ノートブック・metrics・図表。",
    "05.会議": "会議録・中間報告資料。",
    "06.報告書": "最終報告。目的/スコープ、残余リスク、結論。",
}


def _leaf_summary(rel_path: str, name: str, phase: str) -> str:
    for key, hint in _LEAF_HINTS.items():
        if key in rel_path.replace("\\", "/"):
            return hint
    base = _PHASE_HINTS.get(phase, "")
    return f"{name}（{phase}）。{base}".strip()


def build_tree(catalog: Catalog, code_by_key: dict[str, str]) -> dict:
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
                    "summary": _leaf_summary(f.rel_path, name, phase),
                    "children": [],
                    "parent_id": phase_id,
                }

    return {
        "meta": {
            "version": "folder_hierarchy_v1",
            "source": "share-drive catalog phases (not Corpus2Skill clustering)",
            "route_policy": "forced_navigate_first",
            "note": (
                "skill_nav experiment: BM25 search_documents is disabled by toolset switch; "
                "prompt forces root→branch navigation. Results are NOT autonomous routing."
            ),
            "enriched_focus": ["MINAMINO", "KAEDE", "SHR", "TOTO", "AYM"],
        },
        "nodes": nodes,
    }


def main() -> None:
    catalog = load_or_build_catalog()
    glossary = load_or_build_glossary()
    code_by_key = {p.full_name: p.code for p in glossary.projects}
    tree = build_tree(catalog, code_by_key)
    out = DEFAULT_TREE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    n_leaf = sum(1 for n in tree["nodes"].values() if n.get("kind") == "leaf")
    print(f"wrote {out} nodes={len(tree['nodes'])} leaves={n_leaf}")


if __name__ == "__main__":
    main()
