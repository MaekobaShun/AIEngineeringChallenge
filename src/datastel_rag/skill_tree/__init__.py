"""Hand-crafted / folder-hierarchy skill trees for the skill_nav experiment.

This is NOT Corpus2Skill clustering. Nodes mirror the share-drive folder
layout (project -> phase -> file). Serve-time navigation uses list_children;
BM25 is disabled in skill_nav mode (forced toolset switch).
"""

from datastel_rag.skill_tree.tree import SkillTree, load_skill_tree

__all__ = ["SkillTree", "load_skill_tree"]
