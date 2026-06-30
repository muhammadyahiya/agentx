"""Filesystem-backed skill registry.

A *skill* is a named instruction block (e.g. "Always use the STAR method") that
gets injected into agent prompts. Skills are stored as JSON files under a
directory so users can add/version them outside the code.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") or "skill"


@dataclass
class Skill:
    slug: str
    name: str
    description: str
    instructions: str


class SkillRegistry:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, description: str, instructions: str) -> Skill:
        if not name.strip():
            raise ValueError("Skill name is required.")
        skill = Skill(_slug(name), name.strip(), description.strip(), instructions.strip())
        (self.dir / f"{skill.slug}.json").write_text(json.dumps(asdict(skill), indent=2), encoding="utf-8")
        return skill

    def list(self) -> list[Skill]:
        out: list[Skill] = []
        for fp in sorted(self.dir.glob("*.json")):
            try:
                out.append(Skill(**json.loads(fp.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001 - skip malformed files
                continue
        return out

    def delete(self, slug: str) -> None:
        (self.dir / f"{slug}.json").unlink(missing_ok=True)

    def combined_instructions(self, slugs: list[str] | None = None) -> str:
        """Concatenate selected (or all) skills' instructions for prompt injection."""
        skills = self.list()
        if slugs:
            wanted = set(slugs)
            skills = [s for s in skills if s.slug in wanted]
        if not skills:
            return ""
        return "\n".join(f"- {s.name}: {s.instructions}" for s in skills)


def get_skill_registry(directory: str | Path = "data/skills") -> SkillRegistry:
    return SkillRegistry(directory)
