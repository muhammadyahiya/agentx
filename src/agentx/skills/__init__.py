"""User-extensible skills: reusable instruction blocks injected into prompts."""
from .registry import Skill, SkillRegistry, get_skill_registry

__all__ = ["Skill", "SkillRegistry", "get_skill_registry"]
