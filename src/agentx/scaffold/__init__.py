"""Scaffolder: interactive wizard + project generator + prompt management."""
from .spec import AgentSpec, ProjectSpec
from .generator import GenerationResult, generate_project
from .wizard import run_wizard
from . import prompts_store

__all__ = [
    "AgentSpec",
    "ProjectSpec",
    "GenerationResult",
    "generate_project",
    "run_wizard",
    "prompts_store",
]
