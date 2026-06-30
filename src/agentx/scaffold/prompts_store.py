"""Read/write helpers for a generated project's ``prompts.json``.

The generated project is *prompt-file driven*: ``prompts.json`` is the source of
truth for which agents exist and what their system prompts are. Editing it (by
hand or via ``agentx prompt``) changes the running project — no code edits needed.

Schema::

    {
      "with_rag": false,
      "agents": {
        "<name>": {"role": "...", "goal": "...", "system_prompt": "..."}
      }
    }
"""
from __future__ import annotations

import json
from pathlib import Path

from .spec import ProjectSpec, to_snake

FILENAME = "prompts.json"


def build_prompts_data(spec: ProjectSpec) -> dict:
    """Build the prompts.json payload from a ProjectSpec."""
    return {
        "with_rag": spec.use_rag,
        "agents": {
            a.name: {"role": a.role, "goal": a.goal, "system_prompt": a.system_prompt}
            for a in spec.agents
        },
    }


def write_prompts(target_dir: str | Path, spec: ProjectSpec) -> Path:
    path = Path(target_dir) / FILENAME
    path.write_text(json.dumps(build_prompts_data(spec), indent=2) + "\n", encoding="utf-8")
    return path


def find_prompts_file(start: str | Path | None = None) -> Path | None:
    """Walk up from ``start`` (or cwd) to find a project's prompts.json."""
    base = Path(start or Path.cwd()).resolve()
    for d in [base, *base.parents]:
        candidate = d / FILENAME
        if candidate.exists():
            return candidate
    return None


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.setdefault("agents", {})
    data.setdefault("with_rag", False)
    return data


def save(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def set_prompt(path: str | Path, name: str, text: str) -> dict:
    """Set/replace an existing agent's system prompt."""
    data = load(path)
    key = to_snake(name)
    if key not in data["agents"]:
        raise KeyError(f"No agent named '{key}'. Use `add` to create it.")
    data["agents"][key]["system_prompt"] = text
    save(path, data)
    return data


def add_agent(path: str | Path, name: str, role: str = "", goal: str = "", text: str = "") -> dict:
    """Add a new agent. The project picks it up automatically on next run."""
    data = load(path)
    key = to_snake(name)
    if key in data["agents"]:
        raise KeyError(f"Agent '{key}' already exists. Use `set` to change its prompt.")
    data["agents"][key] = {
        "role": role or f"{name} agent",
        "goal": goal or "Help the user accomplish their task accurately.",
        "system_prompt": text,
    }
    save(path, data)
    return data


def remove_agent(path: str | Path, name: str) -> dict:
    data = load(path)
    key = to_snake(name)
    if key not in data["agents"]:
        raise KeyError(f"No agent named '{key}'.")
    if len(data["agents"]) == 1:
        raise ValueError("Cannot remove the last remaining agent.")
    del data["agents"][key]
    save(path, data)
    return data
