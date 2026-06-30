"""Tests for the data-driven prompt store and post-generation editing."""
import json

import pytest

from agentx.scaffold import ProjectSpec, generate_project, prompts_store


def _gen(tmp_path):
    spec = ProjectSpec(name="pbot", provider="openai", framework="langgraph", create_venv=False)
    return generate_project(spec, tmp_path / "pbot", overwrite=True).target_dir


def test_find_and_load_prompts(tmp_path):
    root = _gen(tmp_path)
    found = prompts_store.find_prompts_file(root / "src" / "pbot")  # walks up
    assert found == root / "prompts.json"
    data = prompts_store.load(found)
    assert "assistant" in data["agents"]


def test_add_agent_then_present(tmp_path):
    root = _gen(tmp_path)
    pf = root / "prompts.json"
    prompts_store.add_agent(pf, "Critic", role="Critic", goal="Critique answers", text="You are a critic.")
    data = json.loads(pf.read_text())
    assert "critic" in data["agents"]
    assert data["agents"]["critic"]["system_prompt"] == "You are a critic."


def test_add_duplicate_raises(tmp_path):
    root = _gen(tmp_path)
    with pytest.raises(KeyError):
        prompts_store.add_agent(root / "prompts.json", "assistant")


def test_set_prompt(tmp_path):
    root = _gen(tmp_path)
    pf = root / "prompts.json"
    prompts_store.set_prompt(pf, "assistant", "Brand new prompt.")
    assert json.loads(pf.read_text())["agents"]["assistant"]["system_prompt"] == "Brand new prompt."


def test_set_unknown_agent_raises(tmp_path):
    root = _gen(tmp_path)
    with pytest.raises(KeyError):
        prompts_store.set_prompt(root / "prompts.json", "ghost", "x")


def test_remove_last_agent_blocked(tmp_path):
    root = _gen(tmp_path)
    with pytest.raises(ValueError):
        prompts_store.remove_agent(root / "prompts.json", "assistant")


def test_remove_agent_ok(tmp_path):
    root = _gen(tmp_path)
    pf = root / "prompts.json"
    prompts_store.add_agent(pf, "second")
    prompts_store.remove_agent(pf, "second")
    assert "second" not in json.loads(pf.read_text())["agents"]
