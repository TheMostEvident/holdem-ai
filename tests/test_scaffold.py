"""Test that architecture generation is repeatable and non-destructive."""

from create_architecture import create_scaffold


def test_scaffold_creation_and_idempotency(tmp_path):
    root = tmp_path / "holdem-ai"
    created, existing = create_scaffold(root)
    assert created > 0
    assert existing == 0
    assert (root / "src/holdem_ai/environment/hand_engine.py").exists()
    assert (root / "src/holdem_ai/agents/base_agent.py").exists()

    readme = root / "README.md"
    readme.write_text("Keep my work", encoding="utf-8")
    created_again, existing_again = create_scaffold(root)
    assert created_again == 0
    assert existing_again == created
    assert readme.read_text(encoding="utf-8") == "Keep my work"
