"""One-time functional-package migration from Phase 5-16 into holdem_ai.

The code bodies and checkpoint formats are intentionally left unchanged.
Run only against a checkout of the refactor branch, never against main.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PHASES = range(5, 17)
MOVES = {
    (5, "betting"): "environment/bet_abstraction",
    (5, "policy"): "agents/policies",
    (5, "rewards"): "training/reward_models",
    (5, "rlcard_adapter"): "environment/rlcard_adapter",
    (5, "runner"): "training/episode_runner",
    (5, "schema"): "core/schema",
    (5, "snapshot"): "utils/snapshot",
    (6, "adapter"): "environment/hunl200_adapter",
    (6, "rollout"): "training/global_rollout",
    (7, "actions"): "encoding/actions",
    (7, "encoding"): "encoding/observation",
    (7, "tensors"): "encoding/tensors",
    (7, "vocabulary"): "encoding/vocabulary",
    (8, "network"): "models/transformer",
    (8, "policy"): "agents/neural_policy",
    (9, "ppo"): "training/ppo",
    (9, "trainer"): "training/ppo_session",
    (10, "boost"): "training/qboost",
    (10, "trainer"): "training/qboost_session",
    (11, "collector"): "training/batched_collector",
    (11, "evaluation"): "evaluation/balanced",
    (11, "progress"): "utils/progress",
    (11, "runner"): "experiments/time_budget",
    (12, "checkpoint"): "evaluation/checkpoint_loader",
    (12, "evaluation"): "evaluation/seed_blocked",
    (13, "diagnostics"): "evaluation/action_diagnostics",
    (13, "runner"): "experiments/batch_scaling",
    (14, "statistics"): "evaluation/seed_statistics",
    (14, "study"): "experiments/controlled_study",
    (15, "longrun"): "experiments/longrun",
    (16, "opponents"): "agents/diagnostic_opponents",
    (16, "diagnostics"): "evaluation/strategy_diagnostics",
    (16, "evaluation"): "evaluation/opponent_pool",
    (16, "report"): "evaluation/strategy_report",
}
DEST = {f"holdem_agent_phase{n}.{m}": "holdem_ai." + path.replace("/", ".")
        for (n, m), path in MOVES.items()}
PHASE_ROOT = re.compile(r"^holdem_agent_phase(5|6|7|8|9|10|11|12|13|14|15|16)(?:\.(.*))?$")
ALL_EXPORTS = {}
for n in PHASES:
    p = SRC / f"holdem_agent_phase{n}" / "__init__.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    exports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            target = DEST[f"holdem_agent_phase{n}.{node.module}"]
            for alias in node.names:
                if alias.name == "*":
                    raise RuntimeError(f"Unexpected star import in {p}")
                exports[alias.asname or alias.name] = target
    ALL_EXPORTS[n] = exports


def phase_module_name(source: Path) -> str | None:
    if source.parent.parent != SRC:
        return None
    match = re.fullmatch(r"holdem_agent_phase(5|6|7|8|9|10|11|12|13|14|15|16)",
                         source.parent.name)
    return f"holdem_agent_phase{match.group(1)}" if match else None


def rewrite_one(path: Path, old_parent: str | None) -> bool:
    original = path.read_text(encoding="utf-8")
    tree = ast.parse(original, filename=str(path))
    lines = original.splitlines(keepends=True)
    edits = []

    def collect_group(items, indent):
        groups = defaultdict(list)
        for destination, imported in items:
            groups[destination].append(imported)
        return ("\n" + indent).join(
            f"from {dest} import {', '.join(names)}"
            for dest, names in groups.items()
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                if not old_parent:
                    continue
                if node.level != 1:
                    raise RuntimeError(f"Unsupported relative import in {path}:{node.lineno}")
                old_module = old_parent + ("." + node.module if node.module else "")
            else:
                old_module = node.module or ""
            match = PHASE_ROOT.fullmatch(old_module)
            if not match:
                continue
            number = int(match.group(1))
            submod = match.group(2)
            items = []
            for alias in node.names:
                named = alias.name + (f" as {alias.asname}" if alias.asname else "")
                if submod:
                    if old_module not in DEST:
                        raise RuntimeError(f"Missing module mapping: {old_module} ({path})")
                    items.append((DEST[old_module], named))
                elif alias.name in ALL_EXPORTS[number]:
                    items.append((ALL_EXPORTS[number][alias.name], named))
                elif f"{old_module}.{alias.name}" in DEST:
                    destination = DEST[f"{old_module}.{alias.name}"]
                    prefix, newname = destination.rsplit(".", 1)
                    items.append((prefix, newname + (f" as {alias.asname}" if alias.asname else "")))
                else:
                    raise RuntimeError(f"Missing export: {old_module}.{alias.name} in {path}:{node.lineno}")
            indent = " " * node.col_offset
            replacement = indent + collect_group(items, indent) + "\n"
            edits.append((node.lineno - 1, node.end_lineno, replacement))
        elif isinstance(node, ast.Import):
            changed = False
            chunks = []
            for alias in node.names:
                match = PHASE_ROOT.fullmatch(alias.name)
                if match:
                    if alias.name not in DEST:
                        raise RuntimeError(f"Cannot rewrite package-level import {alias.name}: {path}")
                    name = DEST[alias.name]
                    changed = True
                else:
                    name = alias.name
                chunks.append(name + (f" as {alias.asname}" if alias.asname else ""))
            if changed:
                indent = " " * node.col_offset
                edits.append((node.lineno - 1, node.end_lineno,
                              indent + "import " + ", ".join(chunks) + "\n"))
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    transformed = "".join(lines)
    ast.parse(transformed, filename=str(path))
    if transformed != original:
        path.write_text(transformed, encoding="utf-8", newline="")
    return transformed != original


def main():
    old_files = {}
    for (number, module), target in MOVES.items():
        old = SRC / f"holdem_agent_phase{number}" / f"{module}.py"
        new = SRC / "holdem_ai" / (target + ".py")
        if not old.is_file() or new.exists():
            raise RuntimeError(f"Missing source or conflicting destination: {old} -> {new}")
        old_files[old] = new

    changed, relocated = [], []
    for old, new in old_files.items():
        rewrite_one(old, phase_module_name(old))
        new.parent.mkdir(parents=True, exist_ok=True)
        if not (new.parent / "__init__.py").exists():
            (new.parent / "__init__.py").write_text(
                '"""Consolidated functional package."""\n', encoding="utf-8")
        old.rename(new)
        relocated.append((old, new))

    for number in PHASES:
        package = SRC / f"holdem_agent_phase{number}"
        init = package / "__init__.py"
        if any(p != init for p in package.iterdir()):
            raise RuntimeError(f"Unexpected files still in {package}")
        init.unlink()
        package.rmdir()

    for root in (SRC / "holdem_ai", ROOT / "scripts", ROOT / "tests"):
        for file in root.rglob("*.py"):
            if rewrite_one(file, None):
                changed.append(file)

    for file in [ROOT / "main.py", *(SRC / "holdem_ai").rglob("*.py"),
                 *(ROOT / "scripts").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]:
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and PHASE_ROOT.fullmatch(node.module):
                raise RuntimeError(f"Old import remains: {file}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if PHASE_ROOT.fullmatch(alias.name):
                        raise RuntimeError(f"Old import remains: {file}:{node.lineno}")
    if len(relocated) != 34:
        raise RuntimeError(f"Unexpected relocated file count: {len(relocated)}")
    print(f"MIGRATION_PASS: {len(relocated)} implementation modules relocated")
    print(f"MIGRATION_PASS: {len(changed)} other files rewritten")
    print("MIGRATION_PASS: all old phase package directories removed")
    print("MIGRATION_PASS: Python AST imports no longer reference old phase packages")


if __name__ == "__main__":
    main()
