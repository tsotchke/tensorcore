#!/usr/bin/env python3
"""Validate the cross-repository TensorCore full-capability campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "configs" / "full_capability_campaign.json"
DEFAULT_FRONTIER = ROOT / "configs" / "beyond_sota_research.json"
ALLOWED_STATES = {"verified", "implemented_limited", "in_progress", "planned", "blocked"}
ALLOWED_FINDING_SEVERITIES = {"critical", "high", "medium", "low"}
ALLOWED_FINDING_STATES = {
    "open", "implemented_uncommitted", "implemented_committed", "verified",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be an object")
    return value


def validate(campaign: dict[str, Any], frontier: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if campaign.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not campaign.get("campaign_id"):
        errors.append("campaign_id is required")
    if (campaign.get("control_plane") or {}).get("tool") != "ICC":
        errors.append("ICC must be the campaign control plane")

    repos = campaign.get("repositories")
    if not isinstance(repos, list):
        errors.append("repositories must be a list")
        repos = []
    repo_ids = [row.get("id") for row in repos if isinstance(row, dict)]
    if len(repo_ids) != len(set(repo_ids)):
        errors.append("repository ids must be unique")
    required_repos = {
        "tensorcore",
        "qllm",
        "computer_mesh",
        "quantum_geometric_tensor",
        "eshkol",
        "Selene",
    }
    missing_repos = sorted(required_repos - set(repo_ids))
    if missing_repos:
        errors.append(f"missing required repositories: {', '.join(missing_repos)}")

    waves = campaign.get("waves")
    if not isinstance(waves, list):
        errors.append("waves must be a list")
        waves = []
    wave_ids = [row.get("id") for row in waves if isinstance(row, dict)]
    if wave_ids != sorted(set(wave_ids)):
        errors.append("wave ids must be unique and sorted")
    wave_set = set(wave_ids)

    tasks = campaign.get("tasks")
    if not isinstance(tasks, list):
        errors.append("tasks must be a list")
        tasks = []
    task_rows = {row.get("id"): row for row in tasks if isinstance(row, dict) and row.get("id")}
    if len(task_rows) != len(tasks):
        errors.append("every task must have a unique non-empty id")

    for task_id, task in task_rows.items():
        owner = task.get("owner_repo")
        if owner not in repo_ids:
            errors.append(f"task {task_id}: unknown owner_repo {owner!r}")
        if task.get("status") not in ALLOWED_STATES:
            errors.append(f"task {task_id}: invalid status {task.get('status')!r}")
        if task.get("wave") not in wave_set:
            errors.append(f"task {task_id}: unknown wave {task.get('wave')!r}")
        acceptance = task.get("acceptance")
        if not isinstance(acceptance, list) or not acceptance or not all(
            isinstance(item, str) and item.strip() for item in acceptance
        ):
            errors.append(f"task {task_id}: acceptance must contain non-empty strings")
        dependencies = task.get("depends_on")
        if not isinstance(dependencies, list):
            errors.append(f"task {task_id}: depends_on must be a list")
            dependencies = []
        for dependency in dependencies:
            if dependency not in task_rows:
                errors.append(f"task {task_id}: unknown dependency {dependency!r}")
                continue
            dependency_wave = task_rows[dependency].get("wave")
            if isinstance(dependency_wave, int) and isinstance(task.get("wave"), int):
                if dependency_wave > task["wave"]:
                    errors.append(
                        f"task {task_id}: dependency {dependency!r} is in a later wave"
                    )
        if task.get("kind") == "consumer_adapter":
            if owner == "tensorcore":
                errors.append(f"task {task_id}: consumer adapters may not be owned by tensorcore")
            if not task.get("adapter_surface"):
                errors.append(f"task {task_id}: consumer adapter requires adapter_surface")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            errors.append(f"task dependency cycle includes {task_id!r}")
            return
        visiting.add(task_id)
        for dependency in task_rows[task_id].get("depends_on", []):
            if dependency in task_rows:
                visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in task_rows:
        visit(task_id)

    completion_policy = campaign.get("completion_policy")
    if not isinstance(completion_policy, dict):
        errors.append("completion_policy must be an object")
        completion_policy = {}
    definition_of_done = completion_policy.get("definition_of_done")
    if not isinstance(definition_of_done, list) or not definition_of_done or not all(
        isinstance(item, str) and item.strip() for item in definition_of_done
    ):
        errors.append("completion_policy.definition_of_done must contain non-empty strings")
    waiver_policy = completion_policy.get("waiver_policy")
    if not isinstance(waiver_policy, dict):
        errors.append("completion_policy.waiver_policy must be an object")
    hard_gates = completion_policy.get("hard_gates")
    if not isinstance(hard_gates, list) or not hard_gates:
        errors.append("completion_policy.hard_gates must be a non-empty list")
        hard_gates = []
    gate_ids = [row.get("id") for row in hard_gates if isinstance(row, dict)]
    if len(gate_ids) != len(hard_gates) or len(gate_ids) != len(set(gate_ids)):
        errors.append("every completion hard gate must have a unique non-empty id")
    for gate in hard_gates:
        if not isinstance(gate, dict):
            continue
        gate_id = gate.get("id", "<missing>")
        if gate.get("owner_repo") not in repo_ids:
            errors.append(f"completion hard gate {gate_id}: unknown owner_repo")
        if not isinstance(gate.get("command"), str) or not gate["command"].strip():
            errors.append(f"completion hard gate {gate_id}: command is required")
        evidence = gate.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            errors.append(f"completion hard gate {gate_id}: evidence is required")
        if gate.get("blocks_release") is not True:
            errors.append(f"completion hard gate {gate_id}: blocks_release must be true")

    release_task_id = completion_policy.get("release_task_id")
    if release_task_id not in task_rows:
        errors.append(f"completion_policy.release_task_id references unknown task {release_task_id!r}")
        release_task = None
    else:
        release_task = task_rows[release_task_id]
        if release_task.get("kind") != "acceptance":
            errors.append("completion_policy.release_task_id must reference an acceptance task")
        if release_task.get("owner_repo") != "tensorcore":
            errors.append("the TensorCore release task must be owned by tensorcore")

    closure_reset = campaign.get("closure_reset")
    if not isinstance(closure_reset, dict):
        errors.append("closure_reset must be an object")
        closure_reset = {}
    if not closure_reset.get("audit_id"):
        errors.append("closure_reset.audit_id is required")
    if not closure_reset.get("source_revision"):
        errors.append("closure_reset.source_revision is required")
    blocking_findings = closure_reset.get("blocking_findings")
    if not isinstance(blocking_findings, list) or not blocking_findings:
        errors.append("closure_reset.blocking_findings must be a non-empty list")
        blocking_findings = []
    finding_ids = [row.get("id") for row in blocking_findings if isinstance(row, dict)]
    if len(finding_ids) != len(blocking_findings) or len(finding_ids) != len(set(finding_ids)):
        errors.append("every closure finding must have a unique non-empty id")
    finding_tasks: set[str] = set()
    for finding in blocking_findings:
        if not isinstance(finding, dict):
            continue
        finding_id = finding.get("id", "<missing>")
        if finding.get("severity") not in ALLOWED_FINDING_SEVERITIES:
            errors.append(f"closure finding {finding_id}: invalid severity")
        if finding.get("status") not in ALLOWED_FINDING_STATES:
            errors.append(f"closure finding {finding_id}: invalid status")
        finding_task_id = finding.get("task_id")
        if finding_task_id not in task_rows:
            errors.append(f"closure finding {finding_id}: unknown task_id {finding_task_id!r}")
        else:
            finding_tasks.add(finding_task_id)
    if release_task is not None:
        missing_release_dependencies = sorted(
            finding_tasks - set(release_task.get("depends_on", [])) - {release_task_id}
        )
        if missing_release_dependencies:
            errors.append(
                "release task does not depend directly on closure finding tasks: "
                + ", ".join(missing_release_dependencies)
            )

    baseline = campaign.get("implemented_baseline")
    if not isinstance(baseline, list) or not baseline:
        errors.append("implemented_baseline must be a non-empty list")
    else:
        for row in baseline:
            baseline_id = row.get("id", "<missing>") if isinstance(row, dict) else "<invalid>"
            if not isinstance(row, dict):
                errors.append("implemented_baseline rows must be objects")
                continue
            if row.get("owner_repo") not in repo_ids:
                errors.append(f"baseline {baseline_id}: unknown owner_repo")
            if row.get("state") not in ALLOWED_STATES:
                errors.append(f"baseline {baseline_id}: invalid state")
            if not isinstance(row.get("evidence"), list) or not row["evidence"]:
                errors.append(f"baseline {baseline_id}: evidence is required")
            if not isinstance(row.get("limits"), list):
                errors.append(f"baseline {baseline_id}: limits must be a list")

    frontier_rows = frontier.get("gates")
    frontier_ids = {
        row.get("id") for row in frontier_rows or [] if isinstance(row, dict) and row.get("id")
    }
    ownership = campaign.get("research_gate_ownership")
    if not isinstance(ownership, list):
        errors.append("research_gate_ownership must be a list")
        ownership = []
    ownership_ids = [row.get("gate_id") for row in ownership if isinstance(row, dict)]
    if len(ownership_ids) != len(set(ownership_ids)):
        errors.append("research gate ownership ids must be unique")
    missing_ownership = sorted(frontier_ids - set(ownership_ids))
    extra_ownership = sorted(set(ownership_ids) - frontier_ids)
    if missing_ownership:
        errors.append(f"frontier gates without owners: {', '.join(missing_ownership)}")
    if extra_ownership:
        errors.append(f"unknown owned frontier gates: {', '.join(extra_ownership)}")
    for row in ownership:
        if not isinstance(row, dict):
            continue
        gate_id = row.get("gate_id", "<missing>")
        if row.get("owner_repo") not in repo_ids:
            errors.append(f"frontier gate {gate_id}: unknown owner_repo")
        contributors = row.get("contributors")
        if not isinstance(contributors, list):
            errors.append(f"frontier gate {gate_id}: contributors must be a list")
        elif any(repo not in repo_ids for repo in contributors):
            errors.append(f"frontier gate {gate_id}: unknown contributor")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        campaign = load_json(args.campaign)
        frontier = load_json(args.frontier)
        errors = validate(campaign, frontier)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors = [str(exc)]

    result = {
        "ok": not errors,
        "campaign": str(args.campaign),
        "frontier": str(args.frontier),
        "errors": errors,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif errors:
        for error in errors:
            print(f"full capability campaign error: {error}")
    else:
        print("full capability campaign OK")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
