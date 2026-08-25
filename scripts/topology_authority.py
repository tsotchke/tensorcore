#!/usr/bin/env python3
"""Build a deterministic, drift-checked TensorCore topology snapshot.

The authority consumes snapshots instead of invoking vendor CLIs.  This keeps
the reconciliation path deterministic and cross-platform while allowing a
separate collector to capture each Tailscale profile and cloud account.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "topology_authority.json"
DEFAULT_SCHEDULER = ROOT / "configs" / "mesh_resources.json"
DECLARATION_SCHEMA = "computer_mesh.nodes.v1"
TAILSCALE_OBSERVATION_SCHEMA = "tensorcore.tailscale_profile_observation.v1"
CLOUD_OBSERVATION_SCHEMA = "tensorcore.cloud_inventory_observation.v1"
SCHEDULER_SCHEMA = "tensorcore.mesh_resources.v1"
TRACE_SCHEMA = "tensorcore.topology_authority_trace.v1"


class TopologyError(ValueError):
    """Raised when a topology input violates its source contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise TopologyError(f"{field} must be an RFC3339 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TopologyError(f"{field} must be an RFC3339 timestamp, got {value!r}") from exc
    if parsed.tzinfo is None:
        raise TopologyError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def require_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TopologyError(f"{field} must be a JSON object")
    return value


def require_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TopologyError(f"{field} must be a JSON list")
    return value


def normalized_alias(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    alias = value.strip().lower().rstrip(".")
    if not alias or "@" in alias:
        return None
    # IP addresses are observations, never stable topology identities.
    if re.fullmatch(r"[0-9a-f:.]+", alias) and (":" in alias or alias.count(".") == 3):
        return None
    return alias


def alias_variants(value: Any) -> set[str]:
    alias = normalized_alias(value)
    if alias is None:
        return set()
    variants = {alias}
    if "." in alias:
        variants.add(alias.split(".", 1)[0])
    return variants


def normalize_node_id(value: Any, policy: dict[str, Any], *, field: str) -> str:
    alias = normalized_alias(value)
    if alias is None:
        raise TopologyError(f"{field} must be a non-empty stable name, got {value!r}")
    pattern = str(policy.get("identity", {}).get("node_id_pattern") or r"^[a-z0-9][a-z0-9._-]*$")
    if re.fullmatch(pattern, alias) is None:
        raise TopologyError(f"{field} {alias!r} does not match canonical node pattern {pattern!r}")
    return alias


def scalar_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value).strip().lower() for value in values if str(value).strip()})


def normalize_declarations(
    payload: Any,
    source_id: str,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = require_object(payload, field=f"declaration source {source_id}")
    nodes = require_list(root.get("nodes"), field=f"declaration source {source_id}.nodes")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(nodes):
        row = require_object(raw, field=f"declaration source {source_id}.nodes[{index}]")
        node_id = normalize_node_id(row.get("name"), policy, field=f"{source_id}.nodes[{index}].name")
        if node_id in seen:
            raise TopologyError(f"declaration source {source_id} repeats node {node_id!r}")
        seen.add(node_id)
        aliases: set[str] = {node_id}
        for key in ("ssh_alias", "tailscale_name"):
            aliases.update(alias_variants(row.get(key)))
        for alias in row.get("extra_aliases") or []:
            aliases.update(alias_variants(alias))
        cloud = row.get("cloud") if isinstance(row.get("cloud"), dict) else {}
        aliases.update(alias_variants(cloud.get("instance")))
        compute = (
            row.get("capabilities", {}).get("compute", {})
            if isinstance(row.get("capabilities"), dict)
            else {}
        )
        if not isinstance(compute, dict):
            compute = {}
        roles = scalar_strings(row.get("roles"))
        explicit_lifecycle = str(row.get("lifecycle") or "").strip().lower()
        lifecycle = "retired" if explicit_lifecycle == "retired" or "retired" in roles else "active"
        cloud_identity = None
        if cloud:
            cloud_identity = {
                "provider": str(cloud.get("provider") or "").strip().lower(),
                "project": str(cloud.get("project") or "").strip(),
                "zone": str(cloud.get("zone") or "").strip(),
                "instance": str(cloud.get("instance") or "").strip().lower(),
            }
        output.append(
            {
                "id": node_id,
                "aliases": sorted(aliases),
                "source_id": source_id,
                "lifecycle": lifecycle,
                "os": str(row.get("os") or "unknown").strip().lower(),
                "arch": str(row.get("arch") or "unknown").strip().lower(),
                "roles": roles,
                "capabilities": {
                    "class": str(compute.get("class") or "").strip(),
                    "accelerators": scalar_strings(compute.get("accelerators")),
                    "gpu_count": int(compute.get("gpu_count") or 0),
                    "gpu_memory_mb": int(compute.get("gpu_memory_mb") or 0),
                },
                "cloud_identity": cloud_identity,
                "has_tailscale_identity": bool(normalized_alias(row.get("tailscale_name"))),
            }
        )
    return output, {
        "kind": "declarations",
        "id": source_id,
        "schema": str(root.get("schema") or DECLARATION_SCHEMA),
        "item_count": len(output),
        "input_sha256": canonical_sha256(payload),
    }


def tailscale_rows(status: dict[str, Any]) -> list[tuple[dict[str, Any], bool]]:
    rows: list[tuple[dict[str, Any], bool]] = []
    if isinstance(status.get("Self"), dict):
        rows.append((status["Self"], True))
    peers = status.get("Peer")
    if isinstance(peers, dict):
        rows.extend((row, False) for row in peers.values() if isinstance(row, dict))
    elif isinstance(peers, list):
        rows.extend((row, False) for row in peers if isinstance(row, dict))
    return rows


def normalize_tailscale_observation(
    payload: Any,
    source_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = require_object(payload, field=f"Tailscale source {source_id}")
    if root.get("schema") != TAILSCALE_OBSERVATION_SCHEMA:
        raise TopologyError(
            f"Tailscale source {source_id} must use schema {TAILSCALE_OBSERVATION_SCHEMA!r}"
        )
    embedded_id = str(root.get("profile_id") or "").strip()
    if embedded_id != source_id:
        raise TopologyError(
            f"Tailscale source id {source_id!r} does not match profile_id {embedded_id!r}"
        )
    observed_at = format_timestamp(parse_timestamp(root.get("observed_at"), field=f"{source_id}.observed_at"))
    status = require_object(root.get("status"), field=f"Tailscale source {source_id}.status")
    output = []
    for index, (row, is_self) in enumerate(tailscale_rows(status)):
        aliases: set[str] = set()
        for key in ("HostName", "DNSName", "ComputedName", "Name"):
            aliases.update(alias_variants(row.get(key)))
        if not aliases:
            raise TopologyError(f"Tailscale source {source_id} row {index} has no stable name")
        online_value = row.get("Online")
        if not isinstance(online_value, bool):
            online_value = row.get("Active")
        if not isinstance(online_value, bool) and is_self:
            online_value = str(status.get("BackendState") or "").lower() == "running"
        output.append(
            {
                "kind": "tailscale",
                "source_id": source_id,
                "observed_at": observed_at,
                "aliases": sorted(aliases),
                "online": online_value if isinstance(online_value, bool) else None,
                "is_self": is_self,
            }
        )
    return output, {
        "kind": "tailscale_profiles",
        "id": source_id,
        "schema": TAILSCALE_OBSERVATION_SCHEMA,
        "observed_at": observed_at,
        "item_count": len(output),
        "input_sha256": canonical_sha256(payload),
    }


def cloud_instances(root: dict[str, Any]) -> list[Any]:
    for key in ("instances", "items", "inventory"):
        if isinstance(root.get(key), list):
            return root[key]
    raise TopologyError("cloud observation requires an instances list")


def cloud_accelerators(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    raw = row.get("accelerators")
    if isinstance(raw, list):
        for value in raw:
            if isinstance(value, dict):
                values.append(str(value.get("type") or value.get("acceleratorType") or ""))
            else:
                values.append(str(value))
    guests = row.get("guestAccelerators")
    if isinstance(guests, list):
        for value in guests:
            if isinstance(value, dict):
                values.append(str(value.get("acceleratorType") or value.get("type") or ""))
    return sorted({value.strip().lower().split("/")[-1] for value in values if value.strip()})


def normalize_cloud_observation(
    payload: Any,
    source_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = require_object(payload, field=f"cloud source {source_id}")
    if root.get("schema") != CLOUD_OBSERVATION_SCHEMA:
        raise TopologyError(f"cloud source {source_id} must use schema {CLOUD_OBSERVATION_SCHEMA!r}")
    embedded_id = str(root.get("inventory_id") or "").strip()
    if embedded_id != source_id:
        raise TopologyError(f"cloud source id {source_id!r} does not match inventory_id {embedded_id!r}")
    provider = str(root.get("provider") or "").strip().lower()
    if not provider:
        raise TopologyError(f"cloud source {source_id} requires provider")
    observed_at = format_timestamp(parse_timestamp(root.get("observed_at"), field=f"{source_id}.observed_at"))
    output = []
    for index, raw in enumerate(cloud_instances(root)):
        row = require_object(raw, field=f"cloud source {source_id}.instances[{index}]")
        name = str(row.get("name") or row.get("instance") or row.get("id") or "").strip()
        aliases = alias_variants(name.split("/")[-1])
        if not aliases:
            raise TopologyError(f"cloud source {source_id} instance {index} has no stable name")
        status = str(row.get("status") or row.get("state") or "unknown").strip().lower()
        if status in {"running", "active", "started", "ready"}:
            online: bool | None = True
        elif status in {"terminated", "stopped", "deallocated", "deleted", "suspended"}:
            online = False
        else:
            online = None
        output.append(
            {
                "kind": "cloud",
                "source_id": source_id,
                "provider": provider,
                "observed_at": observed_at,
                "aliases": sorted(aliases),
                "online": online,
                "provider_state": status,
                "accelerators": cloud_accelerators(row),
            }
        )
    return output, {
        "kind": "cloud_inventories",
        "id": source_id,
        "schema": CLOUD_OBSERVATION_SCHEMA,
        "provider": provider,
        "observed_at": observed_at,
        "item_count": len(output),
        "input_sha256": canonical_sha256(payload),
    }


def normalize_scheduler_inventory(
    payload: Any,
    source_id: str,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = require_object(payload, field=f"scheduler source {source_id}")
    if root.get("schema") != SCHEDULER_SCHEMA:
        raise TopologyError(f"scheduler source {source_id} must use schema {SCHEDULER_SCHEMA!r}")
    resources = require_list(root.get("resources"), field=f"scheduler source {source_id}.resources")
    output = []
    seen: set[str] = set()
    for index, raw in enumerate(resources):
        row = require_object(raw, field=f"scheduler source {source_id}.resources[{index}]")
        resource_id = str(row.get("id") or "").strip().lower()
        if not resource_id or resource_id in seen:
            raise TopologyError(f"scheduler source {source_id} has empty or duplicate resource id {resource_id!r}")
        seen.add(resource_id)
        node_id = normalize_node_id(row.get("node"), policy, field=f"{source_id}.{resource_id}.node")
        status = str(row.get("status") or "active").strip().lower()
        if status not in {"active", "reserved", "blocked"}:
            raise TopologyError(f"scheduler resource {resource_id} has unsupported status {status!r}")
        output.append(
            {
                "id": resource_id,
                "source_id": source_id,
                "node_alias": node_id,
                "backend": str(row.get("backend") or "").strip().lower(),
                "class": str(row.get("class") or "").strip(),
                "capacity": int(row.get("capacity") or 1),
                "status": status,
                "general_queue_eligible": bool(row.get("general_queue_eligible", True)),
                "control_plane": str(row.get("control_plane") or "").strip(),
            }
        )
    return output, {
        "kind": "scheduler_inventories",
        "id": source_id,
        "schema": SCHEDULER_SCHEMA,
        "item_count": len(output),
        "input_sha256": canonical_sha256(payload),
    }


def add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    detail: str,
    *,
    source_kind: str = "",
    source_id: str = "",
    record_id: str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "detail": detail,
            "source_kind": source_kind,
            "source_id": source_id,
            "record_id": record_id,
        }
    )


def source_coverage_issues(policy: dict[str, Any], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    by_kind: dict[str, set[str]] = {}
    for source in sources:
        by_kind.setdefault(str(source["kind"]), set()).add(str(source["id"]))
    for kind, raw_contract in policy.get("required_sources", {}).items():
        contract = raw_contract if isinstance(raw_contract, dict) else {}
        required = {str(value) for value in contract.get("ids") or []}
        actual = by_kind.get(str(kind), set())
        minimum = int(contract.get("minimum_count") or 0)
        for missing in sorted(required - actual):
            add_issue(
                issues,
                "error",
                "missing_required_source",
                f"required {kind} source {missing!r} was not supplied",
                source_kind=str(kind),
                source_id=missing,
            )
        if len(actual) < minimum:
            add_issue(
                issues,
                "error",
                "source_count_below_minimum",
                f"{kind} supplied {len(actual)} sources but requires at least {minimum}",
                source_kind=str(kind),
            )
        if contract.get("completeness") == "exact_manifest":
            for unexpected in sorted(actual - required):
                add_issue(
                    issues,
                    "error",
                    "unmanifested_source",
                    f"{kind} source {unexpected!r} is not in the exact source manifest",
                    source_kind=str(kind),
                    source_id=unexpected,
                )
    return issues


def freshness(
    observed_at: str,
    kind: str,
    policy: dict[str, Any],
    as_of: datetime,
) -> tuple[str, float]:
    observed = parse_timestamp(observed_at, field=f"{kind}.observed_at")
    age = (as_of - observed).total_seconds()
    rules = policy.get("freshness", {})
    future_skew = float(rules.get("future_clock_skew_seconds") or 0)
    if age < -future_skew:
        return "future", age
    key = "tailscale_max_age_seconds" if kind == "tailscale" else "cloud_max_age_seconds"
    maximum = float(rules.get(key) or 0)
    return ("fresh" if age <= maximum else "stale"), age


def build_alias_map(
    declarations: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for node in declarations:
        for alias in node["aliases"]:
            aliases.setdefault(alias, set()).add(node["id"])
    for alias, node_ids in sorted(aliases.items()):
        if len(node_ids) > 1:
            add_issue(
                issues,
                "error",
                "declared_alias_conflict",
                f"alias {alias!r} maps to multiple nodes {sorted(node_ids)!r}",
                source_kind="declarations",
                record_id=alias,
            )
    return aliases


def observation_record(
    observation: dict[str, Any],
    state: str,
    age_seconds: float,
) -> dict[str, Any]:
    return {
        "kind": observation["kind"],
        "source_id": observation["source_id"],
        "observed_at": observation["observed_at"],
        "freshness": state,
        "age_seconds": round(age_seconds, 3),
        "online": observation.get("online"),
        "aliases": observation["aliases"],
        "provider_state": observation.get("provider_state"),
        "accelerators": observation.get("accelerators") or [],
    }


def resource_backend_supported(resource: dict[str, Any], node: dict[str, Any]) -> bool:
    backend = resource["backend"]
    if backend in {"", "service"}:
        return True
    accelerators = set(node.get("capabilities", {}).get("accelerators") or [])
    aliases = {"rocm": "hip", "hip": "rocm"}
    return backend in accelerators or aliases.get(backend) in accelerators


def attach_integrity(payload: dict[str, Any], signing_key: str | bytes | None) -> dict[str, Any]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("integrity", None)
    encoded = canonical_json(unsigned).encode("utf-8")
    integrity: dict[str, Any] = {
        "digest_algorithm": "sha256",
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    if signing_key is not None:
        key = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
        integrity["signature"] = {
            "algorithm": "hmac-sha256",
            "value": hmac.new(key, encoded, hashlib.sha256).hexdigest(),
        }
    payload["integrity"] = integrity
    return payload


def verify_snapshot(
    payload: Any,
    signing_key: str | bytes | None = None,
    *,
    require_signature: bool = False,
) -> tuple[bool, str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("integrity"), dict):
        return False, "missing_integrity"
    integrity = payload["integrity"]
    unsigned = copy.deepcopy(payload)
    unsigned.pop("integrity", None)
    encoded = canonical_json(unsigned).encode("utf-8")
    expected_digest = hashlib.sha256(encoded).hexdigest()
    if not hmac.compare_digest(str(integrity.get("payload_sha256") or ""), expected_digest):
        return False, "digest_mismatch"
    signature = integrity.get("signature")
    if signature is None:
        return (False, "missing_signature") if require_signature else (True, "digest_valid")
    if signing_key is None:
        return (False, "signature_key_required") if require_signature else (True, "digest_valid_signature_unchecked")
    if not isinstance(signature, dict) or signature.get("algorithm") != "hmac-sha256":
        return False, "unsupported_signature"
    key = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
    expected_signature = hmac.new(key, encoded, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(signature.get("value") or ""), expected_signature):
        return False, "signature_mismatch"
    return True, "signature_valid"


def public_snapshot(private: dict[str, Any], policy: dict[str, Any], signing_key: str | bytes | None) -> dict[str, Any]:
    nodes = []
    for node in private["nodes"]:
        nodes.append(
            {
                "id": node["id"],
                "declared": node["declared"],
                "lifecycle": node["lifecycle"],
                "os": node["os"],
                "arch": node["arch"],
                "capabilities": node["capabilities"],
                "state": node["state"],
                "observation_sources": sorted(
                    {f"{item['kind']}:{item['source_id']}" for item in node["observations"]}
                ),
            }
        )
    issues = [
        {
            "severity": item["severity"],
            "code": item["code"],
            "source_kind": item["source_kind"],
            "source_id": item["source_id"],
            "record_id_sha256": hashlib.sha256(str(item["record_id"]).encode("utf-8")).hexdigest()
            if item["record_id"]
            else "",
        }
        for item in private["drift"]["issues"]
    ]
    result = {
        "schema": policy["public_snapshot_schema"],
        "generated_at": private["generated_at"],
        "policy_sha256": private["policy_sha256"],
        "source_coverage": [
            {
                "kind": source["kind"],
                "id": source["id"],
                "schema": source["schema"],
                "observed_at": source.get("observed_at"),
                "item_count": source["item_count"],
            }
            for source in private["sources"]
        ],
        "nodes": nodes,
        "resources": private["resources"],
        "drift": {
            "clean": private["drift"]["clean"],
            "error_count": private["drift"]["error_count"],
            "warning_count": private["drift"]["warning_count"],
            "issues": issues,
        },
    }
    return attach_integrity(result, signing_key)


def forbidden_public_paths(payload: Any, policy: dict[str, Any]) -> list[str]:
    rules = policy.get("public_redaction", {})
    fragments = [str(value).lower() for value in rules.get("deny_key_fragments") or []]
    exact = {str(value).lower() for value in rules.get("deny_exact_keys") or []}
    found: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                child_path = f"{path}.{key}" if path else str(key)
                if lowered in exact or any(fragment in lowered for fragment in fragments):
                    found.append(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")
    return sorted(found)


def reconcile(
    policy: dict[str, Any],
    declarations: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    as_of: datetime,
    signing_key: str | bytes | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    issues = source_coverage_issues(policy, sources)
    alias_map = build_alias_map(declarations, issues)
    node_records: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        if declaration["id"] in node_records:
            add_issue(
                issues,
                "error",
                "duplicate_canonical_node",
                f"node {declaration['id']!r} appears in multiple declaration sources",
                source_kind="declarations",
                source_id=declaration["source_id"],
                record_id=declaration["id"],
            )
            continue
        node_records[declaration["id"]] = {
            **declaration,
            "declared": True,
            "observations": [],
            "identity_conflict": False,
        }

    unmanaged_by_alias: dict[str, str] = {}
    for observation in observations:
        matches: set[str] = set()
        for alias in observation["aliases"]:
            matches.update(alias_map.get(alias, set()))
            if alias in unmanaged_by_alias:
                matches.add(unmanaged_by_alias[alias])
        state, age = freshness(observation["observed_at"], observation["kind"], policy, as_of)
        if state == "future":
            add_issue(
                issues,
                "error",
                "observation_from_future",
                f"{observation['kind']} observation exceeds allowed future clock skew",
                source_kind=observation["kind"],
                source_id=observation["source_id"],
            )
        if len(matches) > 1:
            add_issue(
                issues,
                "error",
                "observation_identity_conflict",
                f"observation aliases map to multiple canonical nodes {sorted(matches)!r}",
                source_kind=observation["kind"],
                source_id=observation["source_id"],
            )
            for node_id in matches:
                if node_id in node_records:
                    node_records[node_id]["identity_conflict"] = True
            continue
        if not matches:
            seed = f"{observation['kind']}:{observation['source_id']}:{observation['aliases'][0]}"
            node_id = f"unmanaged-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"
            node_records[node_id] = {
                "id": node_id,
                "aliases": list(observation["aliases"]),
                "source_id": "",
                "lifecycle": "active",
                "os": "unknown",
                "arch": "unknown",
                "roles": [],
                "capabilities": {
                    "class": "unmanaged-observation",
                    "accelerators": observation.get("accelerators") or [],
                    "gpu_count": 0,
                    "gpu_memory_mb": 0,
                },
                "cloud_identity": None,
                "has_tailscale_identity": observation["kind"] == "tailscale",
                "declared": False,
                "observations": [],
                "identity_conflict": False,
            }
            for alias in observation["aliases"]:
                unmanaged_by_alias[alias] = node_id
            severity = "error" if observation.get("accelerators") else "warning"
            code = "unmanaged_accelerator" if severity == "error" else "unmanaged_observation"
            add_issue(
                issues,
                severity,
                code,
                "an observed identity has no computer_mesh declaration",
                source_kind=observation["kind"],
                source_id=observation["source_id"],
                record_id=node_id,
            )
        else:
            node_id = next(iter(matches))
        node_records[node_id]["observations"].append(observation_record(observation, state, age))

    # Explicit cloud declarations must appear in the complete cloud inventory.
    for node in node_records.values():
        cloud_identity = node.get("cloud_identity")
        if not node["declared"] or not cloud_identity or not cloud_identity.get("instance"):
            continue
        cloud_alias = normalized_alias(cloud_identity["instance"])
        found = any(
            item["kind"] == "cloud" and cloud_alias in set(item.get("aliases") or [])
            for item in node["observations"]
        )
        if not found:
            severity = "warning" if node["lifecycle"] == "retired" else "error"
            add_issue(
                issues,
                severity,
                "declared_cloud_instance_missing",
                f"declared cloud instance for node {node['id']!r} is absent from cloud inventory",
                source_kind="cloud",
                record_id=node["id"],
            )

    for node in node_records.values():
        observations_for_node = node["observations"]
        has_fresh = any(item["freshness"] == "fresh" for item in observations_for_node)
        has_stale = any(item["freshness"] == "stale" for item in observations_for_node)
        online = any(item["freshness"] == "fresh" and item["online"] is True for item in observations_for_node)
        offline = has_fresh and not online and any(item["online"] is False for item in observations_for_node)
        observation_state = "fresh" if has_fresh else "stale" if has_stale else "unknown"
        connectivity = "online" if online else "offline" if offline else "unknown"
        if node["lifecycle"] == "retired":
            admission = "blocked"
        elif node["identity_conflict"] or not node["declared"]:
            admission = "quarantined"
        elif observation_state != "fresh" or connectivity != "online":
            admission = "quarantined"
        else:
            admission = "eligible"
        node["state"] = {
            "observation": observation_state,
            "connectivity": connectivity,
            "admission": admission,
        }

    reconciled_resources = []
    for resource in resources:
        matches = alias_map.get(resource["node_alias"], set())
        node = node_records.get(next(iter(matches))) if len(matches) == 1 else None
        authority_state = "blocked" if resource["status"] == "blocked" else "reserved" if resource["status"] == "reserved" else "eligible"
        reason = "scheduler_status"
        if node is None:
            authority_state = "quarantined"
            reason = "unknown_or_conflicting_node"
            add_issue(
                issues,
                "error" if resource["status"] != "blocked" else "warning",
                "scheduler_resource_unknown_node",
                f"resource {resource['id']!r} references unknown node {resource['node_alias']!r}",
                source_kind="scheduler",
                source_id=resource["source_id"],
                record_id=resource["id"],
            )
        elif not resource_backend_supported(resource, node):
            authority_state = "quarantined"
            reason = "backend_capability_mismatch"
            add_issue(
                issues,
                "error",
                "scheduler_backend_capability_mismatch",
                f"resource {resource['id']!r} backend {resource['backend']!r} is absent from node capabilities",
                source_kind="scheduler",
                source_id=resource["source_id"],
                record_id=resource["id"],
            )
        elif resource["status"] == "active" and node["state"]["admission"] != "eligible":
            authority_state = "quarantined"
            reason = f"node_{node['state']['observation']}_{node['state']['connectivity']}"
            add_issue(
                issues,
                "error",
                "active_resource_not_observed_online",
                f"active resource {resource['id']!r} is not backed by a fresh online node",
                source_kind="scheduler",
                source_id=resource["source_id"],
                record_id=resource["id"],
            )
        reconciled_resources.append(
            {
                "id": resource["id"],
                "authority_node_id": node["id"] if node else None,
                "backend": resource["backend"],
                "class": resource["class"],
                "capacity": resource["capacity"],
                "scheduler_status": resource["status"],
                "authority_admission": authority_state,
                "authority_reason": reason,
            }
        )

    issues.sort(key=lambda item: (item["severity"], item["code"], item["source_kind"], item["source_id"], item["record_id"]))
    errors = sum(item["severity"] == "error" for item in issues)
    warnings = sum(item["severity"] == "warning" for item in issues)
    snapshot = {
        "schema": policy["canonical_snapshot_schema"],
        "generated_at": format_timestamp(as_of),
        "policy_sha256": canonical_sha256(policy),
        "sources": sorted(sources, key=lambda item: (item["kind"], item["id"])),
        "nodes": sorted(node_records.values(), key=lambda item: item["id"]),
        "resources": sorted(reconciled_resources, key=lambda item: item["id"]),
        "drift": {
            "clean": errors == 0,
            "error_count": errors,
            "warning_count": warnings,
            "issues": issues,
        },
    }
    attach_integrity(snapshot, signing_key)
    public = public_snapshot(snapshot, policy, signing_key)
    forbidden = forbidden_public_paths(public, policy)
    if forbidden:
        raise TopologyError(f"public snapshot contains forbidden keys: {forbidden!r}")
    return snapshot, public


def parse_source_paths(values: list[str], *, option: str) -> list[tuple[str, Path]]:
    output = []
    seen: set[str] = set()
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        source_id = source_id.strip()
        if not separator or not source_id or not raw_path.strip():
            raise TopologyError(f"{option} values must use SOURCE_ID=PATH")
        if source_id in seen:
            raise TopologyError(f"{option} repeats source id {source_id!r}")
        seen.add(source_id)
        output.append((source_id, Path(raw_path).expanduser()))
    return output


def signing_key_from_policy(policy: dict[str, Any]) -> str | None:
    variable = str(policy.get("integrity", {}).get("signature_key_environment") or "")
    if not variable:
        return None
    direct = os.environ.get(variable)
    if direct:
        return direct
    raw_path = os.environ.get(f"{variable}_FILE", "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.stat().st_mode & 0o077:
        raise TopologyError(f"{variable}_FILE must not be group/world accessible")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise TopologyError(f"{variable}_FILE is empty")
    return value


def emit_trace(path: str | Path, snapshot: dict[str, Any], public: dict[str, Any], *, signed: bool) -> None:
    gates = {
        "canonical_schema": snapshot.get("schema") == "tensorcore.topology_snapshot.v1",
        "source_coverage": not any(
            item["code"] in {"missing_required_source", "source_count_below_minimum", "unmanifested_source"}
            for item in snapshot["drift"]["issues"]
        ),
        "identity_reconciliation": not any("identity_conflict" in item["code"] for item in snapshot["drift"]["issues"]),
        "drift_fail_closed": all(
            resource["authority_admission"] != "eligible"
            or resource["scheduler_status"] == "active"
            for resource in snapshot["resources"]
        ),
        "scheduler_binding": snapshot["drift"]["clean"],
        "signed_snapshot": signed,
        "public_redaction": bool(public.get("integrity")),
    }
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        canonical_json(
            {
                "schema": TRACE_SCHEMA,
                "kind": "topology_authority_gate",
                "name": name,
                "value": "PASS" if passed else "FAIL",
                "status": "PASS" if passed else "FAIL",
                "snippet": f"topology authority gate {name}: {'passed' if passed else 'failed'}",
            }
        )
        for name, passed in gates.items()
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reconcile_parser = subparsers.add_parser("reconcile", help="build private and public topology snapshots")
    reconcile_parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    reconcile_parser.add_argument("--declarations", action="append", default=[], metavar="ID=PATH")
    reconcile_parser.add_argument("--tailscale-profile", action="append", default=[], metavar="ID=PATH")
    reconcile_parser.add_argument("--cloud-inventory", action="append", default=[], metavar="ID=PATH")
    reconcile_parser.add_argument(
        "--scheduler-inventory",
        action="append",
        default=[],
        metavar="ID=PATH",
        help=f"defaults to tensorcore={DEFAULT_SCHEDULER}",
    )
    reconcile_parser.add_argument("--as-of", help="RFC3339 reconciliation time; defaults to current UTC")
    reconcile_parser.add_argument("--private-output", required=True)
    reconcile_parser.add_argument("--public-output", required=True)
    reconcile_parser.add_argument("--icc-trace")
    reconcile_parser.add_argument("--require-clean", action="store_true")
    reconcile_parser.add_argument("--require-signature", action="store_true")

    verify_parser = subparsers.add_parser("verify", help="verify snapshot digest and optional HMAC signature")
    verify_parser.add_argument("snapshot")
    verify_parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    verify_parser.add_argument("--require-signature", action="store_true")
    verify_parser.add_argument("--json", action="store_true")
    return parser


def reconcile_command(args: argparse.Namespace) -> int:
    policy = require_object(load_json(args.policy), field="policy")
    if policy.get("schema") != "tensorcore.topology_authority_policy.v1":
        raise TopologyError("policy schema must be tensorcore.topology_authority_policy.v1")
    declaration_specs = parse_source_paths(args.declarations, option="--declarations")
    tailscale_specs = parse_source_paths(args.tailscale_profile, option="--tailscale-profile")
    cloud_specs = parse_source_paths(args.cloud_inventory, option="--cloud-inventory")
    scheduler_values = args.scheduler_inventory or [f"tensorcore={DEFAULT_SCHEDULER}"]
    scheduler_specs = parse_source_paths(scheduler_values, option="--scheduler-inventory")
    declarations: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for source_id, path in declaration_specs:
        rows, source = normalize_declarations(load_json(path), source_id, policy)
        declarations.extend(rows)
        sources.append(source)
    for source_id, path in tailscale_specs:
        rows, source = normalize_tailscale_observation(load_json(path), source_id)
        observations.extend(rows)
        sources.append(source)
    for source_id, path in cloud_specs:
        rows, source = normalize_cloud_observation(load_json(path), source_id)
        observations.extend(rows)
        sources.append(source)
    for source_id, path in scheduler_specs:
        rows, source = normalize_scheduler_inventory(load_json(path), source_id, policy)
        resources.extend(rows)
        sources.append(source)
    as_of = parse_timestamp(args.as_of, field="--as-of") if args.as_of else datetime.now(timezone.utc)
    signing_key = signing_key_from_policy(policy)
    if args.require_signature and not signing_key:
        raise TopologyError("signature required but topology signing key environment variable is empty")
    private, public = reconcile(
        policy,
        declarations,
        observations,
        resources,
        sources,
        as_of=as_of,
        signing_key=signing_key,
    )
    write_json(args.private_output, private)
    write_json(args.public_output, public)
    if args.icc_trace:
        emit_trace(args.icc_trace, private, public, signed=bool(signing_key))
    print(
        json.dumps(
            {
                "ok": private["drift"]["clean"],
                "schema": private["schema"],
                "nodes": len(private["nodes"]),
                "resources": len(private["resources"]),
                "errors": private["drift"]["error_count"],
                "warnings": private["drift"]["warning_count"],
                "signed": bool(private["integrity"].get("signature")),
                "private_output": str(args.private_output),
                "public_output": str(args.public_output),
            },
            sort_keys=True,
        )
    )
    return 2 if args.require_clean and not private["drift"]["clean"] else 0


def verify_command(args: argparse.Namespace) -> int:
    policy = require_object(load_json(args.policy), field="policy")
    key = signing_key_from_policy(policy)
    ok, reason = verify_snapshot(load_json(args.snapshot), key, require_signature=args.require_signature)
    if args.json:
        print(json.dumps({"ok": ok, "reason": reason, "snapshot": args.snapshot}, sort_keys=True))
    else:
        print(f"topology snapshot {'OK' if ok else 'FAIL'}: {reason}: {args.snapshot}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "reconcile":
            return reconcile_command(args)
        if args.command == "verify":
            return verify_command(args)
    except (OSError, json.JSONDecodeError, TopologyError) as exc:
        print(f"topology authority error: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
