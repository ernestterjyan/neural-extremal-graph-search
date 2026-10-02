"""Audited descriptive reports that retain failures and incomplete panels.

No solver or learned model is run. This report never replaces the preregistered
paired primary analysis with a favorable descriptive average.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
from collections import Counter
from pathlib import Path

AUDITOR_PATH = Path(__file__).with_name("c4_repair_audit.py")
if Path(__file__).stem.endswith("_report"):
    companion = Path(__file__).with_name(
        Path(__file__).stem.removesuffix("_report") + "_auditor.py"
    )
    if companion.exists():
        AUDITOR_PATH = companion
spec = importlib.util.spec_from_file_location("state_report_auditor", AUDITOR_PATH)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def build_report(batch: Path) -> dict:
    try:
        checked = auditor.audit(batch, allow_partial=True)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        return {
            "audit_status": "invalid_or_unavailable",
            "error": f"{type(error).__name__}: {error}",
            "inferential_result_available": False,
            "limitations": ["Invalid evidence cannot establish any gate or solver advantage."],
        }
    raw = (batch / "results.jsonl").read_bytes()[: checked["results_bytes"]]
    auditor.ensure(
        hashlib.sha256(raw).hexdigest() == checked["results_sha256"],
        "audited evidence prefix changed before reporting",
    )
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    protocol = json.loads((batch / "contract.json").read_text())["protocol"]
    auditor.ensure(
        auditor.canonical_digest(protocol) == checked["protocol_sha256"],
        "audited protocol changed before reporting",
    )
    complete = [r for r in records if r.get("status", "complete") == "complete"]
    failures = [
        {
            "n": r["n"],
            "replication": r.get("replication", 0),
            "block": r.get("block", 0),
            "method": r.get("method", "repair"),
            "error": r["error"],
        }
        for r in records
        if r.get("status") == "failed"
    ]
    result = {
        "audit_status": "valid_completed_prefix",
        "audit": checked,
        "completed_cells": len(complete),
        "failed_cells": len(failures),
        "failures": failures,
        "inferential_result_available": False,
        "limitations": [
            "Descriptive report; controlled claims require the frozen paired block analysis.",
            "Incomplete panels and failed searches cannot be silently excluded from inference.",
            "Witness checks do not independently prove CP-SAT optimum certificates.",
        ],
    }
    if protocol["stage"] == "feasibility":
        by_size = {}
        for n in protocol["sizes"]:
            rows = [r for r in complete if r["n"] == n]
            initial = [r["initial_edge_count"] for r in rows]
            repaired = [len(outcome["edges"]) for r in rows for outcome in r["repairs"]]
            strongest = max(initial, default=None)
            by_size[str(n)] = {
                "completed_states": len(rows),
                "improved_states": sum(r["improved"] for r in rows),
                "initial_edge_counts": dict(sorted(Counter(initial).items())),
                "strongest_initial": strongest,
                "strongest_repaired": max(repaired, default=None),
                "states_exceeding_strongest_observed_initial": sum(
                    any(len(q["edges"]) > strongest for q in r["repairs"]) for r in rows
                ),
            }
        result.update(
            feasibility={
                "by_size": by_size,
                "improved_states": sum(r["improved"] for r in complete),
                "declared_states": 100,
                "threshold": 10,
                "gate_status": ("PASS" if checked["feasibility_passed"] else "FAIL")
                if checked["complete_panel"]
                else "PENDING",
                "failed_states_count_as_not_improved": True,
            }
        )
    elif protocol["stage"] in {"validation", "evaluation", "development_baseline"}:
        result["descriptive_methods"] = {}
        for method in protocol["methods"]:
            rows = [r for r in complete if r["method"] == method]
            checkpoints = {}
            for cutoff in [1, 10, 60]:
                available = [r for r in rows if str(cutoff) in r["checkpoints"]]
                values = [r["checkpoints"][str(cutoff)]["edges"] for r in available]
                checkpoints[str(cutoff)] = {
                    "completed_searches": len(values),
                    "mean_edges": statistics.mean(values) if values else None,
                    "by_block": {
                        str(block): {
                            "completed_searches": len(group),
                            "mean_edges": statistics.mean(group),
                        }
                        for block in range(8)
                        if (
                            group := [
                                r["checkpoints"][str(cutoff)]["edges"]
                                for r in available
                                if r.get("block", 0) == block
                            ]
                        )
                    },
                }
            result["descriptive_methods"][method] = checkpoints
    return result


def markdown(report: dict) -> str:
    lines = ["# Independent descriptive state report", ""]
    if report["audit_status"] != "valid_completed_prefix":
        return "\n".join(
            lines + ["**Evidence audit failed. No result is authorized.**", "", report["error"], ""]
        )
    checked = report["audit"]
    lines += [
        f"Stage: `{checked['stage']}`. Recorded cells: {checked['records']}/"
        f"{checked['expected_records']}. Panel complete: **{checked['complete_panel']}**.",
        "",
        f"Failed cells retained: **{report['failed_cells']}**.",
        "",
        "This is a descriptive account. It does not establish a learned advantage, transfer,",
        "a new extremal bound or global optimality. Missing/failed searches are visible below.",
        "",
    ]
    if "feasibility" in report:
        result = report["feasibility"]
        lines += [
            f"Feasibility: **{result['gate_status']}**; {result['improved_states']}/100 "
            "states improved; threshold 10/100.",
            "",
            "| Size | Completed | Improved | Best initial | Best repaired | Above best initial |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for n, row in result["by_size"].items():
            lines.append(
                f"| {n} | {row['completed_states']} | {row['improved_states']} | "
                f"{row['strongest_initial']} | {row['strongest_repaired']} | "
                f"{row['states_exceeding_strongest_observed_initial']} |"
            )
        lines += [
            "",
            "Improving a weaker start is distinct from exceeding the strongest observed",
            "construction. Neither comparison proves a mathematical optimum.",
            "",
        ]
    for method, checkpoints in report.get("descriptive_methods", {}).items():
        lines += [
            f"## {method}",
            "",
            "| Seconds | Completed searches | Mean edges |",
            "|---:|---:|---:|",
        ]
        for cutoff, row in checkpoints.items():
            lines.append(f"| {cutoff} | {row['completed_searches']} | {row['mean_edges']} |")
        lines += ["", "These completed-cell means can be unbalanced in an incomplete panel.", ""]
    if report["failures"]:
        lines += ["## Retained failures", ""]
        for failure in report["failures"]:
            lines += [
                f"n={failure['n']}, block={failure['block']}, "
                f"replication={failure['replication']}, method={failure['method']}",
                "",
                "```text",
                failure["error"].rstrip(),
                "```",
                "",
            ]
    lines += [
        "## Provenance and limits",
        "",
        f"Results prefix SHA256: `{checked['results_sha256']}`.",
        "",
        f"Source SHA256: `{checked['source_sha256']}`.",
        "",
        f"Protocol SHA256: `{checked['protocol_sha256']}`.",
        "",
    ]
    lines += [f"- {text}" for text in checked["limitations"] + report["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    paths = [
        output,
        output.with_suffix(".md"),
        output.with_name(output.stem + "_report.py"),
        output.with_name(output.stem + "_auditor.py"),
    ]
    auditor.ensure(all(not p.exists() for p in paths), "refusing to overwrite report evidence")
    report = build_report(args.batch)
    report["report_program_sha256"] = auditor.digest(Path(__file__))
    report["report_auditor_sha256"] = auditor.digest(AUDITOR_PATH)
    if "audit" in report:
        auditor.ensure(
            report["audit"]["auditor_sha256"] == report["report_auditor_sha256"],
            "auditor changed during report",
        )
    paths[2].write_bytes(Path(__file__).read_bytes())
    paths[3].write_bytes(AUDITOR_PATH.read_bytes())
    paths[0].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    paths[1].write_text(markdown(report))
    print(f"Report: {output}; evidence audit: {report['audit_status']}.")
    if report["audit_status"] != "valid_completed_prefix":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
