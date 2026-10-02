"""Report actual offline costs and the fixed matched-selection amortization scenario."""

import argparse
import json
from pathlib import Path

from extremal_graph.repair.campaign import checkpoint_manifest
from extremal_graph.repair.cost_analysis import analyze_costs
from extremal_graph.repair.evidence import (
    independent_audit,
    load_records,
    require_input_replay,
    sha,
    verify_contract,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace retained cost analysis")
    models = checkpoint_manifest(args.checkpoints)
    checked = independent_audit(args.profile)
    if checked["failed_records"]:
        raise ValueError("failed matched profiling cannot support amortization")
    require_input_replay(args.profile)
    protocol = verify_contract(args.profile)["protocol"]
    if (
        protocol["stage"] != "development_trained_selector_profile"
        or protocol["models"] != models
        or protocol["amortization_decisions_per_search"] != 30
    ):
        raise ValueError("matched profile differs from trained panel or declared scenario")
    result = {
        **analyze_costs(
            load_records(args.checkpoints / "models.jsonl"),
            load_records(args.ledger),
            load_records(args.profile / "results.jsonl"),
        ),
        "models_sha256": sha(args.checkpoints / "models.jsonl"),
        "profile_results_sha256": checked["results_sha256"],
        "ledger_sha256": sha(args.ledger),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
