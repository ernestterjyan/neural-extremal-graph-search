"""Replay a declared subset using the separately installed frozen source."""

import json
from pathlib import Path

from extremal_graph.repair import engine
from extremal_graph.repair.evidence import environment, sha, verify_contract
from extremal_graph.repair.graph import Graph, verify_witness

HERE = Path(__file__).resolve().parent
BATCH = HERE.parent / "feasibility_v1"
selection = json.loads((HERE / "selection.json").read_text())
assert Path(engine.__file__).is_relative_to(BATCH / "source"), "use the frozen-source environment"
assert sha(BATCH / "results.jsonl") == selection["original_results_sha256"]
assert sha(BATCH / "contract.json") == selection["source_contract_sha256"]
contract = verify_contract(BATCH)
records = [json.loads(line) for line in (BATCH / "results.jsonl").read_text().splitlines()]
results = []
for cell in selection["cells"]:
    record = next(
        r for r in records if r["n"] == cell["n"] and r["replication"] == cell["replication"]
    )
    original = record["repairs"][cell["region_index"]]
    outcome = engine.repair(
        Graph.from_edges(record["n"], record["initial_edges"]),
        tuple(original["region"]["vertices"]),
        seconds=contract["protocol"]["repair_seconds"],
        seed=original["seed"],
    )
    verify_witness(record["n"], outcome.graph.edges(), explicit_cycles=True)
    results.append(
        {
            **cell,
            "edge_count_matches": outcome.graph.m == len(original["edges"]),
            "local_certificate_matches": outcome.local_optimal == original["local_optimal"],
            "witness_matches": outcome.graph.edges() == [tuple(e) for e in original["edges"]],
            "elapsed_seconds": outcome.seconds,
            "status": outcome.status,
            "edges": outcome.graph.edges(),
            "c4_free": True,
        }
    )
report = {
    "environment": environment(),
    "source_sha256": contract["source_sha256"],
    "fresh_installation": True,
    "separate_physical_machine": False,
    "results": results,
    "passed": all(r["edge_count_matches"] and r["local_certificate_matches"] for r in results),
    "limitation": "Six certified cells, fresh environment on the same machine; "
    "timed replay is not a full campaign replication.",
}
(HERE / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(f"Replay passed={report['passed']}; {len(results)} independently verified cells.")
