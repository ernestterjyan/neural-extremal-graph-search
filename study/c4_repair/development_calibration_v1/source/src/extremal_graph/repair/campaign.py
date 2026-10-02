"""Gated collection, training, validation, and frozen held-out evaluation."""

from __future__ import annotations

import json
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from .analysis import analyze
from .constructions import starting_pool
from .controls import balanced_method_order
from .engine import repair
from .evidence import (
    EVALUATION_SIZES,
    TRAIN_SIZES,
    VALIDATION_SIZES,
    append_record,
    environment,
    independent_audit,
    json_sha,
    load_records,
    prepare_batch,
    prepare_or_resume,
    require_input_replay,
    sha,
    source_manifest,
    verify_contract,
)
from .graph import Graph, greedy_fill, verify_witness
from .regions import candidate_regions
from .search import SearchConfig, neighborhood_search, tabu_search
from .selectors import RegionGNN, RegionMLP, model_selector
from .training import load_model, train_family


def require_feasibility(directory: Path) -> None:
    contract = verify_contract(directory)
    protocol = contract["protocol"]
    for filename in ["graph.py", "engine.py", "regions.py", "constructions.py"]:
        relative = f"src/extremal_graph/repair/{filename}"
        if contract["source"].get(relative) != source_manifest().get(relative):
            raise ValueError("repair machinery changed since feasibility; rerun its gate")
    checked = independent_audit(directory)
    require_input_replay(directory)
    if not checked["feasibility_passed"]:
        raise ValueError("independent feasibility gate failed")
    records = load_records(directory / "results.jsonl")
    expected = {(n, i) for n in [35, 40] for i in range(50)}
    if (
        protocol["stage"] != "feasibility"
        or len(records) != 100
        or {(r["n"], r["replication"]) for r in records} != expected
    ):
        raise ValueError("complete 100-state feasibility evidence required")
    improved = 0
    for record in records:
        if (
            record["source_sha256"] != contract["source_sha256"]
            or record["protocol_sha256"] != contract["protocol_sha256"]
        ):
            raise ValueError("gate provenance mismatch")
        if record["status"] == "failed":
            continue
        initial = Graph.from_edges(record["n"], record["initial_edges"])
        verify_witness(initial.n, initial.edges())
        if initial.m >= int(protocol["known_exact_values"][str(initial.n)]):
            raise ValueError("gate state not below optimum")
        positive = False
        for outcome in record["repairs"]:
            verify_witness(initial.n, outcome["edges"])
            if len(outcome["edges"]) - initial.m != outcome["gain"]:
                raise ValueError("gate gain mismatch")
            result = Graph.from_edges(initial.n, outcome["edges"])
            outside = set(range(initial.n)) - set(outcome["region"]["vertices"])
            if any(
                initial.has(u, v) != result.has(u, v) for u in outside for v in outside if u < v
            ):
                raise ValueError("gate fixed complement changed")
            positive |= outcome["gain"] > 0
        if positive != record["improved"]:
            raise ValueError("gate improvement flag mismatch")
        improved += positive
    if improved < 10:
        raise ValueError(
            f"feasibility gate failed ({improved}/100); revise neighborhoods before learning"
        )


def collect(args) -> None:
    require_feasibility(args.feasibility)
    gate = verify_contract(args.feasibility)["protocol"]
    if (
        args.seconds != gate["repair_seconds"]
        or args.per_family != gate["per_family"]
        or list(args.region_sizes) != gate["region_sizes"]
    ):
        raise ValueError("collect labels with the region pool and limits that passed feasibility")
    sizes = TRAIN_SIZES if args.split == "train" else VALIDATION_SIZES
    protocol = {
        "stage": "collection",
        "split": args.split,
        "sizes": sizes,
        "states_per_size": args.states_per_size,
        "seed_offset": args.seed,
        "region_sizes": args.region_sizes,
        "per_family": args.per_family,
        "repair_seconds": args.seconds,
        "state_mix": "best, trim, extend, uniform, perturbed best",
        "feasibility_results_sha256": sha(args.feasibility / "results.jsonl"),
    }
    contract = prepare_or_resume(args.output, protocol)
    path = args.output / "results.jsonl"
    done = {(r["n"], r["replication"]) for r in load_records(path)} if path.exists() else set()
    for n in sizes:
        for replication in range(args.states_per_size):
            if (n, replication) in done:
                continue
            seed = args.seed + n * 10000 + replication
            started = time.perf_counter()
            record = {
                "n": n,
                "seed": seed,
                "replication": replication,
                "source_sha256": contract["source_sha256"],
                "protocol_sha256": contract["protocol_sha256"],
                "status": "complete",
                "repairs": [],
            }
            try:
                pool = starting_pool(n, seed)
                best = max(pool, key=lambda s: s.graph.m).graph
                state_type = replication % 5
                if state_type in (0, 4):
                    graph = best.copy()
                elif state_type == 1:
                    graph = pool[0].graph.copy()
                elif state_type == 2:
                    graph = pool[1].graph.copy()
                else:
                    graph = pool[-1].graph.copy()
                if state_type == 4:
                    rng = random.Random(seed + 50000000)
                    chosen = set(rng.sample(range(n), 3))
                    for edge in graph.edges():
                        if set(edge) & chosen:
                            graph.remove(*edge)
                    greedy_fill(graph, rng)
                record.update(
                    initial_edges=graph.edges(),
                    initial_edge_count=graph.m,
                    starting_sha256=graph.digest(),
                    state_type=state_type,
                )
                regions = candidate_regions(
                    graph,
                    random.Random(seed + 100000000),
                    sizes=tuple(args.region_sizes),
                    per_family=args.per_family,
                )
                for j, region in enumerate(regions):
                    outcome = repair(
                        graph, region.vertices, seconds=args.seconds, seed=seed + 1000000 + j
                    )
                    verify_witness(n, outcome.graph.edges())
                    record["repairs"].append(
                        {
                            "region": asdict(region),
                            "seed": seed + 1000000 + j,
                            "gain": outcome.gain,
                            "seconds": outcome.seconds,
                            "local_optimal": outcome.local_optimal,
                            "status": outcome.status,
                            "rounds": outcome.rounds,
                            "cuts": outcome.cuts,
                            "incumbent_events": outcome.event_records(),
                            "edges": outcome.graph.edges(),
                        }
                    )
            except Exception:
                record.update(status="failed", error=traceback.format_exc())
            record["elapsed_seconds"] = time.perf_counter() - started
            append_record(args.output / "results.jsonl", record)
            print(n, replication, record["status"], flush=True)


def train(args) -> None:
    require_feasibility(args.feasibility)
    for directory in [args.data, args.validation_data]:
        checked = independent_audit(directory)
        if checked["failed_records"]:
            raise ValueError("failed label cells cannot enter training")
        require_input_replay(directory)
    training = verify_contract(args.data)["protocol"]
    validation = verify_contract(args.validation_data)["protocol"]
    if training["split"] != "train" or validation["split"] != "validation":
        raise ValueError("training and validation splits must stay separate")
    if set(training["sizes"]) != set(TRAIN_SIZES) or set(validation["sizes"]) != set(
        VALIDATION_SIZES
    ):
        raise ValueError("declared learning size panels required")
    repair_configuration = {
        key: training[key] for key in ["region_sizes", "per_family", "repair_seconds"]
    }
    if any(validation[key] != value for key, value in repair_configuration.items()):
        raise ValueError("training and validation labels used different repair configurations")
    protocol = {
        "stage": "training",
        "families": ["gnn", "mlp"],
        "training_seeds": list(range(args.seed, args.seed + 8)),
        "epochs": args.epochs,
        "repair_configuration": repair_configuration,
        "learning_rate": args.learning_rate,
        "data_sha256": sha(args.data / "results.jsonl"),
        "validation_data_sha256": sha(args.validation_data / "results.jsonl"),
    }
    prepare_batch(args.output, protocol)
    for family in protocol["families"]:
        for seed in protocol["training_seeds"]:
            metadata = train_family(
                args.data,
                args.validation_data,
                args.output / family / str(seed),
                family=family,
                seed=seed,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
            )
            append_record(args.output / "models.jsonl", metadata)
            print(family, seed, metadata["validation_loss"], flush=True)


def checkpoint_manifest(directory: Path) -> list[dict]:
    contract = verify_contract(directory)
    if contract["protocol"]["stage"] != "training":
        raise ValueError("checkpoint panel must come from a retained training batch")
    current_source = source_manifest()
    for filename in ["selectors.py", "training.py"]:
        relative = f"src/extremal_graph/repair/{filename}"
        if contract["source"].get(relative) != current_source.get(relative):
            raise ValueError("model architecture or training changed since checkpoint creation")
    models = load_records(directory / "models.jsonl")
    seeds = contract["protocol"]["training_seeds"]
    expected = {(family, seed) for family in ["gnn", "mlp"] for seed in seeds}
    if (
        len(seeds) != 8
        or len(set(seeds)) != 8
        or any(type(seed) is not int for seed in seeds)
        or len(models) != 16
        or {(m["family"], m["seed"]) for m in models} != expected
    ):
        raise ValueError("eight independently trained models per family required")
    for model in models:
        path = directory / model["family"] / str(model["seed"]) / "best.pt"
        if sha(path) != model["checkpoint_sha256"]:
            raise ValueError("training checkpoint was changed")
        header = torch.load(path, map_location="cpu", weights_only=True)
        architecture = {"gnn": RegionGNN.architecture, "mlp": RegionMLP.architecture}[
            model["family"]
        ]
        if (
            header.get("family") != model["family"]
            or type(header.get("seed")) is not int
            or header["seed"] != model["seed"]
            or header.get("architecture") != architecture
            or model["architecture"] != architecture
        ):
            raise ValueError(
                "checkpoint family, training seed or architecture differs from its panel"
            )
        if (
            model["train_results_sha256"] != contract["protocol"]["data_sha256"]
            or model["validation_results_sha256"] != contract["protocol"]["validation_data_sha256"]
        ):
            raise ValueError("models were trained on different or changed data")
    return [
        {
            "family": m["family"],
            "seed": m["seed"],
            "block": seeds.index(m["seed"]),
            "checkpoint_sha256": m["checkpoint_sha256"],
            "architecture": m["architecture"],
            "training_seconds": m["elapsed_seconds"],
        }
        for m in models
    ]


def evaluate(args) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = checkpoint_manifest(args.checkpoints)
    if args.split == "evaluation":
        frozen = json.loads(args.freeze.read_text())
        current_environment = environment()
        if current_environment["hardware"] != frozen["evaluation_hardware"]:
            raise ValueError("primary evaluation hardware differs from its frozen contract")
        if any(
            current_environment[key] != value for key, value in frozen["evaluation_runtime"].items()
        ):
            raise ValueError("primary evaluation runtime differs from its frozen contract")
        if frozen["source_sha256"] != json_sha(source_manifest()):
            raise ValueError("source changed after freezing")
        if frozen["models"] != models:
            raise ValueError("trained model panel changed after freezing")
        protocol = frozen["evaluation_protocol"]
        # All arguments influencing evaluation come from the frozen contract.
        config = SearchConfig(
            **{
                **protocol["configuration"],
                "region_sizes": tuple(protocol["configuration"]["region_sizes"]),
            }
        )
    else:
        from .development import load_tuning

        require_feasibility(args.feasibility)
        if not environment()["hardware"]["verified_cpu_model"]:
            raise ValueError("identify and record the CPU model before end-to-end validation")
        if getattr(args, "configuration", None) is None:
            raise ValueError("retained classical development tuning required before validation")
        config = load_tuning(args.configuration)
        learned_recipe = verify_contract(args.checkpoints)["protocol"]["repair_configuration"]
        if (
            config.repair_seconds != learned_recipe["repair_seconds"]
            or config.per_family != learned_recipe["per_family"]
            or list(config.region_sizes) != learned_recipe["region_sizes"]
        ):
            raise ValueError("end-to-end search must use the common repair recipe used for labels")
        protocol = {
            "stage": "validation",
            "sizes": list(VALIDATION_SIZES),
            "blocks": 8,
            "replications": 10,
            "methods": ["gnn", "mlp", "adaptive", "tabu"],
            "seed_offset": args.seed,
            "models": models,
            "configuration": asdict(config),
            "method_order": "seeded permutation per size/replication; rotate across blocks",
            "tuning_selection_sha256": sha(args.configuration),
        }
    contract = prepare_or_resume(args.output, protocol)
    path = args.output / "results.jsonl"
    done = (
        {(r["block"], r["n"], r["replication"], r["method"]) for r in load_records(path)}
        if path.exists()
        else set()
    )
    index = {(m["family"], m["block"]): m for m in models}
    for block in range(8):
        for n in protocol["sizes"]:
            for replication in range(10):
                seed = protocol["seed_offset"] + block * 1000000 + n * 10000 + replication
                order = balanced_method_order(
                    protocol["methods"],
                    seed_offset=protocol["seed_offset"],
                    n=n,
                    replication=replication,
                    block=block,
                )
                for position, method in enumerate(order):
                    if (block, n, replication, method) in done:
                        continue
                    record = {
                        "n": n,
                        "block": block,
                        "replication": replication,
                        "seed": seed,
                        "method": method,
                        "method_position": position,
                        "status": "complete",
                        "source_sha256": contract["source_sha256"],
                        "protocol_sha256": contract["protocol_sha256"],
                    }
                    try:
                        selector = None
                        loading_started = time.perf_counter()
                        if method in {"gnn", "mlp"}:
                            metadata = index[(method, block)]
                            checkpoint = (
                                args.checkpoints / method / str(metadata["seed"]) / "best.pt"
                            )
                            selector = model_selector(
                                load_model(checkpoint, metadata["checkpoint_sha256"])
                            )
                            record.update(
                                checkpoint_sha256=metadata["checkpoint_sha256"],
                                training_seed=metadata["seed"],
                            )
                        # Checkpoint loading, graph initialization, candidate generation,
                        # feature extraction, inference and repair all consume search time.
                        record["model_loading_seconds"] = time.perf_counter() - loading_started
                        result = (
                            tabu_search(n, seed, config, started_at=loading_started)
                            if method == "tabu"
                            else neighborhood_search(
                                n,
                                seed,
                                config,
                                method=method,
                                selector=selector,
                                started_at=loading_started,
                            )
                        )
                        record.update(result)
                    except Exception:
                        record.update(status="failed", error=traceback.format_exc())
                    append_record(args.output / "results.jsonl", record)
                    print(
                        block,
                        n,
                        replication,
                        method,
                        record.get("edge_count"),
                        record["status"],
                        flush=True,
                    )


def freeze(args) -> None:
    if args.output.exists():
        raise FileExistsError("refusing to replace a frozen protocol")
    current_environment = environment()
    hardware = current_environment["hardware"]
    contract = verify_contract(args.validation)
    if contract["protocol"]["stage"] != "validation":
        raise ValueError("end-to-end validation required before freezing")
    result = analyze(load_records(args.validation / "results.jsonl"), list(VALIDATION_SIZES))
    checked = independent_audit(args.validation)
    if not checked["eligible_for_inference"]:
        raise ValueError("complete independently audited validation required before freezing")
    require_input_replay(args.validation)
    if not result["learning_gate_passed"]:
        raise ValueError(
            "learning gate failed; stop the larger neural campaign and report the result"
        )
    models = checkpoint_manifest(args.checkpoints)
    if models != contract["protocol"]["models"]:
        raise ValueError("validation used different checkpoints")
    if contract["source_sha256"] != json_sha(source_manifest()):
        raise ValueError("methods changed since validation; repeat validation before freezing")
    if not hardware["verified_cpu_model"]:
        raise ValueError("identify and record the CPU model before freezing primary evaluation")
    if contract["environment"]["hardware"] != hardware:
        raise ValueError(
            "hardware identity changed since validation; repeat validation before freezing"
        )
    runtime = {
        key: current_environment[key]
        for key in ["platform", "machine", "python", "packages", "cpu_workers"]
    }
    if any(contract["environment"][key] != value for key, value in runtime.items()):
        raise ValueError("runtime changed since validation; repeat validation before freezing")
    evaluation_protocol = {
        **contract["protocol"],
        "stage": "evaluation",
        "sizes": list(EVALUATION_SIZES),
        "seed_offset": 17000000,
        "analysis": {
            "primary_seconds": 60,
            "secondary_seconds": [1, 10],
            "pairing": "mean ten searches per size within each of eight model-seed blocks",
            "intervals": "paired t, Bonferroni two comparisons, simultaneous 95%",
            "success": "mean>=1 edge and CI lower>0 vs both; beat both on >=6/8 sizes",
            "failure": "no substitution of secondary metrics for failed primary",
        },
    }
    frozen = {
        "evaluation_protocol": evaluation_protocol,
        "models": models,
        "source_sha256": json_sha(source_manifest()),
        "source": source_manifest(),
        "validation_results_sha256": sha(args.validation / "results.jsonl"),
        "validation_audit": checked,
        "validation_input_replay_sha256": sha(args.validation / "INPUT_REPLAY.json"),
        "learning_gate": result,
        "evaluation_hardware": hardware,
        "evaluation_runtime": runtime,
        "paid_budget_eur": 500,
        "published_witnesses_in_start_pool": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    print(f"Frozen protocol: {args.output}")
