"""Command-line interface for training, evaluation, verification, and reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .serialization import load_graph
from .verification import verify_graph


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="negs",
        description="Neural Extremal Graph Search",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.1")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_parser = subcommands.add_parser("train", help="train one curriculum seed")
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--resume", type=Path)

    evaluate_parser = subcommands.add_parser("evaluate", help="evaluate policies and baselines")
    evaluate_parser.add_argument("--config", required=True, type=Path)

    verify_parser = subcommands.add_parser("verify", help="independently verify a graph JSON file")
    verify_parser.add_argument("graph", type=Path)
    verify_parser.add_argument("--r", type=int)
    verify_parser.add_argument("--allow-nonmaximal", action="store_true")

    report_parser = subcommands.add_parser("report", help="aggregate results and create figures")
    report_parser.add_argument("--results", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "train":
            from .training import train

            checkpoint = train(args.config, seed_override=args.seed, resume=args.resume)
            print(checkpoint)
            return 0
        if args.command == "evaluate":
            from .evaluation import evaluate

            output = evaluate(args.config)
            print(output)
            return 0
        if args.command == "verify":
            graph = load_graph(args.graph)
            report = verify_graph(
                graph,
                r=args.r,
                require_maximal=not args.allow_nonmaximal,
            )
            print(json.dumps(asdict(report), indent=2))
            return 0 if report.valid else 2
        if args.command == "report":
            from .reporting import generate_report

            output = generate_report(args.results)
            print(output)
            return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
