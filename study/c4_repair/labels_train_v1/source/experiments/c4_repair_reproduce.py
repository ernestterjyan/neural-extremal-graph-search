"""Repeat a fixed timed subset in a distinct environment using archived source."""

import argparse
from pathlib import Path

from extremal_graph.repair.controls import ExperimentLease
from extremal_graph.repair.reproduction import prepare, report, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--reference", type=Path, required=True)
    command.add_argument("--checkpoints", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("run")
    command.add_argument("--batch", type=Path, required=True)
    command.add_argument("--checkpoints", type=Path, required=True)
    command = commands.add_parser("report")
    command.add_argument("--batch", type=Path, required=True)
    command.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "run":
        with ExperimentLease(stage="reproduction", batch=args.batch):
            run(args)
    else:
        {"prepare": prepare, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
