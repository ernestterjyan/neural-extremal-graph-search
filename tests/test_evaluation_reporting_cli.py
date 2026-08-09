import json

import pandas as pd

from extremal_graph.cli import main
from extremal_graph.evaluation import evaluate
from extremal_graph.reporting import generate_report
from extremal_graph.serialization import save_graph
from extremal_graph.training import train
from extremal_graph.turan import construct_turan

from .test_rollouts_training import _training_config


def test_evaluation_reporting_and_cli(tmp_path, monkeypatch) -> None:
    training_config = _training_config(tmp_path)
    checkpoint = train(training_config)
    evaluation_config = tmp_path / "evaluate.toml"
    evaluation_config.write_text(
        f"""
[evaluation]
name = "test"
run_dir = "{checkpoint.parent.parent.as_posix()}"
checkpoint_glob = "test-seed-*/best.pt"
output_csv = "{(tmp_path / "results" / "test" / "evaluation.csv").as_posix()}"
sizes = [4, 6]
seeds = [3]
episodes_per_method = 1
methods = ["gnn", "untrained_gnn", "random", "least_degree", "turan_oracle"]
device = "cpu"

[model]
node_feature_dim = 4
candidate_feature_dim = 5
hidden_dim = 8
message_passing_layers = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    results = evaluate(evaluation_config)
    frame = pd.read_csv(results)
    assert len(frame) == 10
    assert frame["constraint_violations"].sum() == 0
    assert frame["terminal_maximal"].all()

    monkeypatch.chdir(tmp_path)
    summary = generate_report(results)
    assert summary.exists()
    assert (tmp_path / "reports" / "figures" / "test" / "optimality_ratio.png").exists()
    assert (results.parent / "summary.md").exists()

    graph_path = tmp_path / "turan.json"
    save_graph(construct_turan(6, 2), graph_path)
    assert main(["verify", str(graph_path)]) == 0
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"n": 3}), encoding="utf-8")
    assert main(["verify", str(bad_path)]) == 1


def test_cli_version(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as error:
        assert error.code == 0
    assert "0.1.1" in capsys.readouterr().out


def test_mlp_evaluation_and_parameter_count(tmp_path) -> None:
    training_config = _training_config(tmp_path).read_text(encoding="utf-8")
    training_config = training_config.replace('name = "test"', 'name = "mlp-test"')
    training_config = training_config.replace(
        "[model]\n", '[model]\nfamily = "mlp"\nmax_nodes = 6\n'
    )
    mlp_training_config = tmp_path / "mlp-train.toml"
    mlp_training_config.write_text(training_config, encoding="utf-8")
    checkpoint = train(mlp_training_config)

    evaluation_config = tmp_path / "mlp-evaluate.toml"
    evaluation_config.write_text(
        f"""
[evaluation]
name = "mlp-test"
run_dir = "{checkpoint.parent.parent.as_posix()}"
checkpoint_glob = "unused-gnn-*.pt"
mlp_checkpoint_glob = "mlp-test-seed-*/best.pt"
output_csv = "{(tmp_path / "results" / "mlp" / "evaluation.csv").as_posix()}"
sizes = [4, 6]
seeds = [3]
episodes_per_method = 2
methods = ["mlp", "untrained_mlp"]
device = "cpu"

[model]
family = "gnn"
node_feature_dim = 4
candidate_feature_dim = 5
hidden_dim = 8
message_passing_layers = 1

[mlp_model]
family = "mlp"
node_feature_dim = 4
candidate_feature_dim = 5
hidden_dim = 8
message_passing_layers = 1
max_nodes = 6
""".strip()
        + "\n",
        encoding="utf-8",
    )

    first_path = evaluate(evaluation_config)
    first = pd.read_csv(first_path).drop(columns=["inference_time_seconds"])
    first_path.unlink()
    second = pd.read_csv(evaluate(evaluation_config)).drop(columns=["inference_time_seconds"])

    pd.testing.assert_frame_equal(first, second)
    assert set(first["method"]) == {"mlp", "untrained_mlp"}
    assert (first["parameter_count"] > 0).all()
    assert first["constraint_violations"].sum() == 0
    assert first["terminal_maximal"].all()
