#!/bin/zsh
# Launch only after the label/training sequence has an authoritative zero exit.
# On interruption, resume the incomplete stage; never restart this whole script.
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
campaign_repository="$PWD"
campaign_evidence="$campaign_repository/study/c4_repair"
campaign_python="$campaign_repository/.venv/bin/python"
campaign_source="$campaign_evidence/labels_train_v1/source"
campaign_validation="$campaign_evidence/validation_v1"
export PYTHONPATH="$campaign_source/src"

"$campaign_python" - <<'PY'
import json
from pathlib import Path
from extremal_graph.repair.campaign import checkpoint_manifest
from extremal_graph.repair.development import load_tuning
from extremal_graph.repair.evidence import independent_audit, json_sha, load_records, require_input_replay, sha, source_manifest, verify_contract
root = Path('study/c4_repair')
assert json_sha(source_manifest()) == '13dfb0462b19f14913184fc98c10d234c6c66ea247d9efb1dc4899b66b16eafb'
state = json.loads((root / 'ACTIVE_RUN.json').read_text())
assert state['stage'] == 'labels_training_and_profile_completed', 'finish and verify the previous sequence first'
for name, expected in [('labels_train_v1', 96), ('labels_validation_v1', 80), ('trained_profile_v1', 3)]:
    batch = root / name
    checked = independent_audit(batch)
    assert checked['complete_panel'] and checked['records'] == expected and not checked['failed_records']
    require_input_replay(batch)
models = checkpoint_manifest(root / 'models_v1')
assert verify_contract(root / 'trained_profile_v1')['protocol']['models'] == models
load_tuning(root / 'tuning_v1/SELECTION.json')
cost = json.loads((root / 'COST_REPORT_v1.json').read_text())
assert cost['models_sha256'] == sha(root / 'models_v1/models.jsonl')
assert cost['profile_results_sha256'] == sha(root / 'trained_profile_v1/results.jsonl')
assert cost['ledger_sha256'] == sha(root / 'costs.jsonl')
assert sum(row['paid_eur'] for row in load_records(root / 'costs.jsonl')) <= 500
state.update(stage='end_to_end_validation', batch='study/c4_repair/validation_v1', status='running', progress_metadata_only=True, execution_session_id=None, verified_live=None,
             expected_validation_searches=1600,
             next_action='Poll the actual validation execution handle and inspect validation_v1_run.log; no concurrent CPU-heavy work. Metadata alone does not prove liveness.')
(root / 'ACTIVE_RUN.json').write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
PY

"$campaign_python" -u "$campaign_source/experiments/c4_repair.py" evaluate \
  --split validation --feasibility "$campaign_evidence/feasibility_v2" \
  --checkpoints "$campaign_evidence/models_v1" \
  --configuration "$campaign_evidence/tuning_v1/SELECTION.json" \
  --seed 7000000 --output "$campaign_validation" \
  > "$campaign_evidence/validation_v1_run.log" 2>&1
"$campaign_python" "$campaign_source/experiments/c4_repair_audit.py" \
  --batch "$campaign_validation" --output "$campaign_validation/INDEPENDENT_AUDIT.json" \
  > "$campaign_validation/audit.log" 2>&1
"$campaign_python" "$campaign_source/experiments/c4_repair_replay_inputs.py" \
  --batch "$campaign_validation" --output "$campaign_validation/INPUT_REPLAY.json" \
  > "$campaign_validation/replay.log" 2>&1
"$campaign_python" "$campaign_source/experiments/c4_repair.py" report \
  --batch "$campaign_validation" > "$campaign_validation/report.log" 2>&1

"$campaign_python" - <<'PY'
import json
from datetime import UTC, datetime
from pathlib import Path
from extremal_graph.repair.evidence import independent_audit, load_records, require_input_replay, sha, verify_contract
root = Path('study/c4_repair')
batch = root / 'validation_v1'
checked = independent_audit(batch)
assert checked['complete_panel'] and checked['eligible_for_inference'] and not checked['failed_records']
require_input_replay(batch)
rows = load_records(batch / 'results.jsonl')
assert len(rows) == 1600
contract = verify_contract(batch)
ledger = root / 'costs.jsonl'
assert not any(r['stage'] == 'validation_v1' for r in load_records(ledger)), 'refusing to double-count validation'
entry = {'date': datetime.now(UTC).date().isoformat(), 'stage': 'validation_v1',
         'local_wall_seconds': sum(r['elapsed_seconds'] for r in rows), 'paid_eur': 0.0,
         'resource': 'local Apple M5 CPU, verified identity',
         'source_sha256': contract['source_sha256'], 'protocol_sha256': contract['protocol_sha256'],
         'results_sha256': sha(batch / 'results.jsonl')}
with ledger.open('a') as stream:
    stream.write(json.dumps(entry, sort_keys=True) + '\n')
summary = json.loads((batch / 'summary.json').read_text())
state = json.loads((root / 'ACTIVE_RUN.json').read_text())
state.update(stage='validation_completed', status='completed', progress_metadata_only=True,
             learning_gate_passed=summary['learning_gate_passed'],
             next_action='Confirm authoritative terminal exit; review audit, replay and report. A positive learning gate permits held-out freeze. A failed gate requires reproduction and a negative report, with no held-out or record search.')
(root / 'ACTIVE_RUN.json').write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
PY
