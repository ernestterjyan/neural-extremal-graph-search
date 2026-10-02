#!/bin/zsh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
campaign_repository="$PWD"
campaign_evidence="$campaign_repository/study/c4_repair"
campaign_python="$campaign_repository/.venv/bin/python"
campaign_source="$campaign_evidence/labels_train_v1/source"
export PYTHONPATH="$campaign_source/src"

set_stage() {
  "$campaign_python" - "$1" "$2" <<'PY'
import json,sys
from pathlib import Path
path=Path('study/c4_repair/ACTIVE_RUN.json')
state=json.loads(path.read_text())
state.update(stage=sys.argv[1],batch=sys.argv[2],status='running',progress_metadata_only=True)
path.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
PY
}

check_labels() {
  "$campaign_python" "$campaign_source/experiments/c4_repair_audit.py" --batch "$1" --output "$1/INDEPENDENT_AUDIT.json" > "$1/audit.log" 2>&1
  "$campaign_python" "$campaign_source/experiments/c4_repair_replay_inputs.py" --batch "$1" --output "$1/INPUT_REPLAY.json" > "$1/replay.log" 2>&1
  "$campaign_python" - "$1" <<'PY'
import sys
from pathlib import Path
from extremal_graph.repair.evidence import independent_audit,require_input_replay
batch=Path(sys.argv[1])
checked=independent_audit(batch)
assert checked['complete_panel'] and not checked['failed_records']
require_input_replay(batch)
PY
}

set_stage collection_train study/c4_repair/labels_train_v1
"$campaign_python" -u "$campaign_source/experiments/c4_repair.py" collect --feasibility "$campaign_evidence/feasibility_v2" --split train --output "$campaign_evidence/labels_train_v1" --seed 4000000 --states-per-size 16 --region-sizes 3 4 5 --per-family 4 --seconds 2 > "$campaign_evidence/labels_train_v1/run.log" 2>&1
check_labels "$campaign_evidence/labels_train_v1"
set_stage collection_validation study/c4_repair/labels_validation_v1
"$campaign_python" -u "$campaign_source/experiments/c4_repair.py" collect --feasibility "$campaign_evidence/feasibility_v2" --split validation --output "$campaign_evidence/labels_validation_v1" --seed 5000000 --states-per-size 16 --region-sizes 3 4 5 --per-family 4 --seconds 2 > "$campaign_evidence/labels_validation_v1/run.log" 2>&1
check_labels "$campaign_evidence/labels_validation_v1"
set_stage training study/c4_repair/models_v1
"$campaign_python" -u "$campaign_source/experiments/c4_repair.py" train --feasibility "$campaign_evidence/feasibility_v2" --data "$campaign_evidence/labels_train_v1" --validation-data "$campaign_evidence/labels_validation_v1" --output "$campaign_evidence/models_v1" --seed 6000000 --epochs 30 --learning-rate 0.001 > "$campaign_evidence/models_v1_run.log" 2>&1
set_stage trained_selector_profile study/c4_repair/trained_profile_v1
"$campaign_python" -u "$campaign_source/experiments/c4_repair_trained_profile.py" --checkpoints "$campaign_evidence/models_v1" --configuration "$campaign_evidence/tuning_v1/SELECTION.json" --output "$campaign_evidence/trained_profile_v1" > "$campaign_evidence/trained_profile_v1_run.log" 2>&1
check_labels "$campaign_evidence/trained_profile_v1"

"$campaign_python" - <<'PY'
import json
from collections import Counter
from datetime import UTC,datetime
from pathlib import Path
from extremal_graph.repair.evidence import load_records,sha,verify_contract
root=Path('study/c4_repair')
ledger=root/'costs.jsonl'
existing=load_records(ledger)
for name in ['labels_train_v1','labels_validation_v1','models_v1','trained_profile_v1']:
    assert not any(r['stage']==name for r in existing), 'refusing to double-count costs'
    batch=root/name
    records=load_records(batch/('models.jsonl' if name=='models_v1' else 'results.jsonl'))
    contract=verify_contract(batch)
    entry={'date':datetime.now(UTC).date().isoformat(),'stage':name,'local_wall_seconds':sum(r['elapsed_seconds'] for r in records),'paid_eur':0.0,'resource':'local Apple M5 CPU, verified identity','source_sha256':contract['source_sha256'],'protocol_sha256':contract['protocol_sha256'],'results_sha256':sha(batch/('models.jsonl' if name=='models_v1' else 'results.jsonl'))}
    with ledger.open('a') as f:f.write(json.dumps(entry,sort_keys=True)+'\n')
summary={}
for name in ['labels_train_v1','labels_validation_v1']:
    rows=load_records(root/name/'results.jsonl')
    gains=[r['gain'] for row in rows for r in row['repairs']]
    summary[name]={'states':len(rows),'repair_outcomes':len(gains),'positive_outcomes':sum(g>0 for g in gains),'informative_ranking_states':sum(len({r['gain'] for r in row['repairs']})>1 for row in rows),'gain_histogram':dict(sorted(Counter(gains).items())),'results_sha256':sha(root/name/'results.jsonl'),'scope':'supervised labels only; not end-to-end validation'}
(root/'LABEL_DIAGNOSTICS_v1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
PY
"$campaign_python" "$campaign_source/experiments/c4_repair_cost_report.py" --checkpoints "$campaign_evidence/models_v1" --profile "$campaign_evidence/trained_profile_v1" --ledger "$campaign_evidence/costs.jsonl" --output "$campaign_evidence/COST_REPORT_v1.json" > "$campaign_evidence/COST_REPORT_v1.log" 2>&1
set_stage labels_training_and_profile_completed study/c4_repair/models_v1
