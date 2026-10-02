# Retained partial tuning evidence

This is an independently audited, immutable prefix of the running tuning_v1
journal: 16 completed cells, no failures, 119 graph occurrences checked. The
report gives the exact source, protocol and prefix hashes. It is not the full
96-cell panel and cannot authorize parameter selection or a learned claim.

DEVELOPMENT_PREFIX_1_results.jsonl retains precisely the audited bytes. The
corresponding report and auditor programs are retained beside it. To reproduce
this descriptive report without later appends, create a temporary batch directory,
copy this file as results.jsonl, and copy tuning_v1/contract.json and its source/
there. Run DEVELOPMENT_PREFIX_1_report.py against that temporary batch with a
new output name. Its companion auditor is selected automatically.

The original live results.jsonl and run.log continue to append independently.
No search or timing experiment is rerun by the descriptive report. The exact
source used for the 96 searches stays frozen and unchanged.
