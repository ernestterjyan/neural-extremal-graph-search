# C4-free solver feasibility result

The [pilot protocol](protocol.json) was committed at `21bc9fe` before the 300 graph draws. All resulting graphs are C4-free and maximal under an independent NetworkX check. The exact reference values are from [Afzaly and McKay's table](https://users.cecs.anu.edu.au/~bdm/data/extremal.html); they were not used by any solver. All graphs and timings are in [results.jsonl](results.jsonl), with per-method aggregates in [summary.csv](summary.csv).

| n | Published exact | Best uniform | Best min-degree | Best sampled look-ahead | Best edge-swap search | Best polarity | Best polarity + edge swaps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 46 | 41 | 42 | 44 | 44 | 45 | 45 |
| 24 | 59 | 53 | 53 | 57 | 59 | 59 | 59 |
| 30 | 85 | 73 | 74 | 76 | 77 | 85 | 85 |
| 35 | 106 | 89 | 91 | 96 | 96 | 101 | 101 |
| 40 | 127 | 107 | 108 | 114 | 115 | 125 | 125 |

The result justifies a **small exploratory learned-solver study**, especially at n=35, where even the strongest pilot method is five edges short in its best run. It does not establish that a neural method will close that gap. The polarity baseline is far stronger than simple greedy and one-step methods and must be included in any future solver claim. The 200-step edge-swap procedure did not improve the polarity construction at n=35 or n=40 in the declared ten draws. The methods have very different computation costs, so these scores are a feasibility screen rather than an equal-time leaderboard.

The table tests only five sizes and six implementations. It does not establish best-known lower bounds beyond the published exact panel, and it does not exclude other stronger constructions or search heuristics. A solver improvement would require beating the strongest relevant baseline under a declared compute budget and verifying every output independently.
