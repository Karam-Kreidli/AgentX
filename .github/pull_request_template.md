## Task
<!-- e.g. YARD-3. One task per PR. -->
Task ID:

## What changed
<!-- 2–4 lines -->

## Done when (copy from TASKS.md and tick)
- [ ]
- [ ]

## Scorecard (`python tools/eval_all.py --all --no-llm`)
| Scenario | Ships departed (before → after) | Honest turnaround h | Violations | Crashes |
|---|---|---|---|---|
| practice_01 | | | | |
| practice_02 | | | | |
| practice_03 | | | | |
| practice_04 | | | | |
| practice_05 | | | | |

KPI or rule affected:

## Ground rules check
- [ ] Nothing reaches the simulator without the guard (only `core/executor.py` changes state)
- [ ] No hard-rule break possible (incl. idle-crane rail order, reefer → I/J only)
- [ ] Can't crash the run; errors are caught and logged
- [ ] No scenario/call/vessel/container IDs in logic; no workbook or `data/` reads in agent code
- [ ] Untrusted text only as quoted, truncated data; LLM can't create actions; fallback exists
- [ ] Every new state-changing action has a reason
- [ ] Shared types / interfaces unchanged (or CONTRACTS.md updated and approved by E)
- [ ] No secrets; dependencies pinned
- [ ] Tests from the task spec added and passing
