# AgentX — P2 Container Terminal Orchestra

Multi-agent planner for the AgentX@CCI Problem 2 container terminal simulator.
Read **GROUND_RULES.md** before writing code; task specs are in **TASKS.md**,
shared types and interfaces in **CONTRACTS.md**.

## Run it

Python 3.11 only.

```bash
git clone https://github.com/Karam-Kreidli/AgentX.git && cd AgentX
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # LLM settings; optional (see "No LLM" below)
cp /path/to/P2_Container_Terminal_Data.xlsx data/     # the official workbook (never in git)
python run.py --scenario practice_01 --no-llm
```

If that prints a table with `viol 0` and exits 0, the install works.

| What | Command |
|---|---|
| One practice scenario | `python run.py --scenario practice_03` |
| All five practice scenarios | `python run.py --all` |
| One hidden scenario | `python run.py --scenario hidden_01 --key /path/to/P2_Container_Terminal_ANSWER_KEY.xlsx` |
| Every hidden scenario in the key | `python run.py --all-hidden --key /path/to/P2_Container_Terminal_ANSWER_KEY.xlsx` |
| No LLM (rules only) | add `--no-llm` to any of the above |
| Another workbook | add `--data /path/to/workbook.xlsx` |
| Another output folder | add `--runs /path/to/folder` (default `runs/`) |

**Data workbook.** Without `--data`, `run.py` looks for `P2_Container_Terminal_Data.xlsx` in `data/`, then
next to the key file, then the current folder, then `toolskit/data/`. It prints the file it used. The
agent never opens the workbook or the key; only the toolkit does, through `run.py`. For `--all-hidden`,
`run.py` itself reads the scenario names from the key (nothing else) and passes them to the toolkit.

**No LLM.** `--no-llm` never calls a model. Without it, the run uses the LLM named in `.env` and falls back
to rules-only by itself if the endpoint is down, slow or returns garbage. Both give a valid run; the LLM only
chooses between options the code made. Check the model first with `python ops/vllm/healthcheck.py`
(see `ops/vllm/runbook.md`); a wrong `LLM_BASE_URL` silently gives a rules-only run. Keys live in `.env` only (git-ignored); never put them in a file that is committed.

**Exit code.** 0 if every scenario ran and has no violations; 1 otherwise.

### Where the output goes

| File | What |
|---|---|
| `runs/<scenario>/actions.jsonl` | the toolkit's own action log (what the organizers score) |
| `runs/P2/<scenario>/actions.jsonl` | a copy at the toolkit's default path (relative to the folder you ran in) |
| `runs/<scenario>/decisions.jsonl` | our log: what each agent did and why |
| `runs/<scenario>/summary.json` | the toolkit's `run_summary()` plus our honest turnaround and loop counters |

Read `summary.json` like this: `violations` must be `[]`; `vessels_departed` / `vessels_total` is the main
score; `avg_turnaround_h` is the toolkit's figure and `honest_turnaround_h` also counts unfinished ships up to
the end of the window; `loop.exceptions`, `loop.breaches` and `loop.actions_refused` must be 0.
If a scenario cannot start at all, `summary.json` has `"crashed": true` and the reason.

### For the organizers: starting us from Python

If you build the `TerminalAPI` yourself, call one function:

```python
from run import run_agent
from vendor import TerminalAPI

api = TerminalAPI("hidden_01", data_xlsx="P2_Container_Terminal_Data.xlsx",
                  key_xlsx="P2_Container_Terminal_ANSWER_KEY.xlsx")      # log_path is yours to choose
summary = run_agent(api, "runs/hidden_01")        # plays the whole scenario, returns the summary dict
```

`run_agent(api, run_dir, agent="multi")` writes `decisions.jsonl` and `summary.json` into `run_dir`. The
actions log goes where the `api` was told to write it (the toolkit default if you give no `log_path`).
It does not raise; a scenario that goes wrong still ends with a summary.

### Checks before a change goes in

```bash
bash tools/check.sh                              # tests, scorecard, variants, red team, honest play, secrets
python tools/eval_all.py --all --no-llm          # the scorecard alone
bash tools/clean_clone_check.sh data/P2_Container_Terminal_Data.xlsx [KEY]   # fresh clone, fresh venv, then run
python tools/perturb.py --n 1 --malformed 0 --hidden-key --out runs/mock     # a mock answer key, to rehearse hidden runs
```

`clean_clone_check.sh` tests **committed** work only (it clones). It runs `practice_01 --no-llm`, every hidden
scenario in the key you give it, and `practice_01` with the LLM endpoint unreachable (proves the fall-back).

### Freeze checklist (Sun 15 Nov)

- [ ] `bash tools/check.sh` green on `main` (CI "check" green on the last merge)
- [ ] Full variant suite green: `python tools/perturb.py --n 20 --seed 1 --clean`, then `python tools/eval_all.py --all --no-llm --variants 99 --fcfs`
- [ ] `bash tools/clean_clone_check.sh <workbook> <mock key>` green on Sat 14 **and** Sun 15, on the commit to be tagged
- [ ] LLM endpoint off: the check above includes the fall-back run; it must pass
- [ ] `.env` is not in git; `python tools/check_secrets.py` green
- [ ] A teammate who did not write code ran the "Run it" section above, nothing else, and got `viol 0`
- [ ] Judges can open the repo (access checked); README lists the exact command they run
- [ ] Tag the release (the release owner runs `git tag` and pushes it); no merges to `main` after the tag

## Layout

```
run.py            entry point (CORE-2)
core/             types, state, events, validate, executor, loop, log, ...
agents/           berth, crane, crane_moves, yard, trucks, admission, stall, options
coordinator.py    negotiation + admission control (CORE-5)
guard.py          hard-rule checks (SAFE-1)
llm/              client, supervisor, voice, operator
config/           params.yaml (tunable parameters)
ops/vllm/         model server scripts, runbook.md (no secrets)
baselines/        single-agent comparison (CORE-8)
tools/            eval_all, perturb, compare, timeline, check.sh
vendor/           organizer toolkit, unmodified
tests/
```

## Model choice (LLM-3)

Bake-off on real decision points from our runs (`python -m llm.bakeoff.collect`, then `run_bakeoff` per model and `choose`). Injection strings are the provisional set in `llm/bakeoff/injections.txt` until SAFE-5's corpus exists.

| Model | Revision | Valid | Agrees with rules | Injections resisted | p50 / p95 s | 16 parallel s | tok/s | Same on repeat |
|---|---|---|---|---|---|---|---|---|
| qwen | `017b9c7af6b5` | 1.0 | 1.0 (22 informative) | 1.0 (29) | 2.52 / 3.078 | 7.42 | 21.5 | 1.0 |
| gptoss | `6cee5e81ee83` | 0.94 | 1.0 (22 informative) | 0.828 (29) | 0.86 / 1.875 | 4.94 | 145.3 | 0.7 |

**Chosen: qwen @ `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`**, the only model that passes every gate (quality 1.00); not eligible: gptoss (valid 0.94 < 0.98, injection resisted 0.828 < 0.9, repeat 0.7 < 0.9). Set `LLM_CALL_BUDGET=87` (one call per tick at most; 5 min per scenario minus ~30 s of simulation, at p95 3.078 s).
