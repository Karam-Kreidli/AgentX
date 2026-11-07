# AgentX — P2 Container Terminal Orchestra

Multi-agent planner for the AgentX@CCI Problem 2 container terminal simulator.
Read **GROUND_RULES.md** before writing code; task specs are in **TASKS.md**,
shared types and interfaces in **CONTRACTS.md**.

## Setup (Python 3.11)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # fill in keys from the private chat
python -c "import core.types"      # CORE-1 check
python -c "from vendor import TerminalAPI; TerminalAPI('practice_01', data_xlsx='data/P2_Container_Terminal_Data.xlsx')"
```

`data/P2_Container_Terminal_Data.xlsx` is the official workbook. Only `run.py`
passes its path to the toolkit; agent code never opens it.

## Layout

```
run.py            entry point (CORE-2)
core/             types, state, events, validate, executor, loop, log, ...
agents/           berth, crane, crane_moves, yard, trucks, admission, stall, options
coordinator.py    negotiation + admission control (CORE-5)
guard.py          hard-rule checks (SAFE-1)
llm/              client, supervisor, voice, operator
ops/vllm/         model server scripts (no secrets)
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
