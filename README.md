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
