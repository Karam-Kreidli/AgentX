"""SAFE-7, part 1: turn one run's logs into timeline.json + a self-contained timeline.html.

    python tools/timeline.py runs/practice_02            # writes runs/practice_02/timeline.{json,html}
    python tools/timeline.py runs/practice_02 --out d/   # writes into d/

Reads only the run's own log files (SAFE-3): ticks.jsonl, decisions.jsonl, negotiation.jsonl,
supervisor_feed.txt, summary.json, run_meta.json. Never the workbook, never the simulator.
Every file is optional and every bad line is skipped and counted, so this never crashes on a
half-written run (a crashed or stopped run is exactly the one you want to look at).

timeline.json (schema "agentx.timeline/1") is the one data file the Sun 15 demo website reads:
  meta        run, scenario, git sha, model, hours covered, skipped-line counts
  kpis        summary.json as written by the loop
  ships       per call: status intervals [{status, berth, t0, t1}] in sim hours
  cranes      per crane: intervals [{status, call, t0, t1}]
  blocks      per block: capacity + fill points [[t, used]] (only where it changed)
  events      ship / crane status changes derived from consecutive ticks
  actions     state-changing actions that went through (agent, choice, reason, t)
  refusals    actions the guard refused, or the simulator rejected
  escalations escalate decisions (issue + safe default taken)
  alerts      loop notes: CRITICAL, breaches, exceptions, clock problems, mismatches
  negotiation request / grant records with a plain-English line
  feed        supervisor feed sentences
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA = "agentx.timeline/1"
MAX_REASON = 300
LOOP_AGENTS = {"loop", "skeleton-loop", "invariants", "safety"}
ALERT_CHOICES = {"CRITICAL", "breach", "exception", "clock", "mismatch", "watchdog", "timeout", "budget"}


# ---------------------------------------------------------------- reading
def _jsonl(path: Path, skipped: dict) -> Iterator[dict]:
    if not path.is_file():
        return
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    rec = None
                if isinstance(rec, dict):
                    yield rec
                else:
                    skipped[path.name] = skipped.get(path.name, 0) + 1
    except OSError:
        skipped[path.name] = skipped.get(path.name, 0) + 1


def _json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _s(x: Any, cap: int = MAX_REASON) -> str:
    s = "" if x is None else str(x)
    return s if len(s) <= cap else s[: cap - 1] + "…"


# ---------------------------------------------------------------- building
class _Intervals:
    """Run-length intervals of a state per id: a new interval starts whenever the key changes."""

    def __init__(self) -> None:
        self.out: dict[str, list[dict]] = {}

    def see(self, ident: str, t: float, key: dict) -> Optional[dict]:
        """Record `key` for `ident` at time t; returns the previous key if it changed."""
        iv = self.out.setdefault(ident, [])
        if iv and all(iv[-1].get(k) == v for k, v in key.items()):
            iv[-1]["t1"] = t
            return None
        prev = {k: iv[-1].get(k) for k in key} if iv else None
        if iv:
            iv[-1]["t1"] = t
        iv.append({**key, "t0": t, "t1": t})
        return prev or {}


def _ticks(run: Path, tl: dict, skipped: dict) -> None:
    ships, cranes = _Intervals(), _Intervals()
    blocks: dict[str, dict] = {}
    last_t = None
    first = True
    for rec in _jsonl(run / "ticks.jsonl", skipped):
        t = _num(rec.get("t"))
        if t is None:
            skipped["ticks.jsonl"] = skipped.get("ticks.jsonl", 0) + 1
            continue
        last_t = t
        if first:
            tl["meta"]["start"] = rec.get("now")
            first = False
        tl["meta"]["end"] = rec.get("now")
        for sh in rec.get("ships") or []:
            if not isinstance(sh, dict) or sh.get("id") is None:
                continue
            ident = _s(sh["id"], 40)
            prev = ships.see(ident, t, {"status": sh.get("status"), "berth": sh.get("berth")})
            if prev and prev.get("status") != sh.get("status"):
                where = f" at {sh.get('berth')}" if sh.get("berth") else ""
                tl["events"].append({"t": t, "kind": "ship", "id": ident,
                                     "text": f"{ident} {prev.get('status')} → {sh.get('status')}{where}"})
        for c in rec.get("cranes") or []:
            if not isinstance(c, dict) or c.get("id") is None:
                continue
            ident = _s(c["id"], 20)
            prev = cranes.see(ident, t, {"status": c.get("status"), "call": c.get("call")})
            if prev and prev.get("status") != c.get("status") and "out_of_service" in (prev.get("status"), c.get("status")):
                tl["events"].append({"t": t, "kind": "crane", "id": ident,
                                     "text": f"{ident} {prev.get('status')} → {c.get('status')}"})
        for bid, val in (rec.get("blocks") or {}).items():
            if not (isinstance(val, (list, tuple)) and len(val) == 2):
                continue
            used, cap = _num(val[0]), _num(val[1])
            if used is None:
                continue
            b = blocks.setdefault(_s(bid, 10), {"slots": cap, "fill": []})
            if not b["fill"] or b["fill"][-1][1] != used:
                b["fill"].append([t, used])
    if last_t is not None:      # close every block series at the last tick
        for b in blocks.values():
            if b["fill"] and b["fill"][-1][0] != last_t:
                b["fill"].append([last_t, b["fill"][-1][1]])
    tl["meta"]["hours"] = last_t
    tl["ships"] = [{"id": k, "intervals": v} for k, v in ships.out.items()]
    tl["cranes"] = [{"id": k, "intervals": v} for k, v in cranes.out.items()]
    tl["blocks"] = [{"id": k, **v} for k, v in sorted(blocks.items())]


def _decisions(run: Path, tl: dict, skipped: dict) -> None:
    for d in _jsonl(run / "decisions.jsonl", skipped):
        base = {"t": _num(d.get("t")), "tick": d.get("tick"), "agent": _s(d.get("agent"), 40),
                "choice": _s(d.get("choice"), 60), "reason": _s(d.get("reason"))}
        res = d.get("tool_result")
        if d.get("guard_ok") is False:
            tl["refusals"].append({**base, "by": "guard", "why": _s(d.get("guard_reason"))})
        elif base["choice"] == "escalate":
            tl["escalations"].append(base)
        elif base["agent"] in LOOP_AGENTS or base["choice"] in ALERT_CHOICES:
            tl["alerts"].append(base)
        elif isinstance(res, dict) and (res.get("ok") is False or res.get("error") is not None):
            tl["refusals"].append({**base, "by": "simulator", "why": _s(res.get("error") or res.get("reason"))})
        else:
            tl["actions"].append({**base, "backend": _s(d.get("backend"), 20)})


def _negotiation(run: Path, tl: dict, skipped: dict) -> None:
    for r in _jsonl(run / "negotiation.jsonl", skipped):
        call, res, amt = _s(r.get("call_id"), 40), _s(r.get("resource"), 40), r.get("amount")
        if r.get("kind") == "grant":
            line = r.get("sentence") or (f"{r.get('status')} {r.get('granted')}/{amt} {res} for {call}. "
                                         f"{r.get('grant_reason') or ''}")
            who = f"coordinator → {_s(r.get('agent'), 40)}"
        else:
            line = r.get("sentence") or f"{call} asks for {amt} {res}. {r.get('reason') or ''}"
            who = f"{_s(r.get('agent'), 40)} → coordinator"
        tl["negotiation"].append({"t": _num(r.get("t")), "tick": r.get("tick"), "kind": _s(r.get("kind"), 10),
                                  "status": r.get("status"), "who": who, "text": _s(line).strip()})


def _feed(run: Path, tl: dict) -> None:
    p = run / "supervisor_feed.txt"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines() if p.is_file() else []
    except OSError:
        lines = []
    tl["feed"] = [_s(x) for x in lines if x.strip()]


def build(run_dir: Path | str) -> dict:
    run = Path(run_dir)
    skipped: dict[str, int] = {}
    meta = _json(run / "run_meta.json")
    tl: dict = {
        "schema": SCHEMA,
        "meta": {"run": meta.get("run") or run.name, "git_sha": meta.get("git_sha"),
                 "llm_model": meta.get("llm_model"), "started_utc": meta.get("started_utc"),
                 "start": None, "end": None, "hours": None},
        "kpis": _json(run / "summary.json"),
        "ships": [], "cranes": [], "blocks": [], "events": [], "actions": [], "refusals": [],
        "escalations": [], "alerts": [], "negotiation": [], "feed": [],
    }
    for step in (lambda: _ticks(run, tl, skipped), lambda: _decisions(run, tl, skipped),
                 lambda: _negotiation(run, tl, skipped), lambda: _feed(run, tl)):
        try:
            step()
        except Exception as e:       # never crash: keep what we have and say what failed
            tl["meta"].setdefault("errors", []).append(f"{type(e).__name__}: {e}"[:200])
    tl["meta"]["scenario"] = tl["kpis"].get("scenario") or tl["meta"]["run"]
    tl["meta"]["skipped_lines"] = skipped
    return tl


# ---------------------------------------------------------------- html
_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Timeline · __TITLE__</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mute:#6b6b66;--card:#fff;--line:#e2e1dc;
--expected:#d7d6d0;--arrived:#e8a33d;--berthed:#7aa6d8;--working:#2f6fb3;--sailing:#79b98a;--departed:#c9e3cf;
--oos:#c0392b;--ok:#2e7d4f;--warn:#b26a00;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mute:#9c9b95;--card:#1f1f1d;--line:#34332f;
--expected:#3a3936;--departed:#2c4733}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 8px}.mute{color:var(--mute)}
.kpis{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:120px}
.kpi b{display:block;font-size:18px}.kpi span{font-size:12px;color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto}
svg text{fill:var(--fg);font-size:11px}.legend span{display:inline-block;margin-right:12px;font-size:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
ul.log{list-style:none;margin:0;padding:0;max-height:360px;overflow:auto;font-size:13px}
ul.log li{padding:4px 0;border-bottom:1px solid var(--line)}ul.log .t{color:var(--mute);font-variant-numeric:tabular-nums;margin-right:6px}
.tag{font-size:11px;padding:1px 6px;border-radius:9px;border:1px solid var(--line);margin-right:4px}
.bad{color:var(--bad)}.warn{color:var(--warn)}.ok{color:var(--ok)}
#cursor{position:sticky;top:0;background:var(--bg);padding:6px 0;z-index:2}input[type=range]{width:100%}
</style></head><body>
<h1>Run timeline · <span id="title"></span></h1>
<div class="mute" id="sub"></div>
<div class="kpis" id="kpis"></div>
<div id="cursor"><label class="mute">Sim time <b id="clock"></b></label><input type="range" id="slider" min="0" max="1" step="0.25" value="0"></div>
<h2>Ships</h2><div class="legend" id="legend"></div><div class="card"><svg id="ships"></svg></div>
<h2>Quay cranes</h2><div class="card"><svg id="cranes"></svg></div>
<h2>Yard blocks (fill at cursor)</h2><div class="card"><svg id="blocks"></svg></div>
<div class="cols">
<div><h2>Guard refusals &amp; alerts</h2><div class="card"><ul class="log" id="refusals"></ul></div></div>
<div><h2>Escalations</h2><div class="card"><ul class="log" id="escalations"></ul></div></div>
<div><h2>Negotiation</h2><div class="card"><ul class="log" id="negotiation"></ul></div></div>
<div><h2>Events</h2><div class="card"><ul class="log" id="events"></ul></div></div>
<div><h2>Actions</h2><div class="card"><ul class="log" id="actions"></ul></div></div>
<div><h2>Supervisor feed</h2><div class="card"><ul class="log" id="feed"></ul></div></div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
const D=JSON.parse(document.getElementById('data').textContent);
const H=Math.max(1,D.meta.hours||1), NS='http://www.w3.org/2000/svg';
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const clk=t=>t==null?'?':'Day '+(Math.floor(t/24)+1)+' '+String(Math.floor(t%24)).padStart(2,'0')+':'+String(Math.round((t%1)*60)).padStart(2,'0');
$('title').textContent=D.meta.scenario||D.meta.run;
$('sub').textContent=[D.meta.start&&('from '+D.meta.start),D.meta.end&&('to '+D.meta.end),D.meta.git_sha&&('git '+D.meta.git_sha),D.meta.llm_model&&('model '+D.meta.llm_model)].filter(Boolean).join(' · ');
const K=D.kpis||{},L=K.loop||{};
[['Departed',K.vessels_departed!=null?K.vessels_departed+'/'+K.vessels_total:null],['Avg turnaround h',K.avg_turnaround_h],
 ['Honest turnaround h',K.honest_turnaround_h],['FCFS baseline h',K.fcfs_baseline_avg_turnaround_h_same_calls],
 ['Crane moves/h',K.crane_moves_per_hour],['Violations',(K.violations||[]).length],['Guard refusals',D.refusals.filter(r=>r.by==='guard').length],
 ['Escalations',D.escalations.length],['Exceptions',L.exceptions]].forEach(([k,v])=>{if(v==null)return;
 const e=document.createElement('div');e.className='kpi';e.innerHTML='<b>'+esc(v)+'</b><span>'+esc(k)+'</span>';$('kpis').appendChild(e);});
const col=s=>({expected:'var(--expected)',arrived:'var(--arrived)',berthed:'var(--berthed)',working:'var(--working)',sailing:'var(--sailing)',departed:'var(--departed)',available:'var(--expected)',out_of_service:'var(--oos)'}[s]||'#999');
['expected','arrived','berthed','working','sailing','departed','out_of_service'].forEach(s=>{$('legend').insertAdjacentHTML('beforeend','<span><i style="background:'+col(s)+'"></i>'+s.replace(/_/g,' ')+'</span>')});
function el(n,a,p){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
const LBL=80,W=Math.max(700,Math.min(1600,H*6)),x=t=>LBL+(t/H)*(W-LBL-10);
function gantt(svg,rows,label,fill){const RH=20;svg.setAttribute('width',W);svg.setAttribute('height',rows.length*RH+24);
 for(let d=0;d<=H;d+=24){el('line',{x1:x(d),x2:x(d),y1:0,y2:rows.length*RH,stroke:'var(--line)'},svg);el('text',{x:x(d)+2,y:rows.length*RH+14},svg).textContent='D'+(d/24+1);}
 rows.forEach((r,i)=>{el('text',{x:0,y:i*RH+14},svg).textContent=r.id;
  r.intervals.forEach(iv=>{const w=Math.max(1,x(iv.t1)-x(iv.t0));const g=el('rect',{x:x(iv.t0),y:i*RH+3,width:w,height:RH-6,rx:2,fill:fill(iv)},svg);
   el('title',{},g).textContent=r.id+': '+label(iv)+' '+clk(iv.t0)+' → '+clk(iv.t1);
   if(w>40&&iv.berth&&iv.status!=='departed'){const tx=el('text',{x:x(iv.t0)+3,y:i*RH+14,style:'fill:#fff'},svg);tx.textContent=iv.berth;}});});
 const cur=el('line',{y1:0,y2:rows.length*RH,stroke:'var(--bad)','stroke-width':1.5},svg);return cur;}
const curs=[gantt($('ships'),D.ships,iv=>iv.status+(iv.berth?' @'+iv.berth:''),iv=>col(iv.status)),
 gantt($('cranes'),D.cranes,iv=>iv.status+(iv.call?' '+iv.call:''),iv=>iv.status==='working'?'var(--working)':col(iv.status))];
function fillAt(b,t){let v=0;for(const p of b.fill){if(p[0]>t)break;v=p[1];}return v;}
function drawBlocks(t){const s=$('blocks');s.innerHTML='';const BW=56,BH=120;s.setAttribute('width',D.blocks.length*(BW+10)+10);s.setAttribute('height',BH+40);
 D.blocks.forEach((b,i)=>{const u=fillAt(b,t),p=b.slots?u/b.slots:0,X=10+i*(BW+10);
  el('rect',{x:X,y:10,width:BW,height:BH,fill:'none',stroke:'var(--line)'},s);
  el('rect',{x:X,y:10+BH*(1-Math.min(1,p)),width:BW,height:BH*Math.min(1,p),fill:p>=0.85?'var(--bad)':p>=0.7?'var(--warn)':'var(--ok)'},s);
  el('text',{x:X+BW/2-10,y:BH+24},s).textContent=b.id+' '+Math.round(p*100)+'%';});}
function list(id,items,fmt){const u=$(id);if(!items.length){u.innerHTML='<li class="mute">none in this run</li>';return;}
 u.innerHTML=items.map(it=>'<li data-t="'+(it.t==null?'':it.t)+'"><span class="t">'+clk(it.t)+'</span>'+fmt(it)+'</li>').join('');}
list('refusals',D.refusals.concat(D.alerts).sort((a,b)=>(a.t||0)-(b.t||0)),r=>r.by?'<span class="tag bad">refused by '+esc(r.by)+'</span><b>'+esc(r.agent)+'</b> '+esc(r.choice)+': '+esc(r.why)+'<div class="mute">'+esc(r.reason)+'</div>':'<span class="tag warn">'+esc(r.choice)+'</span>'+esc(r.reason));
list('escalations',D.escalations,r=>'<span class="tag warn">'+esc(r.agent)+'</span>'+esc(r.reason));
list('negotiation',D.negotiation,r=>'<b>'+esc(r.who)+':</b> '+esc(r.text));
list('events',D.events,r=>'<span class="tag'+(r.kind==='crane'?' bad':'')+'">'+r.kind+'</span>'+esc(r.text));
list('actions',D.actions,r=>'<span class="tag">'+esc(r.agent)+'</span><b>'+esc(r.choice)+'</b> '+esc(r.reason));
const fu=$('feed');fu.innerHTML=D.feed.length?D.feed.map(s=>'<li>'+esc(s)+'</li>').join(''):'<li class="mute">none in this run</li>';
const sl=$('slider');sl.max=H;
function setT(t){$('clock').textContent=clk(t);curs.forEach(c=>{c.setAttribute('x1',x(t));c.setAttribute('x2',x(t));});drawBlocks(t);
 document.querySelectorAll('ul.log li[data-t]').forEach(li=>{const v=parseFloat(li.dataset.t);li.style.opacity=isNaN(v)||v<=t?1:0.35;});}
sl.addEventListener('input',()=>setT(parseFloat(sl.value)));sl.value=H;setT(H);
})();
</script></body></html>
"""


def render_html(tl: dict) -> str:
    data = json.dumps(tl, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/").replace("<!--", "<\\!--")   # cannot close the <script> early
    title = html.escape(str(tl.get("meta", {}).get("scenario") or "run"))
    return _PAGE.replace("__TITLE__", title).replace("__DATA__", data)


def export(run_dir: Path | str, out_dir: Path | str | None = None) -> tuple[Path, Path]:
    tl = build(run_dir)
    out = Path(out_dir) if out_dir else Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp, hp = out / "timeline.json", out / "timeline.html"
    jp.write_text(json.dumps(tl, ensure_ascii=False, indent=1), encoding="utf-8")
    hp.write_text(render_html(tl), encoding="utf-8")
    return jp, hp


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build timeline.json + timeline.html from a run's logs.")
    ap.add_argument("run_dir", nargs="+", help="e.g. runs/practice_02")
    ap.add_argument("--out", help="output directory (default: the run directory)")
    a = ap.parse_args(argv)
    rc = 0
    for rd in a.run_dir:
        if not Path(rd).is_dir():
            print(f"timeline: {rd} is not a directory", file=sys.stderr)
            rc = 1
            continue
        try:
            jp, hp = export(rd, a.out if len(a.run_dir) == 1 else (Path(a.out) / Path(rd).name if a.out else None))
            print(f"timeline: wrote {jp} and {hp}")
        except Exception as e:      # never a traceback for the demo
            print(f"timeline: {rd}: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
