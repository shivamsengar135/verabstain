"""Print a <=15-line progress summary of the unattended queue. Usage: .venv/Scripts/python.exe scripts/status.py"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def main() -> None:
    now = dt.datetime.now(IST)
    dl = {}
    for line in (ROOT / "DEADLINE.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            try:
                dl[k] = dt.datetime.fromisoformat(v)
            except ValueError:
                pass
    left = dl["HARD_STOP"] - now
    lines = [f"RAG-Verify status {now:%m-%d %H:%M} | hard stop in {left.total_seconds()/3600:.1f} h | "
             f"no new GPU jobs after {dl['NO_NEW_GPU_JOBS_AFTER']:%m-%d %H:%M}"]
    pidf = ROOT / "logs" / "runner.pid"
    alive = pidf.exists() and psutil.pid_exists(int(pidf.read_text()))
    cur = {}
    if (ROOT / "logs" / "runner_current.json").exists():
        cur = json.loads((ROOT / "logs" / "runner_current.json").read_text())
    lines.append(f"runner: {'ALIVE pid ' + pidf.read_text() if alive else 'not running'} | current job: {cur.get('job')} since {str(cur.get('since', ''))[11:16]}")
    q = Path(cur.get("queue") or ROOT / "jobs" / "queue_pilot.yaml")
    jobs = yaml.safe_load(q.read_text(encoding="utf-8")) if q.exists() else []
    done = sum(1 for j in jobs if (ROOT / "results" / j["id"] / "DONE").exists())
    lines.append(f"queue: {done}/{len(jobs)} jobs done")
    if cur.get("job"):
        rd = ROOT / "results" / cur["job"]
        n = sum(1 for _ in (rd / "outputs.jsonl").open(encoding="utf-8")) if (rd / "outputs.jsonl").exists() else 0
        tot = json.loads((rd / "meta.json").read_text())["n_examples"] if (rd / "meta.json").exists() else "?"
        lines.append(f"progress: {cur['job']} {n}/{tot}")
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        lines.append(f"gpu: {g}")
    except Exception:  # noqa: BLE001
        pass
    b = psutil.sensors_battery()
    lines.append(f"power: {'AC' if b is None or b.power_plugged else 'BATTERY'} | free RAM {psutil.virtual_memory().available/2**30:.1f} GB")
    led = ROOT / "logs" / "ledger.jsonl"
    if led.exists():
        rows = [json.loads(l) for l in led.open(encoding="utf-8")]
        lines.append(f"ledger: {len(rows)} runs, {sum(r['gpu_s'] for r in rows)/3600:.2f} GPU-h used of 50")
    log = ROOT / "logs" / "runner.log"
    if log.exists():
        tail = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if "PAUSE" in l or "ERROR" in l or "REFUSED" in l][-3:]
        lines += [f"! {l[:150]}" for l in tail]
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
