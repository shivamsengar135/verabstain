"""Unattended queue chaining: run each queue to completion, one runner at a time.

Relaunches a runner that died (e.g. after a crash) at most MAX_RELAUNCH times per queue, then moves on only when
every job of the queue has results/<id>/DONE. Never starts a second runner while logs/runner.pid is alive.
Does not survive a reboot (no system autostart is installed, by design).
Usage: .venv/Scripts/python.exe scripts/watchdog.py jobs/queue_main.yaml jobs/queue_phase4.yaml
Stop: kill this process (pid in logs/watchdog.pid), then the runner (logs/runner.pid).
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
MAX_RELAUNCH = 5
LOG = ROOT / "logs" / "watchdog.log"


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def runner_alive() -> bool:
    p = ROOT / "logs" / "runner.pid"
    if not p.exists():
        return False
    try:
        pid = int(p.read_text().strip())
        return psutil.pid_exists(pid) and "run_queue.py" in " ".join(psutil.Process(pid).cmdline())
    except (ValueError, psutil.Error):
        return False


def remaining(queue: str) -> list[str]:
    jobs = yaml.safe_load((ROOT / queue).read_text(encoding="utf-8"))
    return [j["id"] for j in jobs if not (ROOT / "results" / j["id"] / "DONE").exists()]


def launch(queue: str) -> None:
    err = open(ROOT / "logs" / "runner_stderr.log", "a", encoding="utf-8")
    p = subprocess.Popen([PY, "scripts/run_queue.py", "--queue", queue], cwd=ROOT, stdout=subprocess.DEVNULL,
                         stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    (ROOT / "logs" / "runner.pid").write_text(str(p.pid))
    log(f"launched runner pid {p.pid} on {queue}")


def main() -> None:
    # Launched windowless (pythonw) so console Ctrl-C events cannot reach it (it died twice on 2026-11-06); keep errors.
    sys.stderr = open(ROOT / "logs" / "watchdog_stderr.log", "a", encoding="utf-8", buffering=1)
    (ROOT / "logs" / "watchdog.pid").write_text(str(os.getpid()))
    for queue in sys.argv[1:]:
        relaunches = 0
        while True:
            left = remaining(queue)
            if not left:
                log(f"{queue}: all jobs DONE")
                break
            if not runner_alive():
                if relaunches >= MAX_RELAUNCH:
                    log(f"{queue}: runner died {relaunches} times; giving up on this queue ({len(left)} jobs left)")
                    break
                time.sleep(30)  # let a runner that is just exiting finish writing
                if runner_alive() or not remaining(queue):
                    continue
                relaunches += 1
                launch(queue)
            time.sleep(60)
    log("watchdog finished")


if __name__ == "__main__":
    main()
