"""Unattended job runner: executes jobs from jobs/queue.yaml one at a time.

* per-example outputs appended to results/<run_id>/outputs.jsonl as they finish; finished ids skipped on restart
* results/<run_id>/meta.json: run_id, git commit, job config, global config, seed, server tag, prompt version, times
* logs/<run_id>.log per run; ledger rows appended to COMPUTE_LEDGER.md and logs/ledger.jsonl
* honours DEADLINE.txt: refuses jobs whose projected end passes NO_NEW_GPU_JOBS_AFTER (unless bugfix_rerun),
  stops at HARD_STOP; test-split jobs refused unless PROTOCOL.md is frozen (hash in COMPUTE_LEDGER.md matches)
  and every test run is appended to logs/test_runs.log
* guards: thermal policy v2 in Guard.run (CPU zone >= 78 C or GPU >= 80 C -> pause all LLM requests) and on battery
Usage: .venv/Scripts/python.exe scripts/run_queue.py [--queue jobs/queue.yaml] [--only RUN_ID] [--once]
"""
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Thermal limit (user report: CPU near 100 C): cap CPU threads used by torch / BLAS / FAISS.
CPU_THREADS = int(os.environ.get("RAGV_CPU_THREADS", "2"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(CPU_THREADS)

import llm_server  # noqa: E402
from ragverify import data as D  # noqa: E402
from ragverify import prompts  # noqa: E402
from ragverify.core import LLM, Counter, KVCache, Stats, CACHE_DIR  # noqa: E402

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def now() -> dt.datetime:
    return dt.datetime.now(IST)


def deadlines() -> dict:
    d = {}
    for line in (ROOT / "DEADLINE.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return {k: dt.datetime.fromisoformat(v) for k, v in d.items() if "T" in v}


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "jobs"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def protocol_frozen(name: str = "PROTOCOL.md") -> bool:
    """Frozen iff the ledger records a SHA-256 and logs/protocol_freeze/PROTOCOL.md (the frozen copy) has that hash.
    Later dated Deviations entries in PROTOCOL.md therefore do not unfreeze it."""
    led = (ROOT / "COMPUTE_LEDGER.md").read_text(encoding="utf-8")
    frozen = ROOT / "logs" / "protocol_freeze" / name
    if "not frozen yet" in led or not frozen.exists():
        return False
    return hashlib.sha256(frozen.read_bytes()).hexdigest() in led


class Guard(threading.Thread):
    """GPU temperature / AC-power guard. Workers wait on `ok` before each example."""

    def __init__(self, log: logging.Logger):
        super().__init__(daemon=True)
        self.ok = threading.Event()
        self.ok.set()
        self.log, self.hot = log, 0
        self.paused_s = 0.0

    def temp(self) -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            return int(out.splitlines()[0])
        except Exception:  # noqa: BLE001
            return -1

    def cpu_zone_temp(self) -> float:
        """ACPI thermal zone \_tz.tsz0 (Kelvin perf counter, readable without admin) used as the CPU-heat proxy."""
        try:
            cmd = ("((Get-Counter '\Thermal Zone Information(*)\Temperature').CounterSamples | "
                   "Where-Object { $_.InstanceName -like '*tsz0*' } | Select-Object -First 1).CookedValue")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30).stdout
            return float(out.strip()) - 273.15
        except Exception:  # noqa: BLE001
            return -1.0

    def _stream(self, cmd: list[str], key: str) -> None:
        """Keep one long-lived sensor process per source (cheap: no process spawn per check) and store its latest value."""
        while True:
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                atexit.register(p.kill)  # sensor processes otherwise outlive the runner (seen 2026-11-06)
                for line in p.stdout:
                    line = line.strip()
                    try:
                        v = float(line)
                    except ValueError:
                        continue
                    self.latest[key] = (v - 273.15 if key == "zone" and v > 200 else v, time.time())
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)

    def run(self):
        # Thermal policy v2 (2026-11-05 22:10, after the user reported CPU ~95 C; v1 checked every ~22 s and only stopped
        # new examples, so the CPU zone overshot to 90-99 C): sensors stream every 1-2 s; pause at once if the CPU zone
        # >= 78 C or GPU >= 80 C (or CPU load >= 60% twice, or on battery); a pause blocks every new LLM request
        # (core.THERMAL_OK), not just new examples; resume only after >= 20 s and CPU zone <= 65 C, GPU <= 68 C,
        # load <= 40%. A stale or missing sensor reading counts as hot (fail safe).
        # v3 (2026-11-06 20:20, user decision "Share CPU, keep temp limits"): the CPU-load rule is removed (the user's own
        # notebooks held the runs for hours at 100% load while the CPU zone stayed ~55-60 C with turbo capped);
        # temperature, staleness and battery rules are unchanged. Load is still logged.
        # REVERTED 2026-11-06 20:33: with the user's notebooks at 100% CPU, the CPU zone still jumped from 65 C to
        # 86.9 C and 90.9 C between 1 s readings, so the load rule is back (v2 behaviour).
        from ragverify import core
        self.latest = {}
        zone_cmd = ["powershell", "-NoProfile", "-Command",
                    "Get-Counter '\\Thermal Zone Information(*)\\Temperature' -SampleInterval 1 -Continuous | ForEach-Object "
                    "{ ($_.CounterSamples | Where-Object { $_.InstanceName -like '*tsz0*' } | Select-Object -First 1).CookedValue }"]
        gpu_cmd = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", "-lms", "1000"]
        threading.Thread(target=self._stream, args=(zone_cmd, "zone"), daemon=True).start()
        threading.Thread(target=self._stream, args=(gpu_cmd, "gpu"), daemon=True).start()
        time.sleep(8)
        while True:
            now_t = time.time()
            z, zt = self.latest.get("zone", (999.0, 0))
            t, tt = self.latest.get("gpu", (999.0, 0))
            if now_t - zt > 15:
                z = 999.0
            if now_t - tt > 15:
                t = 999.0
            batt = psutil.sensors_battery()
            on_batt = batt is not None and not batt.power_plugged
            cpu = psutil.cpu_percent(interval=1)
            self.hot = self.hot + 1 if cpu >= 60 else 0
            if self.ok.is_set() and (z >= 78 or t >= 80 or self.hot >= 2 or on_batt):
                self.ok.clear()
                core.THERMAL_OK.clear()
                self.log.warning(f"PAUSE: gpu_temp={t:.0f} cpu_zone={z:.1f} cpu_load={cpu} on_battery={on_batt}")
                self.pause_t0 = now_t
            elif (not self.ok.is_set() and now_t - self.pause_t0 >= 20 and t <= 68 and z <= 65 and cpu <= 40
                  and not on_batt):
                self.ok.set()
                core.THERMAL_OK.set()
                self.paused_s += now_t - self.pause_t0
                self.log.warning(f"RESUME: gpu_temp={t:.0f} cpu_zone={z:.1f} cpu_load={cpu}")
            time.sleep(1)


def ledger_append(row: dict) -> None:
    with (ROOT / "logs" / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    md = ROOT / "COMPUTE_LEDGER.md"
    line = (f"| {row['run_id']} | {row['launched'][:16]} | {row['projected_min']} min | {row['projected_end'][:16]} | "
            f"{row['wall_s']/60:.1f} min | {row['gpu_s']:.0f} | {row['llm_calls']} | {row['prompt_tokens']} | "
            f"{row['completion_tokens']} | {row['status']} | {row.get('notes', '')} |\n")
    with md.open("a", encoding="utf-8") as f:
        f.write(line)


def run_job(job: dict, gcfg: dict, log: logging.Logger, guard: Guard) -> str:
    import torch
    torch.set_num_threads(CPU_THREADS)
    try:
        import faiss
        faiss.omp_set_num_threads(CPU_THREADS)
    except Exception:  # noqa: BLE001
        pass
    run_id = job["id"]
    out_dir = ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        return "done-before"
    dl = deadlines()
    proj_end = now() + dt.timedelta(minutes=float(job.get("projected_minutes", 60)))
    if proj_end > dl["NO_NEW_GPU_JOBS_AFTER"] and not job.get("bugfix_rerun"):
        log.error(f"REFUSED {run_id}: projected end {proj_end:%Y-%m-%d %H:%M} after NO_NEW_GPU_JOBS_AFTER")
        return "refused-deadline"
    if now() > dl["HARD_STOP"]:
        return "refused-hardstop"
    if job["split"] in ("test", "fresh"):
        if not protocol_frozen() or (job["split"] == "fresh" and not protocol_frozen("PROTOCOL_v2.md")):
            log.error(f"REFUSED {run_id}: test split but PROTOCOL.md is not frozen")
            return "refused-protocol"
        with (ROOT / "logs" / "test_runs.log").open("a", encoding="utf-8") as f:
            f.write(f"{now().isoformat(timespec='seconds')}\t{run_id}\t{git_commit()}\t{json.dumps(job, sort_keys=True)}\n")

    cfg = dict(gcfg, **job.get("config_overrides", {}))
    examples = D.load(job["dataset"], job["split"], job.get("n"))
    done = set()
    outp = out_dir / "outputs.jsonl"
    if outp.exists():
        for line in outp.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [e for e in examples if e["id"] not in done]
    launched = now()
    log.info(f"START {run_id}: {len(todo)}/{len(examples)} examples to run; projected {job.get('projected_minutes')} min")

    st = llm_server.start(job["server"])
    stats = Stats()
    llm = LLM(st["tag"], f"http://127.0.0.1:{st['port']}", KVCache(CACHE_DIR / "llm.sqlite"), stats)
    from ragverify.methods import REGISTRY, Ctx
    retr = None
    if job.get("needs_retrieval", True):
        from ragverify.retrieval import Retriever
        retr = Retriever(device=cfg.get("retrieval_device", "cpu"))
    verifier = None
    if job.get("verifier"):
        from ragverify.verifier import make
        verifier = make(job["verifier"], device=cfg.get("verifier_device", "cuda"))
    crag_eval = None
    if job["method"] == "crag":
        from ragverify.crag import CragEvaluator
        crag_eval = CragEvaluator(device=cfg.get("verifier_device", "cuda"))
    ctx = Ctx(llm, retr, cfg, verifier=verifier, crag_eval=crag_eval, seed=job.get("seed", 0))
    if job["method"] == "crag":
        from ragverify.crag import CRAG
        method = CRAG(ctx, **job.get("method_args", {}))
    elif job["method"] == "rarr":
        from ragverify.rarr import RARR
        method = RARR(ctx, **job.get("method_args", {}))
    elif job["method"] == "selfrag":
        from ragverify.selfrag import SelfRAG
        method = SelfRAG(ctx, **job.get("method_args", {}))
    else:
        method = REGISTRY[job["method"]](ctx, **job.get("method_args", {}))

    meta = {"run_id": run_id, "git_commit": git_commit(), "job": job, "config": cfg, "seed": job.get("seed", 0),
            "server": {k: st[k] for k in ("tag", "ctx", "parallel", "profile")}, "prompt_version": prompts.PROMPT_VERSION,
            "started": launched.isoformat(timespec="seconds"), "n_examples": len(examples),
            "retriever_version": getattr(retr, "version", None), "verifier": job.get("verifier")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    lock = threading.Lock()
    t0 = time.time()
    n_err = 0

    def one(ex):
        guard.ok.wait()
        if now() > deadlines()["HARD_STOP"]:
            raise SystemExit("hard stop")
        cnt = Counter()
        t = time.perf_counter()
        rec = method.run(ex, cnt)
        rec.update({"id": ex["id"], "dataset": ex["dataset"], "split": ex["split"], "cost": cnt.d,
                    "seconds": time.perf_counter() - t})
        with lock, outp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return ex["id"]

    with ThreadPoolExecutor(max_workers=int(job.get("workers", st["parallel"]))) as ex_pool:
        futs = [ex_pool.submit(one, e) for e in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception:  # noqa: BLE001
                n_err += 1
                log.error(traceback.format_exc(limit=3))
            if i % 25 == 0:
                log.info(f"{run_id}: {i}/{len(todo)} done, {n_err} errors, {time.time()-t0:.0f}s")
    wall = time.time() - t0
    n_out = sum(1 for _ in outp.open(encoding="utf-8")) if outp.exists() else 0
    status = "complete" if n_out >= len(examples) else f"partial {n_out}/{len(examples)}"
    meta.update({"finished": now().isoformat(timespec="seconds"), "wall_s": wall, "stats": stats.as_dict(),
                 "errors": n_err, "status": status, "paused_s": guard.paused_s})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ledger_append({"run_id": run_id, "launched": launched.isoformat(timespec="seconds"),
                   "projected_min": job.get("projected_minutes"), "projected_end": proj_end.isoformat(timespec="seconds"),
                   "wall_s": wall, "gpu_s": wall, **stats.as_dict(), "status": status, "notes": job.get("notes", "")})
    if status == "complete":
        (out_dir / "DONE").write_text(now().isoformat(timespec="seconds"), encoding="utf-8")
    if verifier is not None:
        verifier.unload()
    # free GPU memory held by side models (verifier, CRAG evaluator, retrieval encoders) before the next job
    import gc
    import torch
    del method, ctx, retr, verifier, crag_eval
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"END {run_id}: {status}, wall {wall/60:.1f} min, stats {stats.as_dict()}")
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(ROOT / "jobs" / "queue.yaml"))
    ap.add_argument("--only")
    ap.add_argument("--once", action="store_true", help="exit when the queue is exhausted (default)")
    a = ap.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "runner.pid").write_text(str(os.getpid()))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "runner.log", encoding="utf-8"), logging.StreamHandler()])
    log = logging.getLogger("runner")
    try:  # heat control: this process (and the workers it runs) stays on 4 logical cores; timing only
        psutil.Process().cpu_affinity(llm_server.CPU_SET)
    except Exception:  # noqa: BLE001
        pass
    guard = Guard(log)
    guard.start()
    gcfg = yaml.safe_load((ROOT / "jobs" / "config.yaml").read_text(encoding="utf-8"))
    jobs = yaml.safe_load(Path(a.queue).read_text(encoding="utf-8")) or []
    for job in jobs:
        if a.only and job["id"] != a.only:
            continue
        fh = logging.FileHandler(ROOT / "logs" / f"{job['id']}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
        try:
            (ROOT / "logs" / "runner_current.json").write_text(json.dumps({"job": job["id"], "since": now().isoformat(), "queue": str(a.queue)}))
            run_job(job, gcfg, log, guard)
        except SystemExit as e:
            log.error(f"STOP: {e}")
            break
        except Exception:  # noqa: BLE001
            log.error(f"JOB FAILED {job['id']}:\n{traceback.format_exc()}")
        finally:
            log.removeHandler(fh)
    (ROOT / "logs" / "runner_current.json").write_text(json.dumps({"job": None, "since": now().isoformat(), "queue": str(a.queue)}))
    log.info("queue exhausted")


if __name__ == "__main__":
    main()
