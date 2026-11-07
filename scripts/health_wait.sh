#!/usr/bin/env bash
# Overnight monitor: returns when the runner log is silent > 15 min, the watchdog log gains a line, or MAX_MIN pass.
# Prints a short health report. Usage: bash scripts/health_wait.sh [MAX_MIN=40]
cd "$(dirname "$0")/.." || exit 1
MAX_MIN=${1:-40}
start=$(date +%s)
wd0=$(wc -l < logs/watchdog.log)
while true; do
  sleep 60
  now=$(date +%s)
  age=$(( now - $(stat -c %Y logs/runner.log) ))
  wd=$(wc -l < logs/watchdog.log)
  if [ "$age" -gt 900 ]; then reason="runner log silent ${age}s"; break; fi
  if [ "$wd" -ne "$wd0" ]; then reason="watchdog event"; break; fi
  if [ $(( now - start )) -ge $(( MAX_MIN * 60 )) ]; then reason="periodic check"; break; fi
done
echo "REASON: $reason  at $(date +%H:%M)"
tail -3 logs/watchdog.log
cat logs/runner_current.json; echo
grep -E "INFO (START|abl_|sec3b_|quant3bq8_|rob_)" logs/runner.log | tail -1
grep -E "PAUSE" logs/runner.log | awk -v s="$(date -d "-${MAX_MIN} min" "+%Y-%m-%d %H:%M")" '($1" "$2)>=s' \
  | sed -E 's/.*cpu_zone=([0-9.]+).*gpu_temp.*|.*gpu_temp=([0-9]+) cpu_zone=([0-9.]+).*/\2 \3/' | sort -k2 -n | tail -1 | awk '{print "max at pauses: gpu " $1 " cpu_zone " $2}'
grep -cE "REFUSED|JOB FAILED" logs/runner.log
tail -2 logs/watchdog_stderr.log
