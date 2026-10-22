#!/usr/bin/env bash
# Runs the robots stack on this station inside a tmux session, so it survives the SSH session and can be watched
# from anywhere on the station. Two ways to run: `serve` keeps the station agent up (Start/Stop/Home over HTTP on
# :8091; the arms rest until a start) and is what a station normally runs; `start` is
# the old one-shot loop that begins moving as soon as the first chunk lands. Logs: ~/robots-station.log for serve,
# ~/robots-run.log for start.
#
#   debug_scripts/station_run.sh serve <config.yaml>    # starts the station agent (a previous session is stopped first)
#   debug_scripts/station_run.sh start <config.yaml>    # starts the one-shot loop (a previous session is stopped first)
#   debug_scripts/station_run.sh stop                    # Ctrl-C into the session (safe hold / disengage), then kill it
#   debug_scripts/station_run.sh home <config.yaml>     # moves both arms to home (zeros, grippers open) in the foreground;
#                                                        #   refuse to run it while a loop owns the arms (stop first)
#   debug_scripts/station_run.sh log                     # the last 40 lines of whichever log the session writes
#
# Uses the robots venv next to this script's parent directory (.venv/bin/python -m robots.station / robots.run /
# robots.home). Nothing here touches the arms directly: the modules do, and every shutdown path commands a safe hold
# or disengages them after an ArmError. Keep a hand on the e-stop for the whole run.
#
# Two variables are carried into the session when set (tmux hands a new session its server's environment, not the
# caller's): SSL_CERT_FILE, for a live node behind a certificate the system store does not know (a test rig's
# self-signed one; the deployed site needs none), and OPENCV_FOR_THREADS_NUM, an operator's override of the OpenCV
# pool size robots.cameras otherwise caps itself.
set -euo pipefail

ROBOTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="dreamrun"
RUN_LOG="$HOME/robots-run.log"
STATION_LOG="$HOME/robots-station.log"
# Seconds between the Ctrl-C and killing the session: enough for the loop's safe-hold shutdown to finish.
STOP_GRACE_S=2

usage() {
  sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

check_config() {
  local config="${1:-}"
  [ -n "$config" ] || usage
  [ -f "$config" ] || { echo "no such config: $config" >&2; exit 1; }
  [ -x "$ROBOTS_DIR/.venv/bin/python" ] || { echo "no venv at $ROBOTS_DIR/.venv; run uv sync in $ROBOTS_DIR" >&2; exit 1; }
  echo "$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"
}

launch() {
  # $1: the module to run, $2: the absolute config path, $3: the log file.
  local env="PYTHONUNBUFFERED=1"
  [ -n "${SSL_CERT_FILE:-}" ] && env="$env SSL_CERT_FILE='$(cd "$(dirname "$SSL_CERT_FILE")" && pwd)/$(basename "$SSL_CERT_FILE")'"
  [ -n "${OPENCV_FOR_THREADS_NUM:-}" ] && env="$env OPENCV_FOR_THREADS_NUM='$OPENCV_FOR_THREADS_NUM'"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" -c "$ROBOTS_DIR" \
    "$env .venv/bin/python -m $1 --config '$2' 2>&1 | tee '$3'"
  echo "started $1 (tmux session $SESSION, config $2); log: $3"
  echo "watch: $0 log   |   attach: tmux attach -t $SESSION   |   stop: $0 stop"
}

serve() {
  local config
  config="$(check_config "${1:-}")"
  launch robots.station "$config" "$STATION_LOG"
  echo "API (on this computer): curl http://127.0.0.1:8091/status"
}

start() {
  local config
  config="$(check_config "${1:-}")"
  launch robots.run "$config" "$RUN_LOG"
}

stop() {
  tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
  sleep "$STOP_GRACE_S"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  echo "stopped"
}

home() {
  local config
  config="$(check_config "${1:-}")"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "a $SESSION session is running; stop it first ($0 stop) or press Home in the station UI" >&2
    exit 1
  fi
  (cd "$ROBOTS_DIR" && PYTHONUNBUFFERED=1 .venv/bin/python -m robots.home --config "$config")
}

log() {
  # The station log when the agent wrote more recently, else the run log; whichever exists.
  if [ -f "$STATION_LOG" ] && { [ ! -f "$RUN_LOG" ] || [ "$STATION_LOG" -nt "$RUN_LOG" ]; }; then
    tail -n 40 "$STATION_LOG"
  else
    tail -n 40 "$RUN_LOG"
  fi
}

case "${1:-}" in
  serve) serve "${2:-}" ;;
  start) start "${2:-}" ;;
  stop) stop ;;
  home) home "${2:-}" ;;
  log) log ;;
  *) usage ;;
esac
