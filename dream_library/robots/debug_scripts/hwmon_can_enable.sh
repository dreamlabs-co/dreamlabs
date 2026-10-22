#!/usr/bin/env bash
# Bounce CAN interface(s) with berr-reporting on + restart-ms 100.
#
# !! This takes the bus DOWN for ~100 ms. Run only when the robot is idle. !!
#
# Why: the udev rule in /etc/udev/rules.d/90-can.rules tries to set restart-ms 100
# but its RUN+= fails (exit 2, see journalctl -u systemd-udevd), so the links come
# up with restart-ms 0 (no auto-recovery from bus-off) and berr-reporting off
# (no per-bit error frames). Changing either requires the link to be down.
#
# Usage: hwmon-can-enable.sh [iface ...]      (default: every can* interface)
# Note: only the plain "up type can bitrate 1000000" / "down" forms are NOPASSWD in
# sudoers, so this script WILL prompt for the sudo password.

set -euo pipefail
BITRATE="${BITRATE:-1000000}"
RESTART_MS="${RESTART_MS:-100}"

if [ $# -gt 0 ]; then ifaces=("$@"); else
  mapfile -t ifaces < <(ip -br link show type can | awk '{print $1}')
fi
[ ${#ifaces[@]} -gt 0 ] || { echo "no CAN interfaces found" >&2; exit 1; }

for i in "${ifaces[@]}"; do
  echo "== $i: before"; ip -details link show "$i" | grep -E "can |restart-ms" | sed 's/^/   /'
  sudo -n ip link set "$i" down
  sudo ip link set "$i" type can bitrate "$BITRATE" restart-ms "$RESTART_MS" berr-reporting on   # not in sudoers NOPASSWD list: prompts
  sudo ip link set "$i" txqueuelen 100
  sudo ip link set "$i" up
  echo "== $i: after";  ip -details link show "$i" | grep -E "can |restart-ms" | sed 's/^/   /'
done
