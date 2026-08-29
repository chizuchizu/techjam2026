#!/usr/bin/env bash
# Build and flash every attached board, then re-attach them (flashing resets
# the board's USB, which hands the device back to Windows and renumbers the tty
# node). Set TM_BOARDS to the number you expect, e.g. TM_BOARDS=8.
set -eu
cd "$(dirname "$0")/.."
PIO=${PIO:-$HOME/.platformio/penv/bin/pio}
WANT=${TM_BOARDS:-2}
export TM_BOARDS="$WANT"

"$PIO" run
ports=$(ls /dev/ttyACM* 2>/dev/null || true)
n=$(printf '%s\n' $ports | grep -c . || true)
echo "=== flashing $n board(s), expecting $WANT ==="
for port in $ports; do
    echo "--- $port ---"
    "$PIO" run -t upload --upload-port "$port" | tail -1
    sleep 2
    ./tools/attach_boards.sh >/dev/null 2>&1 || true
done
./tools/attach_boards.sh
