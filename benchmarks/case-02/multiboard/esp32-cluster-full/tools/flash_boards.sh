#!/usr/bin/env bash
# Build and flash both boards, then re-attach them (flashing resets USB, which
# hands the device back to Windows and renumbers the tty node).
set -eu
cd "$(dirname "$0")/.."
PIO=${PIO:-$HOME/.platformio/penv/bin/pio}
"$PIO" run
for port in $(ls /dev/ttyACM* 2>/dev/null); do
    echo "=== flashing $port ==="
    "$PIO" run -t upload --upload-port "$port" | tail -1
    sleep 2
    ./tools/attach_boards.sh >/dev/null || true
done
./tools/attach_boards.sh
