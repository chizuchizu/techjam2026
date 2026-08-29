#!/usr/bin/env bash
# Re-attach the two ESP32-C3 boards to WSL over usbip and print their tty nodes.
#
# A flash (or a stalled handle) resets the board's USB and Windows takes the
# device back, so the /dev/ttyACM* nodes vanish - and they do not necessarily
# come back with the same numbers, which is why this prints what it found.
set -u
USBIPD="/mnt/c/Program Files/usbipd-win/usbipd.exe"
BUSIDS="${*:-2-1 3-1}"
for b in $BUSIDS; do "$USBIPD" attach --wsl --busid "$b" >/dev/null 2>&1; done
for _ in $(seq 40); do
    n=$(ls /dev/ttyACM* 2>/dev/null | wc -l)
    if [ "$n" -ge 2 ]; then sleep 1; ls /dev/ttyACM*; exit 0; fi
    sleep 0.5
done
echo "boards did not come back" >&2
exit 1
