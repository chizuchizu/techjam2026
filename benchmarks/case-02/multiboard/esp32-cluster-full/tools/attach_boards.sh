#!/usr/bin/env bash
# Attach the ESP32-C3 boards to WSL over usbip and print their tty nodes.
#
# Bus ids are discovered rather than hard-coded: a flash resets the board's USB
# and Windows takes the device back, and physically replugging moves it to a
# different bus entirely. The tty numbering is not stable either, which is why
# this prints what it found and the coordinator matches ports to node roles.
set -u
USBIPD="/mnt/c/Program Files/usbipd-win/usbipd.exe"

busids() {
    "$USBIPD" list 2>/dev/null \
        | awk '/^Connected:/{c=1;next} /^Persisted:/{c=0} c && /JTAG\/serial debug unit/{print $1}'
}

ids=$(busids)
[ -z "$ids" ] && { echo "no ESP32 boards on the USB bus" >&2; exit 1; }
for b in $ids; do "$USBIPD" attach --wsl --busid "$b" >/dev/null 2>&1; done

want=${TM_BOARDS:-2}
for _ in $(seq 40); do
    n=$(ls /dev/ttyACM* 2>/dev/null | wc -l)
    if [ "$n" -ge "$want" ]; then sleep 1; ls /dev/ttyACM*; exit 0; fi
    sleep 0.5
done
echo "only $(ls /dev/ttyACM* 2>/dev/null | wc -l) of $want boards came back" >&2
ls /dev/ttyACM* 2>/dev/null
exit 1
