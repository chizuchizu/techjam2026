# esp32-linkbench

ESP-NOW 2-node link bandwidth benchmark for Seeed XIAO ESP32-C3 (2 boards,
no router needed). Client pumps 3 payload sizes (64/128/240 B, 300 packets
each) back-to-back; server ACKs each frame; both report over USB CDC 115200.

Build: `pio run -e link-server` and `pio run -e link-client`.
Flash: server -> board A, client -> board B, then read both serial ports.
