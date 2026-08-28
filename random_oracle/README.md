# XIAO Random Oracle

This sketch turns a Seeed Studio XIAO ESP32-C3 into a tiny LAN-connected dice
and fortune generator.

1. Configure the local Wi-Fi credentials in `secrets.h`.
2. Connect your phone or computer to the same LAN as the XIAO.
3. Open <http://xiao-oracle.local/> or the IP address printed over USB serial.

The board also emits a new d20 roll and fortune over USB serial at 115200 baud
every ten seconds.
