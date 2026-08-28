# ESP32 cluster transport benchmark

This firmware measures the LAN before treating multiple ESP32 boards as a
compute cluster. It implements a versioned 12-byte binary protocol with UDP
discovery/echo on port 4210 and persistent TCP echo on port 4211.

Copy `secrets.example.h` to the ignored `secrets.h`, set the LAN credentials,
then compile and upload:

```sh
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 esp32_cluster_transport
arduino-cli upload --fqbn esp32:esp32:XIAO_ESP32C3 \
  --port /dev/ttyACM0 esp32_cluster_transport
python3 tools/benchmark_transport.py --output results/esp32c3_transport_v1.csv
```

The host script first tries UDP broadcast discovery. WSL broadcast forwarding
can be blocked even when normal LAN traffic works; in that case read the
`TRANSPORT_READY` address from USB serial and pass it explicitly, for example
`--host 192.168.0.X`. Payloads and request IDs are verified; corrupt or missing
replies are counted rather than discarded.
