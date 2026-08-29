#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <esp_system.h>

#include "secrets.h"

namespace {

constexpr unsigned long SERIAL_ORACLE_INTERVAL_MS = 10000;
constexpr unsigned long WIFI_RETRY_INTERVAL_MS = 10000;

WebServer server(80);
unsigned long lastSerialOracle = 0;
unsigned long lastWiFiRetry = 0;
bool mdnsStarted = false;

const char *const fortunes[] = {
  "A tiny experiment will pay off.",
  "Take the scenic route today.",
  "The bug is probably one line above.",
  "Make something delightfully unnecessary.",
  "Your next idea deserves a prototype.",
  "Ask the question everyone skipped.",
  "A restart may be surprisingly wise.",
  "Ship the small version first.",
  "Trust the measurement, then check it twice.",
  "Today is a good day to learn one strange thing."
};

const char INDEX_HTML[] PROGMEM = R"HTML(
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XIAO Random Oracle</title>
  <style>
    * { box-sizing: border-box; }
    body {
      min-height: 100vh; margin: 0; display: grid; place-items: center;
      background: #111827; color: #f9fafb; font: 18px system-ui, sans-serif;
    }
    main {
      width: min(92vw, 520px); padding: 2rem; border-radius: 24px;
      background: #1f2937; box-shadow: 0 24px 70px #0008; text-align: center;
    }
    h1 { margin-top: 0; font-size: 1.8rem; }
    #die { font-size: 6rem; font-weight: 800; line-height: 1; margin: 1rem; }
    #fortune { min-height: 3rem; color: #d1d5db; }
    button {
      border: 0; border-radius: 999px; padding: .85rem 1.4rem;
      background: #f9fafb; color: #111827; font: inherit; font-weight: 700;
      cursor: pointer;
    }
    small { display: block; margin-top: 1.5rem; color: #9ca3af; }
  </style>
</head>
<body>
  <main>
    <h1>XIAO Random Oracle</h1>
    <div id="die">?</div>
    <p id="fortune">Consult the tiny silicon oracle.</p>
    <button id="ask">Roll the oracle</button>
    <small id="meta"></small>
  </main>
  <script>
    const ask = async () => {
      const r = await fetch('/api/oracle', { cache: 'no-store' });
      const o = await r.json();
      document.body.style.background = o.color;
      document.querySelector('#die').textContent = o.roll;
      document.querySelector('#fortune').textContent = o.fortune;
      document.querySelector('#meta').textContent = `Oracle awake for ${o.uptime}s`;
    };
    document.querySelector('#ask').addEventListener('click', ask);
    ask();
  </script>
</body>
</html>
)HTML";

uint32_t randomBelow(uint32_t limit) {
  return limit == 0 ? 0 : esp_random() % limit;
}

void printOracleToSerial() {
  const uint32_t roll = randomBelow(20) + 1;
  const char *fortune = fortunes[randomBelow(sizeof(fortunes) / sizeof(fortunes[0]))];
  Serial.printf("[oracle] d20=%lu | %s\n", static_cast<unsigned long>(roll), fortune);
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[web] http://%s/ or http://xiao-oracle.local/\n",
                  WiFi.localIP().toString().c_str());
  }
}

void handleOracle() {
  const uint32_t roll = randomBelow(20) + 1;
  const uint32_t color = esp_random() & 0xFFFFFF;
  const char *fortune = fortunes[randomBelow(sizeof(fortunes) / sizeof(fortunes[0]))];

  char colorText[8];
  snprintf(colorText, sizeof(colorText), "#%06lX", static_cast<unsigned long>(color));

  String json;
  json.reserve(200);
  json += F("{\"roll\":");
  json += roll;
  json += F(",\"fortune\":\"");
  json += fortune;
  json += F("\",\"color\":\"");
  json += colorText;
  json += F("\",\"uptime\":");
  json += millis() / 1000;
  json += '}';

  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", json);
}

void announceConnection() {
  Serial.printf("Connected to %s\n", WIFI_SSID);
  Serial.printf("IP address: %s\n", WiFi.localIP().toString().c_str());

  if (!mdnsStarted && MDNS.begin("xiao-oracle")) {
    MDNS.addService("http", "tcp", 80);
    mdnsStarted = true;
    Serial.println("Local name: http://xiao-oracle.local/");
  }
}

void connectToWiFi() {
  Serial.printf("Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  const unsigned long deadline = millis() + 20000;
  while (WiFi.status() != WL_CONNECTED &&
         static_cast<long>(deadline - millis()) > 0) {
    delay(500);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    announceConnection();
  } else {
    Serial.println("Wi-Fi connection timed out; retrying in the background.");
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1200);

  WiFi.mode(WIFI_STA);
  WiFi.setHostname("xiao-oracle");
  connectToWiFi();

  server.on("/", HTTP_GET, []() {
    server.send_P(200, "text/html", INDEX_HTML);
  });
  server.on("/api/oracle", HTTP_GET, handleOracle);
  server.onNotFound([]() {
    server.send(404, "text/plain", "The oracle sees no such path.");
  });
  server.begin();

  Serial.println();
  Serial.println("XIAO Random Oracle is awake.");
  printOracleToSerial();
  lastSerialOracle = millis();
  lastWiFiRetry = millis();
}

void loop() {
  server.handleClient();

  const unsigned long now = millis();
  if (now - lastSerialOracle >= SERIAL_ORACLE_INTERVAL_MS) {
    lastSerialOracle = now;
    printOracleToSerial();
  }

  if (WiFi.status() != WL_CONNECTED &&
      now - lastWiFiRetry >= WIFI_RETRY_INTERVAL_MS) {
    lastWiFiRetry = now;
    Serial.println("Retrying Wi-Fi connection...");
    WiFi.reconnect();
  }

  delay(2);
}
