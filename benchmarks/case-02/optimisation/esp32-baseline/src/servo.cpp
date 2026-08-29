#include "servo.h"

#ifdef TM_SERVO_PIN

#include <Arduino.h>

namespace {

constexpr int LEDC_CH   = 4;       /* channel 0-3 are commonly grabbed by tone/etc */
constexpr int LEDC_FREQ = 50;      /* standard hobby-servo frame: 20 ms            */
/* The C3's LEDC tops out at 14-bit duty resolution (the original ESP32 does
 * 20). Asking for 16 makes ledcSetup() fail outright and no pulse is ever
 * generated. At 50 Hz, 14 bits is ~1.2 us per count - about 0.1 deg. */
constexpr int LEDC_RES  = 14;
constexpr int PULSE_MIN_US = 500;  /* ~0 degrees   */
constexpr int PULSE_MAX_US = 2500; /* ~180 degrees */
constexpr int FRAME_US     = 20000;
constexpr int STEP_DEG     = 4;    /* 4 deg / 20 ms -> a full sweep in ~1.8 s,
                                    * which lands close to one 1.99 s forward */

volatile int g_busy = 0;

void write_us(int us) {
    if (us < PULSE_MIN_US) us = PULSE_MIN_US;
    if (us > PULSE_MAX_US) us = PULSE_MAX_US;
    const uint32_t full = (1u << LEDC_RES) - 1u;
    ledcWrite(LEDC_CH, (uint32_t)((uint64_t)us * full / FRAME_US));
}

void write_deg(int deg) {
    write_us(PULSE_MIN_US + (PULSE_MAX_US - PULSE_MIN_US) * deg / 180);
}

void servo_task(void*) {
    int deg = 0, dir = 1, was_busy = -1;
    for (;;) {
        const int busy = g_busy;
        if (busy) {
            deg += dir * STEP_DEG;
            if (deg >= 180) { deg = 180; dir = -1; }
            if (deg <= 0)   { deg = 0;   dir =  1; }
            write_deg(deg);
        } else if (was_busy != 0) {
            write_deg(90);              /* park centred between forwards */
            deg = 0; dir = 1;
        }
        was_busy = busy;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

}  // namespace

int tm_servo_begin(int pin) {
    const uint32_t hz = ledcSetup(LEDC_CH, LEDC_FREQ, LEDC_RES);
    if (hz == 0) return 0;          /* caller reports it; never fail silently */
    ledcAttachPin(pin, LEDC_CH);
    write_deg(90);
    /* priority 2 sits above the Arduino loop task (1) so it preempts the
     * forward, and well below the WiFi/system tasks */
    xTaskCreate(servo_task, "tmservo", 2048, nullptr, 2, nullptr);
    return (int)hz;
}

void tm_servo_busy(int on) { g_busy = on ? 1 : 0; }

void tm_servo_set_deg(int deg) {
    if (deg < 0) deg = 0;
    if (deg > 180) deg = 180;
    g_busy = 0;
    write_deg(deg);
}

#else
int  tm_servo_begin(int pin)   { (void)pin; return 0; }
void tm_servo_busy(int on)     { (void)on; }
void tm_servo_set_deg(int deg) { (void)deg; }
#endif
