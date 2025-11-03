# Hardware Components and Circuit Guide

This document lists the hardware required for the Smart Locking System and provides clear wiring instructions to integrate the hardware with this repository's software (Hardware/, app/, main.py, config.py).

## 1) Bill of Materials (BOM)

Core controller and power
- 1x Microcontroller with Wi‑Fi (ESP32 DevKit V1 recommended; alternative: ESP8266 NodeMCU if resources allow)
- 1x 5V DC power supply (2A recommended) or USB 5V supply
- 1x Buck converter (optional if powering motors/solenoids from higher voltage)
- 1x Logic level shifter (optional; only if mixing 3.3V controller with 5V peripherals that require 5V logic)

Actuation (choose one)
- Option A: 1x 12V solenoid door lock + 1x N‑channel MOSFET (e.g., IRLZ44N or AO3400) + 1x Flyback diode (1N5819/1N4007)
- Option B: 1x SG90/SG92R micro servo (5V), or MG995/MG996R (higher torque, 5–6V)

Sensors and user input
- 1x Magnetic reed switch (door open/close) + 10k pull‑up resistor (if needed)
- 1x PIR motion sensor (HC‑SR501) (optional)
- 1x Keypad (4x4 matrix) or 1x RFID module (RC522 + key tags) or 1x Capacitive touch sensor (TTP223) depending on design
- 1x Buzzer (active 5V or passive with series resistor) (optional)
- 1x Status LEDs (Red/Green) + 220Ω resistors

RFID option parts
- 1x RC522 RFID reader (SPI)
- Female‑female jumpers for SPI

Keypad option parts
- 1x 4x4 membrane keypad
- 8x jumper wires

Miscellaneous
- Breadboard or perfboard
- Dupont jumper wires (male‑male, male‑female)
- Screws, standoffs, enclosure
- Door lock mounting bracket/fixture

## 2) Pin Mapping (suggested)
The actual pins can be configured in config.py. Suggested defaults for ESP32:

- Lock actuator
  - Solenoid + MOSFET: Gate -> GPIO 25, Source -> GND, Drain -> solenoid -, solenoid + -> +12V. Flyback diode across solenoid (A to +12V, K to solenoid -)
  - Servo: Signal -> GPIO 25, Vcc -> 5V, GND -> GND. Use separate 5V rail with common ground.

- Door sensor (reed): GPIO 26 with internal pull‑up. One lead to GND, other to GPIO 26.
- Status LED Green: GPIO 18 via 220Ω to LED anode, LED cathode to GND.
- Status LED Red: GPIO 19 via 220Ω to LED anode, LED cathode to GND.
- Buzzer (optional): GPIO 27 -> 220Ω -> Buzzer +, Buzzer - to GND.

- RFID RC522 (SPI):
  - SDA/SS -> GPIO 5
  - SCK -> GPIO 18 (or 14 depending board; adjust in config.py)
  - MOSI -> GPIO 23
  - MISO -> GPIO 19
  - RST -> GPIO 22
  - VCC -> 3.3V, GND -> GND (RC522 is 3.3V logic)

- Keypad (4x4) alternative (if used instead of RFID):
  - Rows -> GPIOs [32, 33, 34, 35] (note: GPIO34/35 are input‑only)
  - Cols -> GPIOs [12, 13, 14, 15]
  Adjust in config.py accordingly.

- PIR (optional): OUT -> GPIO 4, VCC -> 5V, GND -> GND.

## 3) Power and Grounding
- Always share ground between the microcontroller and all peripherals (GND common).
- If using a solenoid or high‑torque servo, use a dedicated supply for actuators (e.g., 5–6V for servo or 12V for solenoid). Do NOT power a servo/solenoid directly from the ESP32 5V pin if current > 500mA.
- Place a flyback diode across inductive loads (solenoid, relay coils).
- For servos, add a 470µF–1000µF electrolytic capacitor across the 5V and GND near the servo connector to suppress brownouts.

## 4) Circuit Wiring Guide

A) Solenoid + MOSFET path (Option A)
- Connect solenoid + to +12V.
- Connect solenoid - to MOSFET Drain.
- Connect MOSFET Source to GND.
- Connect MOSFET Gate to ESP32 GPIO 25 through a 100Ω–220Ω series resistor (optional for ringing). Add a 10k pulldown from Gate to GND.
- Place flyback diode across solenoid: diode cathode to +12V, anode to solenoid -.
- Ensure ESP32 GND and 12V supply GND are common.

B) Servo path (Option B)
- Servo Vcc to 5V rail (capable of 1–2A), Servo GND to common GND, Servo signal to GPIO 25.
- Add bulk capacitor on 5V rail near servo.

C) Door sensor (reed switch)
- One lead to GND, the other to GPIO 26. Enable internal pull‑up in software or add external 10k pull‑up to 3.3V.

D) Status LEDs
- Green LED: GPIO 18 -> 220Ω -> LED anode; cathode to GND.
- Red LED: GPIO 19 -> 220Ω -> LED anode; cathode to GND.

E) RFID RC522 (SPI)
- Wire as per the SPI mapping above to 3.3V logic. Keep wires short. Do not power RC522 with 5V on logic.

F) Keypad (alternative to RFID)
- Connect 8 lines (4 rows + 4 columns) to the assigned GPIOs. Configure in config.py.

G) Buzzer
- GPIO 27 -> resistor -> Buzzer +, Buzzer - to GND. For passive buzzer, you can PWM the pin for tones.

H) PIR (optional)
- OUT to GPIO 4. Provide 5V and GND. Some PIRs work at 3.3–5V; check your module.

## 5) Repository Software Integration
- Hardware/lock_controller.py: Drives the actuator (servo or MOSFET). Ensure PIN_ACTUATOR in config.py matches the chosen GPIO (default: 25). For servo, the controller should generate PWM; for solenoid, it toggles the MOSFET gate.
- Hardware/sensor.py: Reads reed switch state and any optional sensors (PIR). Match the GPIOs (e.g., REED_PIN=26, PIR_PIN=4) in config.py.
- config.py: Central place to set pin numbers, select input method (RFID or keypad), and enable optional peripherals. Update:
  - USE_SERVO or USE_SOLENOID
  - RFID_ENABLED / KEYPAD_ENABLED
  - PIN mappings for ACTUATOR, REED, LED_RED, LED_GREEN, BUZZER, SPI pins, etc.
- main.py: Initializes controllers and runs the main loop/app. After wiring, flash and run to test lock/unlock and sensor events.

## 6) Bring‑Up and Testing Checklist
1) Power up only the ESP32 and LEDs first; verify no shorts and LEDs toggle via a simple blink script.
2) Add reed switch; verify open/close detection in logs (Hardware/data/logs).
3) Add actuator supply and wiring. For solenoid, test brief pulses; for servo, verify smooth motion to lock/unlock positions.
4) Add RFID or keypad and verify credential read.
5) Integrate all pieces and run end‑to‑end: door closed -> authenticate -> actuator toggles -> status LEDs indicate state.

## 7) Safety and Installation Notes
- Isolate high‑current wiring from signal lines.
- Use heat‑shrink and proper connectors; avoid loose wires in doors.
- Mount the actuator solidly; ensure mechanical end stops don’t stall servos.
- If mounting on metal doors, test RFID read distance; consider external antenna if needed.

## 8) Schematic Reference (textual)
- MCU GPIO25 -> Actuator signal (MOSFET gate or Servo signal)
- MCU GPIO26 <- Reed sensor (with pull‑up)
- MCU GPIO18 -> Green LED (via 220Ω)
- MCU GPIO19 -> Red LED (via 220Ω)
- MCU GPIO27 -> Buzzer (via 220Ω)
- SPI: SS=5, SCK=18, MOSI=23, MISO=19, RST=22 -> RC522 (3.3V)
- Common GND across all modules; dedicated 5V/12V rails for actuators

Adjust pin numbers to match your board and update config.py accordingly.
