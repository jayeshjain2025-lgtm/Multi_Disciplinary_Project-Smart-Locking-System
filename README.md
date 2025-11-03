# Multi_Disciplinary_Project-Smart-Locking-System
Custom smart door locking system project (multidisciplinary home automation). Organized for modular code, documentation, and automation features.

## Hardware Architecture

### Wiring Diagram

A detailed wiring diagram showing the hardware circuit connections is available in the components directory.

**To generate the wiring diagram PNG:**

```bash
cd components
python3 generate_wiring_diagram.py
```

This will create a `wiring_diagram.png` file showing:
- **Microcontroller**: ESP32/Arduino central controller
- **Solenoid Actuator**: Electric lock mechanism control
- **Sensors**: 
  - Magnetic sensor (door status detection)
  - Motion sensor (PIR detector)
  - Pressure sensor (force detection)
- **Power Supply**: 12V/5V power distribution
- **Communication Modules**:
  - WiFi Module (ESP-NOW protocol)
  - Bluetooth Module (BLE/HC-05)

![Smart Locking System Wiring Diagram](components/wiring_diagram.png)

### Component Details

Refer to `components/HARDWARE_COMPONENTS.md` for detailed component specifications and pinout information.

## Project Structure

```
.
├── Hardware/          # Hardware design files
├── Utils/             # Utility scripts and helpers
├── app/               # Main application
├── components/        # Hardware components documentation and generator
├── scripts/           # Automation scripts
└── README.md          # This file
```

## Getting Started

1. Review the hardware wiring diagram
2. Assemble hardware components according to the diagram
3. Install required libraries from `components/requirements.txt`
4. Configure settings in `config.py`
5. Run `main.py` to start the smart locking system

## License

MIT License - See LICENSE file for details
