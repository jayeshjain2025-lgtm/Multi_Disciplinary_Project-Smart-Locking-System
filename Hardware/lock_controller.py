"""
Hardware/lock_controller.py

Camera-based Lock Controller:
- Uses cv2 to capture frames and ask FaceRecognizer to identify faces.
- If a face is recognized (name != "Unknown" and confidence below threshold) -> unlock door.
- Supports RELAY or SERVO control via gpiozero on a Raspberry Pi.
- Simulation mode available (default) so you can run on your laptop without hardware.

Usage:
    python -m Hardware.lock_controller
"""

import sys
import os
import time
import logging
from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta

import cv2
import numpy as np

# ---------------------------
# Project root & imports
# ---------------------------
# Ensure project root is on sys.path so "app.core_modules" is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project_root/Hardware -> parents[1] = project_root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try importing your face recognition class
try:
    from app.core_modules.face_recognition import FaceRecognizer
    FACE_RECOGNITION_AVAILABLE = True
except Exception as e:
    FaceRecognizer = None
    FACE_RECOGNITION_AVAILABLE = False
    # We'll still run simulation, but recognition will be disabled.
    # Log below.

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("lock_controller")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# Configuration
# ---------------------------
SIMULATION_MODE = os.getenv("SIMULATE_PI", "1") == "1"  # set SIMULATE_PI=0 to attempt real GPIO
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "50.0"))  # LBPH lower is better
UNLOCK_DURATION = int(os.getenv("UNLOCK_DURATION", "10"))  # seconds to wait before auto-lock
UNLOCK_COOLDOWN = int(os.getenv("UNLOCK_COOLDOWN", "5"))  # seconds between unlocks to avoid loops

# Default GPIO pins (BCM)
DEFAULT_RELAY_PIN = int(os.getenv("RELAY_PIN", "17"))
DEFAULT_SERVO_PIN = int(os.getenv("SERVO_PIN", "18"))
DEFAULT_STATUS_LED_PIN = int(os.getenv("STATUS_LED_PIN", "27"))

# ---------------------------
# GPIO / Simulation setup
# ---------------------------
if not SIMULATION_MODE:
    try:
        from gpiozero import OutputDevice, Servo, LED
        logger.info("gpiozero imported - attempting hardware mode")
    except Exception as e:
        logger.warning("gpiozero import failed or no pin factory available: %s", e)
        logger.warning("Falling back to SIMULATION MODE")
        SIMULATION_MODE = True

if SIMULATION_MODE:
    logger.info("[SIMULATION MODE] No GPIO hardware will be used.")

    class LED:
        def __init__(self, pin):
            self.pin = pin

        def on(self):
            print(f"[SIM] LED {self.pin}: ON")

        def off(self):
            print(f"[SIM] LED {self.pin}: OFF")

        def blink(self, on_time=0.2, off_time=0.2, n=3):
            for _ in range(n):
                self.on(); time.sleep(on_time)
                self.off(); time.sleep(off_time)

    class OutputDevice:
        def __init__(self, pin, active_high=True, initial_value=False):
            self.pin = pin
            self.state = initial_value
            print(f"[SIM] OutputDevice (pin {pin}) initialized")

        def on(self):
            self.state = True
            print(f"[SIM] Relay pin {self.pin}: ON")

        def off(self):
            self.state = False
            print(f"[SIM] Relay pin {self.pin}: OFF")

    class Servo:
        def __init__(self, pin):
            self.pin = pin

        # simulated servo positions
        def unlock(self):
            print(f"[SIM] Servo {self.pin}: UNLOCK position")

        def lock(self):
            print(f"[SIM] Servo {self.pin}: LOCK position")

else:
    # Real servo API differs; we'll use gpiozero.Servo with .value in real mode
    Servo = Servo  # already imported from gpiozero above
    OutputDevice = OutputDevice
    LED = LED

# ---------------------------
# Lock types
# ---------------------------
class LockType(Enum):
    RELAY = "RELAY"
    SERVO = "SERVO"

# ---------------------------
# LockController class
# ---------------------------
class LockController:
    """
    Controls the physical (or simulated) lock mechanism.
    Methods:
      - unlock(person_name=None)
      - lock()
      - get_status()
    """

    def __init__(self,
                 lock_type=LockType.RELAY,
                 relay_pin=DEFAULT_RELAY_PIN,
                 servo_pin=DEFAULT_SERVO_PIN,
                 status_led_pin=DEFAULT_STATUS_LED_PIN,
                 unlock_duration=UNLOCK_DURATION):
        self.lock_type = lock_type
        self.relay_pin = relay_pin
        self.servo_pin = servo_pin
        self.status_led_pin = status_led_pin
        self.unlock_duration = unlock_duration

        self.is_locked = True
        self.last_unlocked_at = None

        # instantiate hardware or simulated devices
        try:
            self.status_led = LED(self.status_led_pin) if self.status_led_pin is not None else None
        except Exception as e:
            logger.warning("Status LED init failed: %s", e)
            self.status_led = None

        if self.lock_type == LockType.RELAY:
            try:
                self.relay = OutputDevice(self.relay_pin, active_high=True, initial_value=False)
            except Exception as e:
                logger.warning("Relay init failed: %s -- switching to simulation relay", e)
                self.relay = OutputDevice(self.relay_pin)
        else:  # servo
            try:
                if SIMULATION_MODE:
                    self.servo = Servo(self.servo_pin)
                else:
                    self.servo = Servo(self.servo_pin)  # gpiozero Servo
            except Exception as e:
                logger.warning("Servo init failed: %s", e)
                self.servo = None

        logger.info("LockController initialized (type=%s)", self.lock_type.value)

    def unlock(self, person_name=None, force=False):
        """
        Unlock the door.
        - person_name: optional, logged for audit
        - force: if True, ignore cooldown
        Returns True if unlock action performed.
        """
        now = datetime.now()
        if not force and self.last_unlocked_at:
            if (now - self.last_unlocked_at).total_seconds() < UNLOCK_COOLDOWN:
                logger.info("Unlock suppressed due to cooldown (last unlocked %s seconds ago).",
                            (now - self.last_unlocked_at).total_seconds())
                return False

        logger.info("Unlocking door for: %s", person_name or "UNKNOWN")
        if self.status_led:
            try:
                self.status_led.on()
            except Exception:
                pass

        if self.lock_type == LockType.RELAY:
            try:
                self.relay.on()
            except Exception:
                # in simulation or if relay missing, print
                print("[SIM] relay.on() (fallback)")

        else:  # SERVO
            try:
                if SIMULATION_MODE:
                    # our simulated Servo class uses unlock() helper
                    if hasattr(self.servo, "unlock"):
                        self.servo.unlock()
                    else:
                        print(f"[SIM] servo {self.servo_pin} UNLOCK")
                else:
                    # gpiozero.Servo uses value -1..1; set to unlock value (1)
                    self.servo.value = 1
            except Exception:
                logger.exception("Error moving servo to unlock position")

        self.is_locked = False
        self.last_unlocked_at = datetime.now()

        # auto-lock after unlock_duration in a non-blocking way (simple thread)
        try:
            # spawn a simple timer thread so we don't block camera loop
            import threading
            t = threading.Timer(self.unlock_duration, self.lock)
            t.daemon = True
            t.start()
        except Exception:
            # fallback: blocking sleep (shouldn't happen)
            time.sleep(self.unlock_duration)
            self.lock()

        logger.info("Door UNLOCKED")
        return True

    def lock(self):
        """Locks the door immediately."""
        logger.info("Locking door")
        if self.status_led:
            try:
                self.status_led.off()
            except Exception:
                pass

        if self.lock_type == LockType.RELAY:
            try:
                self.relay.off()
            except Exception:
                print("[SIM] relay.off() (fallback)")
        else:
            try:
                if SIMULATION_MODE:
                    if hasattr(self.servo, "lock"):
                        self.servo.lock()
                    else:
                        print(f"[SIM] servo {self.servo_pin} LOCK")
                else:
                    self.servo.value = -1
            except Exception:
                logger.exception("Error moving servo to lock position")

        self.is_locked = True
        logger.info("Door LOCKED")
        return True

    def get_status(self):
        return {
            "is_locked": self.is_locked,
            "lock_type": self.lock_type.value,
            "last_unlocked_at": self.last_unlocked_at.isoformat() if self.last_unlocked_at else None
        }

# ---------------------------
# Camera-based recognition loop
# ---------------------------
def camera_auth_loop(lock_controller: LockController,
                     camera_id=0,
                     scan_duration=10,
                     confidence_threshold=CONFIDENCE_THRESHOLD,
                     show_window=True):
    """
    Capture frames from camera for 'scan_duration' seconds and try to identify faces.
    If a recognized face (name != 'Unknown') with confidence < threshold is found -> unlock.
    Returns tuple (access_granted: bool, name or None, avg_confidence)
    """
    if not FACE_RECOGNITION_AVAILABLE:
        logger.error("Face recognition module not available. Aborting camera auth.")
        return False, None, None

    recognizer = FaceRecognizer(confidence_threshold=confidence_threshold)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("Camera could not be opened (id=%s).", camera_id)
        return False, None, None

    start_time = time.time()
    recognition_results = []

    try:
        while time.time() - start_time < scan_duration:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed from camera")
                time.sleep(0.2)
                continue

            # optional resize for speed (comment/uncomment as needed)
            # frame = cv2.resize(frame, (640, 480))

            # Ask recognizer to detect/recognize faces in the frame
            results = recognizer.recognize_face(frame)  # returns list of (name, confidence, bbox)
            # collect results ignoring "Untrained"
            for name, conf, bbox in results:
                if name != "Untrained":
                    recognition_results.append((name, conf))

            # Draw boxes for visualization if desired
            if show_window:
                annotated = recognizer.draw_results(frame, results)
                cv2.imshow("Lock Camera - Press 'q' to cancel", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User cancelled camera auth via 'q'")
                    break

        # analyze collected results
        if not recognition_results:
            logger.info("No recognized face during scan")
            return False, None, None

        # pick most common recognized name
        names = [n for n, _ in recognition_results]
        confidences_by_name = {}
        for n, c in recognition_results:
            confidences_by_name.setdefault(n, []).append(c)

        from collections import Counter
        most_common_name, count = Counter(names).most_common(1)[0]
        avg_conf = float(np.mean(confidences_by_name[most_common_name]))

        logger.info("Most common: %s (seen %d frames) avg_confidence=%.2f", most_common_name, count, avg_conf)

        # Determine access
        if most_common_name == "Unknown":
            logger.info("Result is Unknown -> deny")
            return False, "Unknown", avg_conf

        if avg_conf > confidence_threshold:
            logger.info("Confidence too low (%.2f > %.2f) -> deny", avg_conf, confidence_threshold)
            return False, most_common_name, avg_conf

        # else grant access
        logger.info("Access granted to %s (avg_conf=%.2f)", most_common_name, avg_conf)
        lock_controller.unlock(person_name=most_common_name)
        return True, most_common_name, avg_conf

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

# ---------------------------
# Main entrypoint
# ---------------------------
if __name__ == "__main__":
    logger.info("Starting camera-based lock controller (simulation=%s, face_recognition=%s)",
                SIMULATION_MODE, FACE_RECOGNITION_AVAILABLE)

    # create controller in simulation mode by default
    controller = LockController(lock_type=LockType.RELAY,
                                relay_pin=DEFAULT_RELAY_PIN,
                                servo_pin=DEFAULT_SERVO_PIN,
                                status_led_pin=DEFAULT_STATUS_LED_PIN,
                                unlock_duration=UNLOCK_DURATION)

    try:
        # If face recognition module is missing, run a simulated demo loop
        if not FACE_RECOGNITION_AVAILABLE:
            print("\n[DEMO] FaceRecognizer not available. Running simulated demo.")
            for _ in range(3):
                print("[DEMO] Simulating recognized John Doe -> unlocking")
                controller.unlock(person_name="John Doe", force=True)
                time.sleep(2)
            controller.lock()
            print("[DEMO] Done.")
            sys.exit(0)

        # Real camera scan: press Ctrl+C or 'q' in window to exit early
        access, name, avg_conf = camera_auth_loop(controller, camera_id=0, scan_duration=12, show_window=True)
        if access:
            print(f"ACCESS GRANTED: {name} (avg_conf={avg_conf:.2f})")
        else:
            print("ACCESS DENIED")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        controller.lock()
        logger.info("Shutdown complete")
