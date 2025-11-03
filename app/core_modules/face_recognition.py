"""
Raspberry Pi Face Recognition Module
Uses OpenCV LBPH face recognizer for real-time face detection and recognition
"""

import cv2
import numpy as np
import pickle
import os
import logging
import time
from pathlib import Path
from datetime import datetime
from collections import Counter


# Setup directory structure
BASE_DIR = Path(__file__).parent / 'data'
TRAINING_DATA_DIR = BASE_DIR / 'training_data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face detection and recognition using LBPH algorithm"""
    
    def __init__(self, confidence_threshold=70):
        """
        Initialize the face recognizer
        
        Args:
            confidence_threshold: Maximum distance for match (lower = stricter)
        """
        self.model_path = MODELS_DIR / 'face_model.yml'
        self.label_path = MODELS_DIR / 'labels.pkl'
        self.confidence_threshold = confidence_threshold
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load model and labels if they exist
        self.labels = {}
        self.is_trained = False
        self._load_model()
        
        logger.info("FaceRecognizer initialized")
    
    def _load_model(self):
        """Load trained model and label mappings"""
        if self.model_path.exists() and self.label_path.exists():
            try:
                self.recognizer.read(str(self.model_path))
                with open(self.label_path, 'rb') as f:
                    self.labels = pickle.load(f)
                self.is_trained = True
                logger.info(f"Model loaded. Known faces: {list(self.labels.values())}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.is_trained = False
        else:
            logger.warning("No trained model found. Please train the model first.")
    
    def train(self, save=True):
        """
        Train the recognizer with face images from data/training_data/
        
        Args:
            save: Whether to save the trained model
        
        Returns:
            True if training successful, False otherwise
        """
        faces = []
        labels = []
        label_ids = {}
        current_id = 0
        
        if not TRAINING_DATA_DIR.exists():
            logger.error(f"Training data path not found: {TRAINING_DATA_DIR}")
            return False
        
        # Iterate through each person's directory
        for person_dir in TRAINING_DATA_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            
            if person_name not in label_ids:
                label_ids[person_name] = current_id
                current_id += 1
            
            label_id = label_ids[person_name]
            
            # Process each image for this person
            for img_path in person_dir.glob('*'):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Detect faces in the image
                detected_faces = self.face_cascade.detectMultiScale(
                    img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                for (x, y, w, h) in detected_faces:
                    face_roi = img[y:y+h, x:x+w]
                    faces.append(face_roi)
                    labels.append(label_id)
        
        if len(faces) == 0:
            logger.error("No faces found in training data")
            return False
        
        logger.info(f"Training with {len(faces)} face samples from {len(label_ids)} people")
        
        # Train the recognizer
        self.recognizer.train(faces, np.array(labels))
        self.labels = {v: k for k, v in label_ids.items()}
        self.is_trained = True
        
        if save:
            self.save_model()
        
        logger.info("Training complete!")
        return True
    
    def save_model(self):
        """Save the trained model and labels"""
        self.recognizer.write(str(self.model_path))
        with open(self.label_path, 'wb') as f:
            pickle.dump(self.labels, f)
        logger.info(f"Model saved to {self.model_path}")
    
    def recognize_face(self, frame):
        """
        Detect and recognize faces in a frame
        
        Args:
            frame: BGR image from camera (numpy array)
        
        Returns:
            List of tuples: (name, confidence, (x, y, w, h))
            - name: Recognized person's name or "Unknown"
            - confidence: Recognition confidence (lower is better, 0-100)
            - (x, y, w, h): Face bounding box coordinates
        """
        results = []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            if self.is_trained:
                # Recognize the face
                label_id, confidence = self.recognizer.predict(face_roi)
                
                # Check if confidence is within threshold
                name = self.labels.get(label_id, "Unknown")
                if confidence >= self.confidence_threshold:
                    name = "Unknown"
                    
                results.append((name, confidence, (x, y, w, h)))
            else:
                # No trained model, just detect
                results.append(("Untrained", 0, (x, y, w, h)))
        
        return results
    
    def draw_results(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: BGR image to draw on
            results: Output from recognize_face()
        
        Returns:
            Frame with annotations
        """
        annotated = frame.copy()
        
        for name, confidence, (x, y, w, h) in results:
            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle around face
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Draw label background
            label = f"{name} ({confidence:.1f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y-30), (x+label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated


def scan_and_identify(recognizer, scan_duration=10, camera_id=0):
    """
    Scan face for a specified duration and determine if it's a match or new person
    
    Args:
        recognizer: FaceRecognizer instance
        scan_duration: Duration in seconds to scan (default 10)
        camera_id: Camera device ID (default 0)
    
    Returns:
        tuple: (is_match, name, avg_confidence) or (False, "New Person", 0)
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error("Camera initialization failed. Check connection or permissions.")
        raise RuntimeError("Camera not accessible")

    
    # Set camera properties for better performance on Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    start_time = time.time()
    recognition_results = []
    frame_count = 0
    
    logger.info(f"Starting {scan_duration} second face scan...")
    print(f"\n{'='*50}")
    print(f"SCANNING FACE - Please look at the camera")
    print(f"{'='*50}")
    
    time.sleep(1)
    while time.time() - start_time < scan_duration:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            continue
        
        frame_count += 1
        elapsed = time.time() - start_time
        remaining = scan_duration - elapsed
        
        # Recognize faces in the frame
        results = recognizer.recognize_face(frame)
        
        # Store results
        for name, confidence, coords in results:
            if name != "Untrained":
                recognition_results.append((name, confidence))
        
        # Draw results on frame
        annotated = recognizer.draw_results(frame, results)
        
        # Add scanning indicator
        progress = int((elapsed / scan_duration) * 100)
        cv2.rectangle(annotated, (10, 10), (630, 60), (0, 0, 0), -1)
        cv2.rectangle(annotated, (20, 20), (20 + int(600 * progress / 100), 50), (0, 255, 0), -1)
        
        scan_text = f"Scanning... {remaining:.1f}s remaining ({progress}%)"
        cv2.putText(annotated, scan_text, (25, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Face Scan', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Scan cancelled by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Analyze results
    if not recognition_results:
        logger.info("No face detected during scan")
        print(f"\n{'='*50}")
        print("RESULT: No face detected")
        print(f"{'='*50}\n")
        return False, "No Face", 0
    
    # Count occurrences of each name
    name_counts = Counter([name for name, _ in recognition_results])
    confidence_by_name = {}
    
    for name, conf in recognition_results:
        if name not in confidence_by_name:
            confidence_by_name[name] = []
        confidence_by_name[name].append(conf)
    
    # Get most common name
    most_common_name, count = name_counts.most_common(1)[0]
    
    if most_common_name == "Unknown":
        avg_confidence = 0
        is_match = False
        result_name = "New Person"
        logger.info(f"New person detected (seen in {count}/{len(recognition_results)} frames)")
    else:
        avg_confidence = np.mean(confidence_by_name[most_common_name])
        is_match = True
        result_name = most_common_name
        logger.info(f"Match found: {result_name} (confidence: {avg_confidence:.2f}, " 
                   f"seen in {count}/{len(recognition_results)} frames)")
    
    # Display result
    print(f"\n{'='*50}")
    print(f"SCAN COMPLETE")
    print(f"{'='*50}")
    print(f"Result: {'MATCH' if is_match else 'NEW PERSON'}")
    if is_match:
        print(f"Name: {result_name}")
        print(f"Confidence: {avg_confidence:.2f}")
        print(f"Detection Rate: {count}/{len(recognition_results)} frames ({100*count/len(recognition_results):.1f}%)")
    else:
        print(f"This appears to be a new person")
        print(f"Detection Rate: {count}/{len(recognition_results)} frames")
    print(f"{'='*50}\n")
    
    return is_match, result_name, avg_confidence


def capture_training_data(person_name, num_samples=30, camera_id=0):
    """
    Capture face samples for training
    
    Args:
        person_name: Name of the person
        num_samples: Number of face samples to capture
        camera_id: Camera device ID (default 0)
    """
    # Create directory for this person
    person_dir = TRAINING_DATA_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Capturing training data for {person_name}")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    count = 0
    print(f"\n{'='*50}")
    print(f"Capturing {num_samples} samples for {person_name}")
    print(f"Press SPACE to capture, ESC to cancel")
    print(f"{'='*50}\n")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display progress
        progress_text = f"Captured: {count}/{num_samples} ({100*count/num_samples:.0f}%)"
        cv2.putText(frame, progress_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if len(faces) > 0:
            cv2.putText(frame, "Press SPACE to capture", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Capture Training Data', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # SPACE key
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = gray[y:y+h, x:x+w]
                img_path = person_dir / f"{count:03d}.jpg"
                cv2.imwrite(str(img_path), face_img)
                count += 1
                logger.info(f"Captured sample {count}/{num_samples}")
                print(f"✓ Captured {count}/{num_samples}")
        elif key == 27:  # ESC key
            logger.info("Training data capture cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*50}")
    print(f"Captured {count} samples for {person_name}")
    print(f"Saved to: {person_dir}")
    print(f"{'='*50}\n")
    
    logger.info(f"Training data capture complete: {count} samples")


# Example usage
if __name__ == "__main__":
    # Initialize recognizer
    recognizer = FaceRecognizer(confidence_threshold=70)
    
    print("\n" + "="*50)
    print("FACE RECOGNITION SYSTEM")
    print("="*50)
    print("1. Capture training data")
    print("2. Train model")
    print("3. Scan and identify face (10 seconds)")
    print("4. Exit")
    print("="*50)
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        # Capture training data
        person_name = input("Enter person's name: ").strip()
        num_samples = input("Number of samples (default 30): ").strip()
        num_samples = int(num_samples) if num_samples else 30
        
        capture_training_data(person_name, num_samples)
        
    elif choice == "2":
        # Train the model
        print("\nTraining model with data from:", TRAINING_DATA_DIR)
        if recognizer.train():
            print("✓ Training successful!")
        else:
            print("✗ Training failed!")
        
    elif choice == "3":
        # Scan and identify
        if not recognizer.is_trained:
            print("\n⚠ Warning: Model not trained yet!")
            print("The system will only detect faces, not recognize them.")
            cont = input("Continue anyway? (y/n): ").strip().lower()
            if cont != 'y':
                print("Please train the model first (option 2)")
                exit()
        
        scan_time = input("\nScan duration in seconds (default 10): ").strip()
        scan_time = int(scan_time) if scan_time else 10
        
        is_match, name, confidence = scan_and_identify(recognizer, scan_duration=scan_time)
        
        # Log the result
        if is_match:
            logger.info(f"Identity confirmed: {name} (confidence: {confidence:.2f})")
        else:
            logger.info(f"New person detected")
    
    elif choice == "4":
        print("\nGoodbye!")
        exit()
    
    else:
        print("\n✗ Invalid option!")