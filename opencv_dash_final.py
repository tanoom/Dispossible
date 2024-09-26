import cv2
import numpy as np
from food_classification_model.cv_food_detection3 import FoodDetector
from face_recoginizing2 import FaceRecognizer
from jwh_pressure_sensor_serial import PressureSensor
import threading
import time

# Initialize detectors
food_detector = FoodDetector("./food_classification_model")
face_recognizer = FaceRecognizer()

# Initialize video captures
cap1 = cv2.VideoCapture(1)  # Food detection stream
cap2 = cv2.VideoCapture(0)  # Face recognition stream

# Initialize pressure sensor
pressure_sensor = PressureSensor()

# Global variable to store pressure data
current_pressure = "No data"

def read_pressure_data():
    global current_pressure
    while True:
        pressure = pressure_sensor.get_real_mass()
        print(pressure)
        if pressure is not None:
            current_pressure = f"{pressure:.2f} g"
        time.sleep(0.1)

# Start pressure reading in a separate thread
pressure_thread = threading.Thread(target=read_pressure_data, daemon=True)
pressure_thread.start()

def add_caption(image, caption):
    h, w = image.shape[:2]
    caption_bg = np.zeros((100, w, 3), dtype=np.uint8)
    cv2.putText(caption_bg, caption, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack((caption_bg, image))

def add_pressure_display(image, pressure):
    h, w = image.shape[:2]
    pressure_bg = np.zeros((100, w, 3), dtype=np.uint8)
    cv2.putText(pressure_bg, f"Total mass: {pressure}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return np.vstack((image, pressure_bg))

while True:
    # Capture frames from both cameras
    ret1, food_frame = cap1.read()
    ret2, face_frame = cap2.read()
    
    if not ret1 or not ret2:
        print("Failed to capture frame from one or both cameras")
        break

    # Process food frame
    processed_food_frame, food_class, food_confidence = food_detector.process_frame(food_frame)

    if food_confidence < 0.3:
        food_caption = "No food detected"
    else:
        food_caption = f"Food: {food_class} ({food_confidence:.2f})"
    processed_food_frame = add_caption(processed_food_frame, food_caption)
    
    # Process face frame
    processed_face_frame = face_recognizer.process_frame(face_frame, rotation=270)
    face_caption = "Face Recognition"
    processed_face_frame = add_caption(processed_face_frame, face_caption)
    
    # Resize frames to have the same height and maintain aspect ratio
    height = min(processed_food_frame.shape[0], processed_face_frame.shape[0])
    processed_food_frame = cv2.resize(processed_food_frame, (int(height * processed_food_frame.shape[1] / processed_food_frame.shape[0]), height))
    processed_face_frame = cv2.resize(processed_face_frame, (int(height * processed_face_frame.shape[1] / processed_face_frame.shape[0]), height))
    
    # Increase the size of each frame proportionally
    scale_factor = 1.5  # Increase size by 50%
    new_height_food = int(processed_food_frame.shape[0] * scale_factor)
    new_width_food = int(processed_food_frame.shape[1] * scale_factor)
    new_height_face = int(processed_face_frame.shape[0] * scale_factor)
    new_width_face = int(processed_face_frame.shape[1] * scale_factor)
    processed_food_frame = cv2.resize(processed_food_frame, (new_width_food, new_height_food))
    processed_face_frame = cv2.resize(processed_face_frame, (new_width_face, new_height_face))
    
    # Combine frames horizontally
    combined_frame = np.hstack((processed_food_frame, processed_face_frame))
    
    # Add pressure display
    combined_frame = add_pressure_display(combined_frame, current_pressure)
    
    # Display the combined frame
    cv2.imshow('Dispossible', combined_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the captures and close windows
cap1.release()

cap2.release()
cv2.destroyAllWindows()
pressure_sensor.close()
