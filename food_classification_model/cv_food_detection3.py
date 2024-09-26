import cv2
import numpy as np
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import json

class FoodDetector:
    def __init__(self, model_path):
        # Check if CUDA available, use it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")

        # Load the model and processor
        self.model = ViTForImageClassification.from_pretrained(model_path).to(self.device)
        
        self.processor = ViTImageProcessor.from_pretrained(model_path)

        # Disable SDPA (which includes Flash Attention)
        self.model.config.use_sdpa = True

        if hasattr(self.model.config, 'use_attention_fusion'):
            self.model.config.use_attention_fusion = True

        # Load the id2label mapping
        with open(f"{model_path}/config.json") as f:
            config = json.load(f)
            self.id2label = config["id2label"]

        if torch.backends.cuda.flash_sdp_enabled():
            torch.backends.cuda.enable_flash_sdp(True)

    def process_frame(self, frame, rotation=180, zoom_factor=0.5):
        # Rotate the image if needed
        if rotation != 0:
            frame = self.rotate_image(frame, rotation)

        # Get the dimensions of the frame
        height, width = frame.shape[:2]

        # Calculate the size of the zoomed-in area
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

        # Calculate the coordinates to crop the center of the image
        top = (height - new_height) 
        left = (width - new_width) // 2
        bottom = top + new_height
        right = left + new_width

        # Crop the center of the image
        frame = frame[top:bottom, left:right]

        # Resize the cropped image back to the original dimensions
        frame = cv2.resize(frame, (width, height))

        # Preprocess the frame for the model
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Make a prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = self.id2label[str(predicted_class_idx)]
        confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()

        # If confidence is above 30%, draw rectangle and label
        if confidence > 0.3:
            # Draw rectangle
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

            # Prepare label
            label = f"{predicted_class}: {confidence:.2f}"
            
            # Put label on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize the processed frame to 640x480
        frame = cv2.resize(frame, (640, 480))

        return frame, predicted_class, confidence

    @staticmethod
    def rotate_image(image, angle):
        """Rotate the image by the given angle."""
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

# Example usage (can be commented out when using in Dash app)
if __name__ == "__main__":
    detector = FoodDetector("./food_classification_model")
    cap = cv2.VideoCapture(1)
    print(f"cv2 API {cap.getBackendName()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, predicted_class, confidence = detector.process_frame(frame)


        cv2.imshow('Food Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()