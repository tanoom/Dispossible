import cv2

class FaceRecognizer:
    def __init__(self, img_src='faces'):
        self.img_src = img_src
        self.faces_people = {1: "Tom", 2: "Irene"}
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Initialize face detector and recognizer
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(f'{self.img_src}/model.yml')

    def process_frame(self, img, rotation=270):
        img = cv2.flip(img, 1)
        
        # Rotate the image if needed
        if rotation != 0:
            img = self.rotate_image(img, rotation)
        
        h, w, c = img.shape
        w1 = h * 240 // 320
        x1 = (w - w1) // 2
        img = img[:, x1:x1+w1]
        img = cv2.resize(img, (240, 320))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x - 50, y - 50), (x + w + 50, y + h + 50), (0, 255, 0), 2)
            img_id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
            
            if confidence > 80:
                cv2.putText(img, "Student", (x, y), self.font, 0.6, (0, 255, 0), 2)
                cv2.putText(img, self.faces_people.get(img_id, "Unknown"), (x, y + h), self.font, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y + h), self.font, 0.6, (0, 255, 0), 2)

        return img
    
    @staticmethod
    def rotate_image(image, angle):
        """Rotate the image by the given angle."""
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

# Example usage
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # You can change the rotation angle here (e.g., 0, 90, 180, 270)
        processed_frame = recognizer.process_frame(frame, rotation=270)

        cv2.imshow('Face Recognition', processed_frame)

        print(f"Width: {processed_frame.shape[1]}, Height: {processed_frame.shape[0]}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()