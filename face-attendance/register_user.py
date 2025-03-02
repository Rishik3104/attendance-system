import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
import os
import pandas as pd

# Load MTCNN model
detector = MTCNN()

# Directory to store embeddings
embedding_dir = 'data/embeddings/'
os.makedirs(embedding_dir, exist_ok=True)

def get_face_embedding(face_pixels):
    """Embed face using DeepFace."""
    embedding = DeepFace.represent(face_pixels, model_name='Facenet')[0]["embedding"]
    return embedding

def register_user():
    name = input("Enter user's name: ")
    user_id = input("Enter user's ID: ")
    dept = input("Enter user's department: ")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    print("Capturing face... Look at the camera.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]['box']
            face_pixels = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Show preview
            cv2.imshow("Register User", frame)

            # Press 's' to save the face
            key = cv2.waitKey(1)
            if key == ord('s'):
                embedding = get_face_embedding(face_pixels)

                # Save embedding and details
                user_data = {
                    'Name': name,
                    'ID': user_id,
                    'Department': dept,
                    'Embedding': embedding
                }
                df = pd.DataFrame([user_data])
                df.to_csv(os.path.join(embedding_dir, f'{user_id}.csv'), index=False)
                print(f"User {name} registered successfully!")
                break

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_user()