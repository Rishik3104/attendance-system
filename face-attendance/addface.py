import os
import time
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# Constants
DATA_DIR = 'data/'
NAMES_FILE = os.path.join(DATA_DIR, 'names.pkl')
FACES_FILE = os.path.join(DATA_DIR, 'faces_data.pkl')
EXPECTED_DIM = 512
MAX_EMBEDDINGS = 100
COOLDOWN = 0.5  # seconds

# Ensure directories
os.makedirs(DATA_DIR, exist_ok=True)

# Load saved data if available
def load_data():
    if os.path.exists(NAMES_FILE) and os.path.exists(FACES_FILE):
        with open(NAMES_FILE, 'rb') as f:
            labels = pickle.load(f)
        with open(FACES_FILE, 'rb') as f:
            faces = pickle.load(f)
        if faces.shape[1] != EXPECTED_DIM:
            print(f"[!] Incompatible embeddings dimension: found {faces.shape[1]}-d, expected {EXPECTED_DIM}-d.")
            print("[!] Deleting old data...")
            return [], np.empty((0, EXPECTED_DIM))
        return labels, faces
    return [], np.empty((0, EXPECTED_DIM))

# Save data to disk
def save_data(labels, faces):
    with open(NAMES_FILE, 'wb') as f:
        pickle.dump(labels, f)
    with open(FACES_FILE, 'wb') as f:
        pickle.dump(faces, f)

# Draw face box and landmarks
def draw_faces(frame, faces):
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        for kp in face.kps.astype(int):
            cv2.circle(frame, tuple(kp), 2, (0, 0, 255), -1)

# Initialize face detection model
def init_model():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # or 'buffalo_sc'
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# Delete a user entry
def delete_entry(labels, faces):
    username = input("Enter username to delete: ").strip()
    if username in labels:
        mask = np.array(labels) != username
        updated_labels = np.array(labels)[mask].tolist()
        updated_faces = faces[mask]
        print(f"[+] Deleted entry for {username}")
        return updated_labels, updated_faces
    else:
        print("[!] User not found.")
    return labels, faces

# Register a new face
def register_face(app, labels, faces):
    username = input("Enter new username: ").strip()
    if username in labels:
        print("[!] Username already exists. Aborting.")
        return labels, faces

    cap = cv2.VideoCapture(0)
    collected, last_capture = 0, 0
    embeddings = []

    print("\n[INFO] Face collection started. Look into the camera.")
    print("Press 'q' to stop early.\n")

    while cap.isOpened() and collected < MAX_EMBEDDINGS:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        faces_detected = app.get(frame)

        if faces_detected:
            draw_faces(display_frame, faces_detected)
            if len(faces_detected) == 1 and (time.time() - last_capture) > COOLDOWN:
                embeddings.append(faces_detected[0].embedding)
                collected += 1
                last_capture = time.time()
                print(f"[+] Captured {collected}/{MAX_EMBEDDINGS}")
            status = f"Capturing: {collected}/{MAX_EMBEDDINGS}"
        else:
            status = "No face detected"

        cv2.putText(display_frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if faces_detected else (0, 0, 255), 2)
        cv2.imshow("Face Registration", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected > 0:
        labels.extend([username] * collected)
        faces = np.append(faces, np.array(embeddings), axis=0)
        save_data(labels, faces)
        print(f"\n[✓] Added {collected} embeddings for '{username}'\n")
    else:
        print("\n[!] No faces captured.")

    return labels, faces

# Main script
def main():
    labels, faces = load_data()
    app = init_model()

    if input("Delete existing entry? (yes/no): ").strip().lower() == 'yes':
        labels, faces = delete_entry(labels, faces)
        save_data(labels, faces)

    if input("Add new face? (yes/no): ").strip().lower() == 'yes':
        labels, faces = register_face(app, labels, faces)

    print("[✓] Done.")

if __name__ == '__main__':
    main()
