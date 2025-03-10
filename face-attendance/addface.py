from insightface.app import FaceAnalysis
import cv2
import pickle
import numpy as np
import os
import time

# Ensure directories exist
os.makedirs('data/', exist_ok=True)

# Load existing data with validation
LABELS = []
expected_dim = 512  # Set to 128 if using buffalo_sc

if os.path.exists('data/names.pkl') and os.path.exists('data/faces_data.pkl'):
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    
    # Check dimension compatibility
    if FACES.shape[1] != expected_dim:
        print(f"Warning: Existing data has {FACES.shape[1]}-d embeddings, but model expects {expected_dim}-d.")
        print("Deleting old data to avoid conflicts.")
        LABELS = []
        FACES = np.empty((0, expected_dim))
else:
    FACES = np.empty((0, expected_dim))

# Initialize InsightFace (CHANGE MODEL NAME IF NEEDED)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # 512-d
# app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])  # 128-d
app.prepare(ctx_id=0, det_size=(640, 640))

# Delete existing entry
delete_choice = input("Delete existing entry? (yes/no): ").strip().lower()
if delete_choice == 'yes':
    username = input("Enter username to delete: ").strip()
    if username in LABELS:
        mask = np.array(LABELS) != username
        LABELS = np.array(LABELS)[mask].tolist()
        FACES = FACES[mask]
        print(f"Deleted {username}")
    else:
        print("User not found")

# Add new entry
add_choice = input("Add new face? (yes/no): ").strip().lower()
if add_choice == 'yes':
    username = input("Enter new username: ").strip()
    if username in LABELS:
        print("Username exists. Aborting.")
    else:
        cap = cv2.VideoCapture(0)
        collected = 0
        embeddings = []
        last_capture = 0  # For cooldown
        cooldown = 0.5  # 500ms between captures

        print("Starting automatic capture. Keep your face in view.")
        print("Press 'q' to quit early.")

        while cap.isOpened() and collected < 100:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = app.get(frame)
            display_frame = frame.copy()

            if faces:
                # Draw bounding box and landmarks
                for face in faces:
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    for kp in face.kps.astype(int):
                        cv2.circle(display_frame, tuple(kp), 2, (0, 0, 255), -1)

                # Auto-capture with cooldown
                if len(faces) == 1 and (time.time() - last_capture) > cooldown:
                    embedding = faces[0].embedding
                    embeddings.append(embedding)
                    collected += 1
                    last_capture = time.time()
                    print(f"Captured {collected}/100")

                # Show status
                status = f"Auto-capturing: {collected}/100"
                cv2.putText(display_frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Registration", display_frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if collected > 0:
            # Save even if interrupted early
            LABELS += [username] * collected
            FACES = np.append(FACES, np.array(embeddings), axis=0)

            with open('data/names.pkl', 'wb') as f:
                pickle.dump(LABELS, f)
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(FACES, f)

            print(f"Added {collected}/100 embeddings for {username}")
        else:
            print("No faces captured")

print("Exiting...")