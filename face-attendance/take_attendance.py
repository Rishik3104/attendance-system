from flask import Flask, render_template, Response
import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Load MTCNN model
detector = MTCNN()

# Load registered users' embeddings
embedding_dir = 'data/embeddings/'
registered_users = {}
for file in os.listdir(embedding_dir):
    user_data = pd.read_csv(os.path.join(embedding_dir, file))
    user_id = user_data['ID'][0]
    embedding = np.array(eval(user_data['Embedding'][0]))
    registered_users[user_id] = {
        'Name': user_data['Name'][0],
        'Department': user_data['Department'][0],
        'Embedding': embedding
    }

def get_face_embedding(face_pixels):
    """Embed face using DeepFace."""
    embedding = DeepFace.represent(face_pixels, model_name='Facenet')[0]['embedding']
    return np.array(embedding)

def recognize_face(embedding):
    """Recognize face by comparing embeddings."""
    min_distance = 1.0  # Threshold for similarity
    recognized_user = None
    for user_id, user_data in registered_users.items():
        distance = np.linalg.norm(embedding - user_data['Embedding'])
        if distance < min_distance:
            min_distance = distance
            recognized_user = user_data
    return recognized_user if min_distance < 0.7 else None

def generate_frames():
    cap = cv2.VideoCapture(0)
    attendance_marked = set()  # Track marked attendance
    attendance_file = 'data/attendance.csv'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face['box']
            face_pixels = frame[y:y+h, x:x+w]

            # Get embedding and recognize face
            embedding = get_face_embedding(face_pixels)
            user = recognize_face(embedding)

            if user:
                label = f"{user['Name']} ({user['ID']})"
                color = (0, 255, 0)
                # Mark attendance on 'Enter' key press
                if cv2.waitKey(1) & 0xFF == 13:  # Enter key
                    if user['ID'] not in attendance_marked:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(attendance_file, 'a') as f:
                            f.write(f"{timestamp},{user['ID']},{user['Name']},{user['Department']}\n")
                        attendance_marked.add(user['ID'])
                        print(f"Attendance marked for {user['Name']}")
            else:
                label = "Unknown"
                color = (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)