from flask import Flask, render_template, Response
from insightface.app import FaceAnalysis
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from threading import Thread

app = Flask(__name__)

# Initialize directories and CSV columns
if not os.path.exists('Attendance/'):
    os.makedirs('Attendance/')
COL_NAMES = ['NAME', 'TIME']

# Load trained data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Initialize InsightFace (detection + recognition)
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Open the video feed
video = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        success, frame = video.read()
        if not success:
            break

        # Detect faces with InsightFace
        faces = face_app.get(frame)
        
        # Process each detected face
        for face in faces:
            # Get bounding box coordinates
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Recognize face
            name = recognize_face(face.embedding)
            
            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def recognize_face(embedding, threshold=0.6):
    # Calculate cosine similarity
    similarities = np.dot(FACES, embedding) / (np.linalg.norm(FACES, axis=1) * np.linalg.norm(embedding))
    best_match_index = np.argmax(similarities)
    
    if similarities[best_match_index] > threshold:
        return LABELS[best_match_index]
    return "Unknown"

def save_attendance(name):
    if name == "Unknown":
        return  # Don't save unknown entries
        
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    attendance = [name, timestamp]
    file_path = f"Attendance/Attendance_{date}.csv"
    
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(file_path).st_size == 0:
            writer.writerow(COL_NAMES)
        writer.writerow(attendance)

if __name__ == "__main__":
    # Start Flask in separate thread
    Thread(target=lambda: app.run(debug=True, use_reloader=False)).start()

    # Main loop for attendance marking
    while True:
        success, frame = video.read()
        if not success:
            break

        # Detect faces
        faces = face_app.get(frame)
        
        # Process each face
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            name = recognize_face(face.embedding)
            
            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Show window and handle keypresses
        cv2.imshow("Face Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and name != "Unknown":
            save_attendance(name)
            print(f"Attendance marked for {name}")
        elif key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()