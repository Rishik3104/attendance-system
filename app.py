from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import shutil
from deepface import DeepFace
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Configuration
REGISTERED_USERS_DIR = "registered_users"
ATTENDANCE_RECORDS = "attendance_records"
GROUP_PHOTOS_DIR = "group_photos"
os.makedirs(REGISTERED_USERS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_RECORDS, exist_ok=True)
os.makedirs(GROUP_PHOTOS_DIR, exist_ok=True)

# Initialize attendance CSV
ATTENDANCE_CSV = os.path.join(ATTENDANCE_RECORDS, "attendance.csv")
if not os.path.exists(ATTENDANCE_CSV):
    pd.DataFrame(columns=["Username", "Date"]).to_csv(ATTENDANCE_CSV, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    user_dir = os.path.join(REGISTERED_USERS_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    
    for angle in ['front', 'left', 'right']:
        file = request.files.get(angle)
        if file:
            file.save(os.path.join(user_dir, f"{angle}.jpg"))
    
    return jsonify({"message": "User registered successfully!"})

@app.route('/users', methods=['GET'])
def list_users():
    users = os.listdir(REGISTERED_USERS_DIR)
    return jsonify({"users": users})  # Return JSON for frontend

@app.route('/delete_user', methods=['POST'])
def delete_user():
    if 'username' not in request.form:
        return jsonify({"error": "Username is required!"}), 400
    
    username = request.form['username']
    user_dir = os.path.join(REGISTERED_USERS_DIR, username)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        return jsonify({"message": "User deleted successfully!"})
    return jsonify({"error": "User not found!"})

@app.route('/attendance', methods=['POST'])
def mark_attendance():
    # Save uploaded group photos
    uploaded_files = []
    for i in range(1, 4):
        file = request.files.get(f'group{i}')
        if file:
            file_path = os.path.join(GROUP_PHOTOS_DIR, f"group{i}.jpg")
            file.save(file_path)
            uploaded_files.append(file_path)
    
    # Recognize faces in group photos
    recognized_users = set()
    for image_path in uploaded_files:
        try:
            results = DeepFace.find(
                img_path=image_path,
                db_path=REGISTERED_USERS_DIR,
                model_name="Facenet",
                detector_backend="retinaface",
                enforce_detection=False,
                silent=True,
            )
            for result in results:
                if not result.empty:
                    username = os.path.basename(os.path.dirname(result["identity"][0]))
                    recognized_users.add(username)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Cleanup group photos
    for file in uploaded_files:
        if os.path.exists(file):
            os.remove(file)
    
    # Update attendance records
    date = request.form.get("date", datetime.now().strftime("%Y-%m-%d"))
    df = pd.read_csv(ATTENDANCE_CSV)
    for user in recognized_users:
        df = pd.concat([df, pd.DataFrame([{"Username": user, "Date": date}])], ignore_index=True)
    
    try:
        df.to_csv(ATTENDANCE_CSV, index=False)
    except Exception as e:
        return jsonify({"error": "Failed to update attendance records."}), 500
    
    return jsonify({"message": "Attendance marked!", "recognized_users": list(recognized_users)})

@app.route('/view_attendance', methods=['GET'])
def view_attendance():
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        records = df.to_dict("records")
    else:
        records = []
    return render_template('attendance.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
