from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
import signal
import sys
import traceback
import glob
import re
import csv

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
# Configuration
VENV_PATH = "../venv/bin/activate"
PYTHON_SCRIPTS = {
    "addface": "../addface.py",
    "attendance": "../test.py"
}
ATTENDANCE_FOLDER = "../Attendance"  # Adjust this path as needed

@app.route('/api/execute', methods=['POST', 'OPTIONS'])
def execute_command():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
    # Log received request
        print(f"Received request with data: {request.data}")
        
        data = request.json
        if not data:
            print("Error: No JSON data received")
            return jsonify({
                'success': False,
                'error': "No data received"
            }), 400
            
        command_type = data.get('command')
        inputs = data.get('inputs', [])  # Get inputs if provided, otherwise empty list
        print(f"Command type: {command_type}")
        
        if command_type not in PYTHON_SCRIPTS:
            return jsonify({
                'success': False,
                'error': f"Invalid command: {command_type}"
            }), 400
            
        script_name = PYTHON_SCRIPTS[command_type]
        
        # Check if script exists
        if not os.path.exists(script_name):
            print(f"Error: Script {script_name} does not exist")
            return jsonify({
                'success': False,
                'error': f"Script {script_name} not found"
            }), 404
        
        # Check if venv exists
        if not os.path.exists(VENV_PATH):
            print(f"Error: Virtual environment at {VENV_PATH} does not exist")
            return jsonify({
                'success': False,
                'error': f"Virtual environment not found at {VENV_PATH}"
            }), 404
            
        # Construct the command to activate the virtual environment and run the Python script
        activate_cmd = f"source {VENV_PATH}"
        python_cmd = f"python3 {script_name}"
        full_cmd = f"{activate_cmd} && {python_cmd}"
        
        print(f"Executing command: {full_cmd}")
        
        # Prepare input string if inputs are provided
        input_string = None
        if inputs:
            input_string = "\n".join(inputs) + "\n"
            input_bytes = input_string.encode('utf-8')
            print(f"Will provide inputs: {input_string}")
        
        # Execute the command in a bash shell
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if inputs else None,
            shell=True,
            executable='/bin/bash'
        )
        
        stdout, stderr = process.communicate(input=input_bytes if inputs else None, timeout=60)  # 60-second timeout
        
        stdout_str = stdout.decode('utf-8')
        stderr_str = stderr.decode('utf-8')
        
        print(f"Command output: {stdout_str}")
        if stderr_str:
            print(f"Command error: {stderr_str}")
        
        if process.returncode != 0:
            return jsonify({
                'success': False,
                'error': stderr_str,
                'command': full_cmd
            }), 500
        
        return jsonify({
            'success': True,
            'output': stdout_str,
            'command': full_cmd
        })
        
    except subprocess.TimeoutExpired:
        print("Error: Command execution timed out")
        # Kill the process if it times out
        process.kill()
        return jsonify({
            'success': False,
            'error': 'Command execution timed out after 60 seconds',
            'command': full_cmd if 'full_cmd' in locals() else 'Command not executed'
        }), 504
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'command': full_cmd if 'full_cmd' in locals() else 'Command not executed'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/api/attendance-dates', methods=['GET'])
def get_attendance_dates():
    """Retrieve a list of all available attendance dates from CSV files"""
    try:
        # Get list of all attendance CSV files in the Attendance folder
        attendance_path = os.path.join(ATTENDANCE_FOLDER, "Attendance_*.csv")
        attendance_files = glob.glob(attendance_path)
        
        # Extract dates from filenames
        dates = []
        for file in attendance_files:
            # Extract date part from "Attendance_DD-MM-YY.csv"
            date_match = re.search(r'Attendance_(\d{2}-\d{2}-\d{2})\.csv', os.path.basename(file))
            if date_match:
                dates.append(date_match.group(1))
        
        print(f"Found attendance dates: {dates}")
        return jsonify({"dates": dates})
    
    except Exception as e:
        print(f"Error retrieving attendance dates: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f"Failed to retrieve attendance dates: {str(e)}"
        }), 500

@app.route('/api/attendance-data', methods=['GET'])
def get_attendance_data():
    """Retrieve attendance data for a specific date"""
    try:
        date = request.args.get('date')
        if not date:
            return jsonify({
                'success': False,
                'error': "Date parameter is required"
            }), 400
        
        # Validate date format (DD-MM-YY)
        if not re.match(r'^\d{2}-\d{2}-\d{2}$', date):
            return jsonify({
                'success': False,
                'error': "Invalid date format. Expected format: DD-MM-YY"
            }), 400
        
        filename = os.path.join(ATTENDANCE_FOLDER, f"Attendance_{date}.csv")
        
        # Check if file exists
        if not os.path.exists(filename):
            return jsonify({
                'success': False,
                'error': f"Attendance file for date {date} not found"
            }), 404
        
        attendees = []
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader, None)  # Skip header row
            
            if header is None:
                return jsonify({"attendees": []})
            
            for row in csv_reader:
                if len(row) >= 2:
                    attendees.append({
                        "name": row[0],
                        "time": row[1]
                    })
        
        print(f"Retrieved {len(attendees)} attendees for date {date}")
        return jsonify({"attendees": attendees})
    
    except FileNotFoundError:
        return jsonify({
            'success': False,
            'error': f"Attendance file for date {date} not found"
        }), 404
    
    except Exception as e:
        print(f"Error retrieving attendance data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f"Failed to retrieve attendance data: {str(e)}"
        }), 500

# Handle CTRL+C gracefully
def signal_handler(sig, frame):
    print('Server shutting down...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
 # Ensure attendance folder path exists
    if not os.path.exists(ATTENDANCE_FOLDER):
        print(f"Warning: Attendance folder '{ATTENDANCE_FOLDER}' does not exist.")
        print(f"Creating attendance folder at '{ATTENDANCE_FOLDER}'")
        try:
            os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)
        except Exception as e:
            print(f"Failed to create attendance folder: {str(e)}")
    
    print("Starting backend server on http://127.0.0.1:5000")
    print("Make sure your virtual environment path is correct:")
    print(f"Current path: {VENV_PATH}")
    print(f"Attendance folder: {ATTENDANCE_FOLDER}")
    print("Press CTRL+C to exit")
    app.run(debug=True, host='0.0.0.0', port=5000)
