from flask import Flask, render_template, Response, jsonify, request
import cv2
import face_recognition
from ultralytics import YOLO
import os
import numpy as np
import threading
import requests
from pymongo import MongoClient
import io
from PIL import Image
from playsound import playsound
from dotenv import load_dotenv

app = Flask(__name__)

model = YOLO('yolov5n.pt')  # Load YOLO model

dangerous_objects = ["knife", "gun", "bomb"]
known_face_encodings = []
known_face_names = []

task_running = False  # Flag to control surveillance

load_dotenv()

# Use environment variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
FLASK_RUN_PORT = int(os.getenv("FLASK_RUN_PORT", 10000))  
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True") == "True"

def load_known_faces():
    global known_face_encodings, known_face_names
    print("Connecting to MongoDB...")

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    documents = collection.find({}, {"image": 1, "title": 1})
    print("Fetching known faces from MongoDB...")

    faces_folder = "faces_folder"
    os.makedirs(faces_folder, exist_ok=True)

    for doc in documents:
        try:
            if "image" in doc and "url" in doc["image"]:
                image_url = doc["image"]["url"]
                title = doc.get("title", "Unknown")
                image_path = os.path.join(faces_folder, f"{title}.jpg")
                response = requests.get(image_url, stream=True)
                
                if response.status_code == 200:
                    with open(image_path, "wb") as image_file:
                        for chunk in response.iter_content(1024):
                            image_file.write(chunk)
                    print(f"Image downloaded for {title}")

        except Exception as e:
            print(f"Error processing image for {doc.get('title', 'Unknown')}: {e}")

    for filename in os.listdir(faces_folder):
        image_path = os.path.join(faces_folder, filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
                print(f"Successfully processed {filename}")
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    client.close()
    print(f"Total faces loaded: {len(known_face_encodings)}")

@app.route('/')
def index():
    return render_template('index.html')

# Function to play alert sound
def play_alert_sound():
    playsound("alert.mp3", block=False) 

@app.route('/start_surveillance', methods=['POST'])
def start_surveillance():
    global task_running
    task_running = True
    return jsonify({"message": "Surveillance started!"})

@app.route('/stop_surveillance', methods=['POST'])
def stop_surveillance():
    global task_running
    task_running = False
    return jsonify({"message": "Surveillance stopped!"})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global task_running
    if not task_running:
        return Response(status=204)

    file = request.files['frame']
    image = Image.open(io.BytesIO(file.read()))
    frame = np.array(image)

    # Face Recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[best_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        play_alert_sound()

    # Object Detection
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])
            class_name = model.names[class_id] if class_id < len(model.names) else f"Unknown ({class_id})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    return Response(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    load_known_faces()
    app.run(host='0.0.0.0', port=10000, debug=True)