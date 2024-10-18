from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import threading

app = Flask(__name__)

model = load_model('model/emotion_detector_model_v1.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
camera = cv2.VideoCapture(0)

emotion_counts = {label: 0 for label in emotion_labels}
people_count = 0
people_data = []
start_time = time.time()  # Initialize start_time here
last_update_time = start_time

def update_emotion_counts():
    global emotion_counts, people_count, people_data, last_update_time, start_time
    while True:
        time.sleep(1)
        current_time = time.time()
        if current_time - last_update_time >= 1:
            last_update_time = current_time
            current_counts = emotion_counts.copy()
            people_data.append({
                'time': int(current_time - start_time),
                'people': people_count,
                'emotions': current_counts
            })
            emotion_counts = {label: 0 for label in emotion_labels}
            if len(people_data) > 60:
                people_data.pop(0)

def generate_frames():
    global emotion_counts, people_count
    face_tracker = {}
    while True:
        success, frame = camera.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        face_ids = {}
        for (x, y, w, h) in faces:
            face_id = f"{x}-{y}-{w}-{h}"
            face_ids[face_id] = True
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            prediction = model.predict(roi_gray)
            emotion = emotion_labels[np.argmax(prediction)]
            emotion_counts[emotion] += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        people_count = len(face_ids)
        
        for face_id in list(face_tracker.keys()):
            if face_id not in face_ids:
                del face_tracker[face_id]
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    return jsonify(emotion_counts=emotion_counts, people_data=people_data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    update_thread = threading.Thread(target=update_emotion_counts)
    update_thread.daemon = True
    update_thread.start()
    app.run(host='0.0.0.0', port=5000,debug=True)
