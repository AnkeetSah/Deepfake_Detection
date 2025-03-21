from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
lstm_model = load_model("deepfake_detection_lstm.keras")


def extract_frames_from_video(video_path, frame_interval=30):
    """Extract frames from the given video at intervals."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        return frames
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while frame_count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += frame_interval
    
    cap.release()
    print(f"Extracted {len(frames)} frames from the video.")
    return frames


def extract_features_from_frames(frames):
    """Extract features from frames using the ResNet model."""
    features = []
    batch_size = 16  
    batch_frames = []

    for frame in frames:
        img = cv2.resize(frame, (224, 224))
        img_array = img_to_array(img)
        batch_frames.append(preprocess_input(img_array))

        if len(batch_frames) == batch_size:
            batch_features = resnet_model.predict(np.array(batch_frames), verbose=0)
            features.extend(batch_features)
            batch_frames = []

    if batch_frames:
        batch_features = resnet_model.predict(np.array(batch_frames), verbose=0)
        features.extend(batch_features)

    return np.array(features)


def preprocess_for_lstm(features, time_steps=10):
    """Preprocess features for LSTM."""
    num_frames = features.shape[0]

    if num_frames < time_steps:
        padding = np.zeros((time_steps - num_frames, features.shape[1]))
        features = np.vstack([features, padding])
    else:
        features = features[:time_steps]

    return features.reshape(1, time_steps, features.shape[1])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'uploadFile' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    file = request.files['uploadFile']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Video processing
        frames = extract_frames_from_video(file_path, frame_interval=30)
        if not frames:
            return jsonify({"message": "Error processing video or empty frames"}), 500

        # Feature extraction and prediction
        video_features = extract_features_from_frames(frames)
        if video_features.size == 0:
            return jsonify({"message": "No features extracted from video"}), 500

        video_features_lstm = preprocess_for_lstm(video_features)
        prediction = (lstm_model.predict(video_features_lstm) > 0.5).astype("int32")

        result = "Fake" if prediction == 1 else "Real"
        
        # Delete the uploaded video after processing
        os.remove(file_path)

        return jsonify({"message": f"Prediction for the video: {result}"})

    except Exception as e:
        # In case of any error, make sure to remove the uploaded video as well
        os.remove(file_path)
        return jsonify({"message": f"Error processing video: {str(e)}"}), 500



# Route to serve files from the uploads folder
@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
