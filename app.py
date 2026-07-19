import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tf_keras as keras
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = 300

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "model/retina_model.h5"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("Warning: No trained model found. Train the model first using train.py")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return jsonify({"status": "RetinaAI backend is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img_array = preprocess_image(filepath)
    predictions = model.predict(img_array)
    grade = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    all_probabilities = [round(float(p), 4) for p in predictions[0]]

    os.remove(filepath)

    return jsonify({
        "grade": grade,
        "confidence": confidence,
        "all_probabilities": all_probabilities
    })

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
    