"""
Plant Leaf Disease Detection - Flask Backend
==============================================
Serves the frontend, accepts an uploaded leaf image, preprocesses it with
NumPy/Pillow, runs it through a trained Keras (.h5) CNN model, and returns
a JSON response with the predicted disease, confidence score, and
prevention/treatment guidance.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import io
import os
import json

import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError

# TensorFlow / Keras is imported lazily (see load_model_safe) so the server
# can still start and explain itself clearly if the model file is missing
# or TensorFlow isn't installed yet.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "plant_disease_model.h5")
CLASS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "model", "class_indices.json")
DISEASE_INFO_PATH = os.path.join(os.path.dirname(__file__), "model", "disease_info.json")

IMG_SIZE = (224, 224)  # must match the size used during training
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB upload limit

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Model / metadata loading
# ---------------------------------------------------------------------------

_model = None
_class_names = None
_disease_info = None


def load_class_names():
    """Load class index -> class name mapping produced during training."""
    global _class_names
    if _class_names is not None:
        return _class_names
    if os.path.exists(CLASS_INDEX_PATH):
        with open(CLASS_INDEX_PATH, "r") as f:
            idx_to_class = json.load(f)
        # JSON keys are strings; sort numerically to get a clean list
        _class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    else:
        _class_names = None
    return _class_names


def load_disease_info():
    """Load disease name -> {description, prevention_steps} lookup."""
    global _disease_info
    if _disease_info is not None:
        return _disease_info
    if os.path.exists(DISEASE_INFO_PATH):
        with open(DISEASE_INFO_PATH, "r") as f:
            _disease_info = json.load(f)
    else:
        _disease_info = {}
    return _disease_info


def get_model():
    """Lazily load the trained Keras model (keeps startup fast and gives a
    clear error message if the model hasn't been trained/exported yet)."""
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'. "
            "Train the model using model_training.ipynb in Google Colab, "
            "then download plant_disease_model.h5 into the model/ folder."
        )
    import tensorflow as tf  # imported here so app.py can run even without TF installed yet

    _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_stream):
    """Load an image from a file stream, resize it, and normalize it into
    a (1, H, W, 3) float32 NumPy array in the [0, 1] range, matching the
    preprocessing used during training in model_training.ipynb."""
    image = Image.open(file_stream)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.asarray(image, dtype=np.float32)
    img_array = img_array / 255.0  # normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def humanize_label(raw_label):
    """Turn a PlantVillage-style label like 'Tomato___Early_blight' into
    'Tomato - Early Blight' for friendlier display."""
    parts = raw_label.replace("___", "__").split("__")
    cleaned = [p.replace("_", " ").strip() for p in parts if p.strip()]
    return " - ".join(cleaned) if cleaned else raw_label


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file found in the request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        }), 400

    try:
        img_bytes = io.BytesIO(file.read())
        img_array = preprocess_image(img_bytes)
    except UnidentifiedImageError:
        return jsonify({"error": "The uploaded file isn't a valid image."}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to process the image: {exc}"}), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        predictions = model.predict(img_array)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Model prediction failed: {exc}"}), 500

    class_names = load_class_names()
    disease_info = load_disease_info()

    pred_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100

    if class_names and pred_index < len(class_names):
        raw_label = class_names[pred_index]
    else:
        raw_label = f"class_{pred_index}"

    friendly_name = humanize_label(raw_label)
    info = disease_info.get(raw_label, {})

    is_healthy = "healthy" in raw_label.lower()

    response = {
        "disease_name": friendly_name,
        "confidence": round(confidence, 2),
        "is_healthy": is_healthy,
        "description": info.get(
            "description",
            "No detailed description available for this class yet. "
            "Add an entry for this label in model/disease_info.json."
        ),
        "prevention_steps": info.get(
            "prevention_steps",
            [
                "No specific guidance stored for this class yet.",
                "Add an entry for this label in model/disease_info.json.",
            ],
        ),
    }
    return jsonify(response), 200


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File too large. Maximum upload size is 8 MB."}), 413


if __name__ == "__main__":
    app.run(debug=True)
