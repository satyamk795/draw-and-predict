from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS if frontend is separate

# Load trained model and label encoder
model = tf.keras.models.load_model("draw_model.keras")  # Updated extension
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def preprocess_image(data_url):
    try:
        header, encoded = data_url.split(",", 1)
        decoded = base64.b64decode(encoded)

        # Load image, convert to grayscale, resize to 28x28
        image = Image.open(io.BytesIO(decoded)).convert("L").resize((28, 28))

        # Invert colors: black background and white drawing
        image = np.array(image)
        image = 255 - image  # invert colors
        image = image / 255.0  # normalize
        image = image.reshape(1, 28, 28, 1)  # add batch and channel dims

        return image
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("image", None)
        if data is None:
            return jsonify({"error": "No image data provided"}), 400

        img = preprocess_image(data)
        prediction = model.predict(img)[0]
        label_index = np.argmax(prediction)
        label = label_encoder.inverse_transform([label_index])[0]
        confidence = float(np.max(prediction))

        return jsonify({
            "label": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

