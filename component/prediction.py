from flask import Blueprint, jsonify, request, Flask
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
import pickle
import os
from PIL import Image
import time
import hashlib

# Create Flask application
app = Flask(__name__)
    
# Configuration settings for the Flask app
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FILE'] = 'mobilenetV2.h5'

prediction_routes = Blueprint('prediction_routes', __name__)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# Hash agar setiap foto yang di up namanya beda
def generate_unique_filename(original_filename):
    timestamp = str(int(time.time()))
    hash_object = hashlib.md5(original_filename.encode())
    unique_hash = hash_object.hexdigest()[:8]
    return f"{timestamp}_{unique_hash}_{original_filename}"

# Load model with label
model = load_model(app.config['MODEL_FILE'], compile=False)


# Define the custom_objects dictionary
custom_objects = {'F1Score': F1Score}

# Load the model with custom_objects
model = tf.keras.models.load_model('mobilenetV2.h5', custom_objects=custom_objects)

# Now, you can use the 'model' object to make predictions or perform other operations.

with open('class_indices_food_detection.pkl', 'rb') as indices:
    loaded_indices = pickle.load(indices)

@prediction_routes.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = generate_unique_filename(secure_filename(image.filename))
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            def preds(image_path):
                image = load_img(image_path, target_size=(224, 224))
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = image / 255.0
                predictions = model.predict(image)
                predicted_class = np.argmax(predictions, axis=1)
                class_labels = loaded_indices
                return list(class_labels)[predicted_class[0]]

            image_uploads = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            prediction_nutrition = preds(image_uploads)
            preds = preds(image_uploads)
            print(preds)

            return jsonify({
                "status": 200,
                "message": "File uploaded and saved successfully",
                "data": {
                    "prediction": preds,
                    "image_url": f"/{app.config['UPLOAD_FOLDER']}/{filename}"
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Client side error"
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None
        }), 405

# Register Blueprints
app.register_blueprint(prediction_routes)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
