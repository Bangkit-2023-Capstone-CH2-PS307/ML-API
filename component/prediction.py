from flask import Blueprint, jsonify, request, Flask
import numpy as np
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
from google.cloud import firestore

# Create Flask application
app = Flask(__name__)

# Configuration settings for the Flask app
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FILE'] = 'mobilenetV2.h5'

# Initialize Firestore client
db = firestore.Client()

prediction_routes = Blueprint('prediction_routes', __name__)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def generate_unique_filename(original_filename):
    timestamp = str(int(time.time()))
    hash_object = hashlib.md5(original_filename.encode())
    unique_hash = hash_object.hexdigest()[:8]
    return f"{timestamp}_{unique_hash}_{original_filename}"

model = load_model(app.config['MODEL_FILE'], compile=False)
custom_objects = {'F1Score': F1Score}
model = tf.keras.models.load_model('mobilenetV2.h5', custom_objects=custom_objects)

with open('class_indices_food_detection.pkl', 'rb') as indices:
    loaded_indices = pickle.load(indices)

@prediction_routes.route("/prediction", methods=["POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            try:
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
                
                label_map = {
                    'gambar agar-agar' :'Agar-Agar',
                    'bubur' : 'Bubur',
                    'cheese' : 'Cheese',
                    'daging-cincang' : 'Daging Cincang',
                    'kentang' : 'Kentang',
                    'olahan ikan' : 'Ikan',
                    'susu' : 'Susu',
                    'telur' : 'Telur',
                    'wortel' : 'Wortel',
                    'yogurt' : 'Yogurt'
                }
                
                prediction_nutrition = label_map[prediction_nutrition]
                # Get description from Firestore based on the prediction
                database_ref = db.collection('foods').document(prediction_nutrition)
                data = database_ref.get()
                
                if data.exists:
                    document_data = data.to_dict()

                    return jsonify({
                        "status": 200,
                        "message": "File uploaded and saved successfully",
                        "data": {
                            "prediction": prediction_nutrition,
                            "description": document_data.get('description', ''),
                            "nutritions": document_data.get('nutritions', ''),
                            "percentages": document_data.get('percentages', ''),
                            "image_url": f"/{app.config['UPLOAD_FOLDER']}/{filename}"
                        }
                    }), 200
                else:
                    return jsonify({
                        "status": 404,
                        "message": "Data not found"
                    }), 404
            except Exception as e:
                return jsonify({
                    "status": 500,
                    "message": str(e),
                }), 500
        else:
            return jsonify({
                "status": 400,
                "message": "Incorrect format. Only accept: png, jpg, jpeg"
            }), 400
    else:
        return jsonify({
            "status": 405,
            "message": "Method not allowed"
        }), 405

# Register Blueprints
app.register_blueprint(prediction_routes)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
