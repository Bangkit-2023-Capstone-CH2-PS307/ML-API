from flask import Flask
from component.form import form_routes
# from component.prediction import prediction_routes

app = Flask(__name__)

app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FILE'] = 'mobilenetV2.h5'

# Register Blueprints
app.register_blueprint(form_routes)
# app.register_blueprint(prediction_routes)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
