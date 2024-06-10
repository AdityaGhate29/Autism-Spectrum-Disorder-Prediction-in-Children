from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os

from load_image import image
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Define the directory to store uploaded images
UPLOAD_FOLDER = '/home/thor/ASDUsingImage/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the pre-trained model
model = tf.keras.models.load_model('/home/thor/Downloads/vgg_model.h5')

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the index page
@app.route('/')
def index():
    return render_template('index11.html')

@app.route('/')
def prediction():
    return render_template('prediction.html')

# Route for uploading an image and making predictions
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Preprocess the uploaded image
        img = filepath
        img = image(img)
        
        # Make predictions using the loaded model
        predictions = model.predict(img).argmax()
        
        # Assuming the model returns a single class prediction
        if predictions == 1:
            class_index = 'Autistic'
        else:
            class_index = 'Non Autistic'
        
        # Render the prediction result with the appropriate styling
        return render_template("prediction.html", class_index=class_index)
    
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
