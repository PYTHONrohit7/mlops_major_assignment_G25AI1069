import joblib
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = joblib.load('savedmodel.pth')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    try:
        # Read the image
        img = Image.open(io.BytesIO(file.read())).convert('L') # Convert to grayscale
        img = img.resize((64, 64)) # Olivetti faces are 64x64
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Flatten the array to 1D (4096 features)
        img_flattened = img_array.flatten().reshape(1, -1)
        
        # Predict
        prediction = model.predict(img_flattened)
        
        return f"Predicted Class: {prediction[0]}"
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
