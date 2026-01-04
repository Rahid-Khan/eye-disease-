# web_app/app.py

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow import keras
import traceback
from PIL import Image
import cv2

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# E:\code\eye_disease\project\models\eye_disease\ensemble_final.keras
# Load trained model
MODEL_PATH = '../models/eye_disease/ensemble_final.keras'
CLASS_NAMES = [
    'Central Serous Chorioretinopathy',
    'Diabetic Retinopathy',
    'Disc Edema',
    'Glaucoma',
    'Healthy',
    'Macular Scar',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

# Disease descriptions and recommendations
DISEASE_INFO = {
    'Central Serous Chorioretinopathy': {
        'description': 'A condition where fluid accumulates under the retina, causing vision distortion.',
        'recommendation': 'Consult a retinal specialist. May resolve on its own or require laser treatment.',
        'severity': 'Moderate'
    },
    'Diabetic Retinopathy': {
        'description': 'Diabetes complications that affect the eyes, potentially leading to blindness.',
        'recommendation': 'Immediate consultation with ophthalmologist. Control blood sugar levels.',
        'severity': 'High'
    },
    'Disc Edema': {
        'description': 'Swelling of the optic disc, often indicating increased intracranial pressure.',
        'recommendation': 'Urgent neurological evaluation required.',
        'severity': 'High'
    },
    'Glaucoma': {
        'description': 'A group of eye conditions that damage the optic nerve, often due to high eye pressure.',
        'recommendation': 'Regular eye pressure monitoring and treatment to prevent vision loss.',
        'severity': 'High'
    },
    'Healthy': {
        'description': 'No signs of pathological eye conditions detected.',
        'recommendation': 'Regular eye check-ups recommended annually.',
        'severity': 'None'
    },
    'Macular Scar': {
        'description': 'Scar tissue formation on the macula, affecting central vision.',
        'recommendation': 'Retinal specialist consultation for possible treatment options.',
        'severity': 'Moderate to High'
    },
    'Myopia': {
        'description': 'Nearsightedness - difficulty seeing distant objects clearly.',
        'recommendation': 'Regular eye examinations and appropriate corrective lenses.',
        'severity': 'Low'
    },
    'Pterygium': {
        'description': 'A non-cancerous growth of the conjunctiva that may extend to the cornea.',
        'recommendation': 'Monitor for growth. Surgical removal if it affects vision.',
        'severity': 'Low to Moderate'
    },
    'Retinal Detachment': {
        'description': 'Emergency condition where the retina separates from its underlying layer.',
        'recommendation': 'EMERGENCY - Immediate surgical intervention required.',
        'severity': 'Critical'
    },
    'Retinitis Pigmentosa': {
        'description': 'A genetic disorder that causes retinal degeneration.',
        'recommendation': 'Genetic counseling and low-vision rehabilitation services.',
        'severity': 'High'
    }
}

# Track model load error for debugging
MODEL_LOAD_ERROR = None

# Make DISEASE_INFO and model status available in all Jinja templates
@app.context_processor
def inject_disease_info():
    return dict(
        DISEASE_INFO=DISEASE_INFO,
        MODEL_LOADED=(model is not None) if 'model' in globals() else False,
        MODEL_LOAD_ERROR=MODEL_LOAD_ERROR,
    )

# Initialize model with improved error reporting and fallback
model = None
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception:
    # Try loading without compiling (sometimes helps across TF versions/custom losses)
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully with compile=False!")
    except Exception:
        MODEL_LOAD_ERROR = traceback.format_exc()
        print("Error loading model (see MODEL_LOAD_ERROR):")
        print(MODEL_LOAD_ERROR)
        model = None
        # write a short log to disk for convenience
        try:
            with open(os.path.join('models', 'model_load_error.log'), 'w', encoding='utf-8') as f:
                f.write(MODEL_LOAD_ERROR)
        except Exception:
            pass

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_disease(image_path):
    """Predict eye disease from image"""
    if model is None:
        return None
    
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    
    # Get top predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    
    results = []
    for idx in top_3_idx:
        disease = CLASS_NAMES[idx]
        confidence = float(predictions[0][idx])
        info = DISEASE_INFO.get(disease, {})
        
        results.append({
            'disease': disease,
            'confidence': confidence,
            'description': info.get('description', ''),
            'recommendation': info.get('recommendation', ''),
            'severity': info.get('severity', '')
        })
    
    return results

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', diseases=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        results = predict_disease(filepath)
        
        if results is None:
            return jsonify({'error': 'Model not available', 'details': MODEL_LOAD_ERROR}), 500
        
        return jsonify({
            'success': True,
            'predictions': results,
            'image_url': f'/static/uploads/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        results = predict_disease(filepath)
        
        # Clean up
        os.remove(filepath)
        
        if results is None:
            return jsonify({'error': 'Model not available'}), 500
        
        return jsonify({'predictions': results})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')


@app.route('/status')
def status():
    """Return model load status and details for debugging"""
    return jsonify({
        'model_loaded': (model is not None),
        'model_path': MODEL_PATH,
        'error': MODEL_LOAD_ERROR
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)