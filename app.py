from flask import Flask, request, jsonify
import os
import subprocess
import json
import requests
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)

UPLOAD_FOLDER = 'images/predict'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': '🚀 Server Flask đã deploy thành công!'})

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received predict request")
    
    if 'image' not in request.files:
        logger.error("No image uploaded")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        logger.error("Invalid file format. Only JPEG, JPG, or PNG allowed")
        return jsonify({'error': 'Chỉ hỗ trợ file ảnh định dạng JPEG, JPG hoặc PNG!'}), 400

    filename = secure_filename(f"{os.urandom(24).hex()}{os.path.splitext(file.filename)[1]}")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)
    logger.info(f"Image uploaded to: {image_path}")

    try:
        model_path = os.getenv("MODEL_PATH", "/app/AI/plant-disease-model-complete.pth")
        if not os.path.exists(model_path):
            model_url = os.getenv("MODEL_URL", "YOUR_DEFAULT_MODEL_URL")
            logger.info(f"Downloading model from {model_url}")
            os.makedirs('/app/AI', exist_ok=True)
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("Model downloaded successfully")

        normalized_image_path = image_path.replace('\\', '/')
        command = f'python3 AI/predict.py "{normalized_image_path}"'  # Quay lại python3 cho Fly.io
        logger.info(f"Executing command: {command}")

        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if process.returncode != 0:
            logger.error(f"Python script error: {process.stderr}")
            return jsonify({'error': 'Prediction failed', 'details': process.stderr}), 500

        if not process.stdout:
            logger.error("No output from Python script")
            return jsonify({'error': 'No prediction returned from script'}), 500

        try:
            result = json.loads(process.stdout.strip())
            logger.info(f"Prediction result: {result}")
            return jsonify(result)
        except json.JSONDecodeError as parse_err:
            logger.error(f"Error parsing Python output: {str(parse_err)}")
            return jsonify({'error': 'Invalid prediction result', 'details': str(parse_err)}), 500

    except subprocess.TimeoutExpired:
        logger.error("Python script timed out")
        return jsonify({'error': 'Prediction timed out'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'Unexpected error', 'details': str(e)}), 500
    finally:
        try:
            if os.path.exists(image_path):
                os.unlink(image_path)
                logger.info(f"Deleted image: {image_path}")
            else:
                logger.warning(f"Image not found for deletion: {image_path}")
        except Exception as unlink_err:
            logger.error(f"Error deleting image {image_path}: {str(unlink_err)}")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)  # Sử dụng port 8080 cho Fly.io