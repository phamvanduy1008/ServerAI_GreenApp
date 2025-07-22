from flask import Flask, request, jsonify
import os
import subprocess
import uuid
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Cấu hình thư mục upload
UPLOAD_FOLDER = 'images/predict'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': '🚀 Flask server is deployed successfully!'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only JPEG, JPG, or PNG files are supported'}), 400

    # Lưu ảnh
    filename = f"{uuid.uuid4().hex}{os.path.splitext(secure_filename(file.filename))[1]}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Đường dẫn chuẩn hóa cho lệnh shell
    normalized_path = image_path.replace('\\', '/')
    command = ['python3', 'AI/predict.py', normalized_path]

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)

        os.remove(image_path)  # Xóa ảnh sau khi xử lý

        if result.returncode != 0:
            return jsonify({'error': 'Prediction failed', 'details': result.stderr.strip()}), 500

        # Chuyển kết quả stdout (dạng JSON) thành dict
        try:
            prediction_result = json.loads(result.stdout.strip())
            return jsonify(prediction_result)
        except json.JSONDecodeError as e:
            return jsonify({'error': 'Failed to parse prediction result', 'details': str(e)}), 500

    except subprocess.TimeoutExpired:
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'error': 'Prediction script timed out'}), 500

    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run()
