from flask import Flask, request, jsonify
import os
import subprocess
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Giới hạn kích thước file tải lên (1MB)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

# Cấu hình thư mục upload
UPLOAD_FOLDER = 'images/predict'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Tạo thư mục upload nếu chưa tồn tại
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': '🚀 Flask server is deployed successfully!'})

@app.route('/predict', methods=['POST'])
def predict():
    print('Nhận yêu cầu dự đoán')
    
    # Kiểm tra xem có file được tải lên không
    if 'image' not in request.files:
        print('Không có ảnh được tải lên')
        return jsonify({'error': 'Không có ảnh được tải lên'}), 400
    
    file = request.files['image']
    
    # Kiểm tra file hợp lệ
    if file.filename == '':
        print('Không chọn ảnh')
        return jsonify({'error': 'Không chọn ảnh'}), 400
    
    if not allowed_file(file.filename):
        print('Định dạng file không hợp lệ')
        return jsonify({'error': 'Chỉ hỗ trợ file JPEG, JPG hoặc PNG'}), 400

    # Lưu file với tên duy nhất
    unique_filename = f"{uuid.uuid4().hex}{os.path.splitext(secure_filename(file.filename))[1]}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(image_path)
    print(f'Ảnh được tải lên: {image_path}')

    # Chuẩn hóa đường dẫn
    normalized_image_path = image_path.replace('\\', '/')
    command = f'python AI/predict.py "{normalized_image_path}"'
    print(f'Thực thi lệnh: {command}')

    try:
        # Chạy script dự đoán với timeout 10 giây
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Xóa ảnh sau khi xử lý
        try:
            os.unlink(image_path)
            print(f'Đã xóa ảnh: {image_path}')
        except Exception as unlink_err:
            print(f'Lỗi xóa ảnh: {image_path}, {unlink_err}')

        # Kiểm tra lỗi từ script
        if result.returncode != 0:
            print('Lỗi script Python:', result.stderr)
            return jsonify({'error': 'Dự đoán thất bại', 'details': result.stderr}), 500

        print('Kết quả stdout:', result.stdout)
        print('Kết quả stderr:', result.stderr)

        if not result.stdout:
            print('Không có đầu ra từ script Python')
            return jsonify({'error': 'Không có kết quả dự đoán từ script'}), 500

        # Phân tích kết quả
        try:
            prediction_result = eval(result.stdout.strip())  # Giả sử predict.py trả về dict
            print('Kết quả dự đoán:', prediction_result)
            return jsonify(prediction_result)
        except Exception as parse_err:
            print('Lỗi phân tích đầu ra Python:', parse_err)
            return jsonify({'error': 'Kết quả dự đoán không hợp lệ', 'details': str(parse_err)}), 500

    except subprocess.TimeoutExpired as timeout_err:
        print('Hết thời gian chạy script:', timeout_err)
        try:
            os.unlink(image_path)
            print(f'Đã xóa ảnh: {image_path}')
        except Exception as unlink_err:
            print(f'Lỗi xóa ảnh: {image_path}, {unlink_err}')
        return jsonify({'error': 'Hết thời gian dự đoán', 'details': str(timeout_err)}), 500
    except Exception as err:
        print('Lỗi không xác định:', err)
        try:
            os.unlink(image_path)
            print(f'Đã xóa ảnh: {image_path}')
        except Exception as unlink_err:
            print(f'Lỗi xóa ảnh: {image_path}, {unlink_err}')
        return jsonify({'error': 'Dự đoán thất bại', 'details': str(err)}), 500

if __name__ == '__main__':
    app.run(debug=False) 