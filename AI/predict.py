import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import io
import json

# Thiết lập mã hóa UTF-8 cho stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Script started", file=sys.stderr)

# Định nghĩa kiến trúc ResNet9
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(torch.nn.MaxPool2d(4))
        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class ResNet9(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = torch.nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = torch.nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = torch.nn.Sequential(
            torch.nn.MaxPool2d(4),
            torch.nn.Flatten(),
            torch.nn.Linear(512, num_classes)
        )
    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Danh sách các lớp
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Ánh xạ nhãn sang tên bệnh tiếng Việt
disease_names_vn = {
    'Apple___Apple_scab': 'Bệnh vảy nến táo',
    'Apple___Black_rot': 'Thối đen',
    'Apple___Cedar_apple_rust': 'Rỉ sắt táo',
    'Apple___healthy': 'Táo khỏe mạnh',
    'Blueberry___healthy': 'Việt quất khỏe mạnh',
    'Cherry_(including_sour)___Powdery_mildew': 'Bệnh phấn trắng anh đào',
    'Cherry_(including_sour)___healthy': 'Anh đào khỏe mạnh',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Đốm lá xám ngô',
    'Corn_(maize)___Common_rust_': 'Rỉ sắt thông thường ngô',
    'Corn_(maize)___Northern_Leaf_Blight': 'Bệnh cháy lá phía bắc ngô',
    'Corn_(maize)___healthy': 'Ngô khỏe mạnh',
    'Grape___Black_rot': 'Thối đen nho',
    'Grape___Esca_(Black_Measles)': 'Bệnh đốm đen nho',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Bệnh cháy lá nho',
    'Grape___healthy': 'Nho khỏe mạnh',
    'Orange___Haunglongbing_(Citrus_greening)': 'Bệnh vàng lá cam',
    'Peach___Bacterial_spot': 'Đốm vi khuẩn đào',
    'Peach___healthy': 'Đào khỏe mạnh',
    'Pepper,_bell___Bacterial_spot': 'Đốm vi khuẩn ớt chuông',
    'Pepper,_bell___healthy': 'Ớt chuông khỏe mạnh',
    'Potato___Early_blight': 'Bệnh cháy lá sớm khoai tây',
    'Potato___Late_blight': 'Bệnh cháy lá muộn khoai tây',
    'Potato___healthy': 'Khoai tây khỏe mạnh',
    'Raspberry___healthy': 'Mâm xôi khỏe mạnh',
    'Soybean___healthy': 'Đậu tương khỏe mạnh',
    'Squash___Powdery_mildew': 'Bệnh phấn trắng bí',
    'Strawberry___Leaf_scorch': 'Bệnh cháy lá dâu tây',
    'Strawberry___healthy': 'Dâu tây khỏe mạnh',
    'Tomato___Bacterial_spot': 'Đốm vi khuẩn cà chua',
    'Tomato___Early_blight': 'Bệnh cháy lá sớm cà chua',
    'Tomato___Late_blight': 'Bệnh cháy lá muộn cà chua',
    'Tomato___Leaf_Mold': 'Bệnh mốc lá cà chua',
    'Tomato___Septoria_leaf_spot': 'Đốm lá Septoria cà chua',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Nhện đỏ hại cà chua',
    'Tomato___Target_Spot': 'Đốm mục tiêu cà chua',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Virus xoăn lá vàng cà chua',
    'Tomato___Tomato_mosaic_virus': 'Virus khảm cà chua',
    'Tomato___healthy': 'Cà chua khỏe mạnh'
}

# Giải pháp khắc phục cho các bệnh
disease_solutions_vn = {
    'Apple___Apple_scab': [
        'Phun thuốc trừ nấm như captan hoặc myclobutanil vào đầu mùa xuân.',
        'Loại bỏ lá rụng và cành khô để giảm nguồn lây bệnh.',
        'Tưới nước vào buổi sáng để lá khô nhanh, tránh ẩm ướt kéo dài.',
        'Trồng các giống táo kháng bệnh như Enterprise hoặc Liberty'
    ],
    # (Giữ nguyên các giải pháp như trong file gốc)
    'Apple___Black_rot': [
        'Cắt bỏ cành và quả bị nhiễm bệnh, tiêu hủy để tránh lây lan.',
        'Phun thuốc trừ nấm như sulfur hoặc captan trong mùa sinh trưởng.',
        'Đảm bảo cây được thông thoáng bằng cách tỉa cành hợp lý.',
        'Kiểm tra và vệ sinh dụng cụ làm vườn để tránh lây nhiễm.'
    ],
    # ... (Các giải pháp khác giữ nguyên như trong file gốc)
    'Tomato___healthy': ['Tiếp tục chăm sóc tốt, tưới nước đều, bón phân cân đối và kiểm tra sâu bệnh.']
}

def predict_image(image_path, model_path):
    print("Starting prediction process", file=sys.stderr)
    
    # Kiểm tra file mô hình
    print(f"Checking model file: {model_path}", file=sys.stderr)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Kiểm tra file ảnh
    print(f"Checking image file: {image_path}", file=sys.stderr)
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    # Tải mô hình
    print("Loading model...", file=sys.stderr)
    try:
        # Sử dụng torch.jit.load để tối ưu hóa trên PythonAnywhere
        model = torch.jit.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        print("Model loaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Tải và xử lý ảnh
    print("Processing image...", file=sys.stderr)
    try:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        print("Image processed successfully", file=sys.stderr)
    except Exception as e:
        print(f"Error processing image: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Dự đoán
    print("Running prediction...", file=sys.stderr)
    try:
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, dim=1)
            class_label = classes[pred.item()]
            predicted_disease = disease_names_vn[class_label]
            solutions = disease_solutions_vn[class_label]
        print("Prediction completed", file=sys.stderr)
    except Exception as e:
        print(f"Error during prediction: {str(e)}", file=sys.stderr)
        sys.exit(1)

    return {"prediction": predicted_disease, "solutions": solutions}

if __name__ == '__main__':
    print("Checking arguments...", file=sys.stderr)
    if len(sys.argv) < 2:
        print("Error: Image path is required", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    # Sử dụng đường dẫn tương đối phù hợp với PythonAnywhere
    model_path = os.path.join(os.path.dirname(__file__), 'plant-disease-model-complete.pth')
    
    print(f"Image path: {image_path}", file=sys.stderr)
    print(f"Model path: {model_path}", file=sys.stderr)
    
    try:
        result = predict_image(image_path, model_path)
        print(json.dumps(result, ensure_ascii=False))  # Đảm bảo hỗ trợ UTF-8
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)
