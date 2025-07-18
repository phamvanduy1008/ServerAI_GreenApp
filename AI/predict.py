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

# Định nghĩa kiến trúc ResNet9 giống như trong code huấn luyện
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
        'Trồng các giống táo kháng bệnh như Enterprise hoặc Liberty.'
    ],
    'Apple___Black_rot': [
        'Cắt bỏ cành và quả bị nhiễm bệnh, tiêu hủy để tránh lây lan.',
        'Phun thuốc trừ nấm như sulfur hoặc captan trong mùa sinh trưởng.',
        'Đảm bảo cây được thông thoáng bằng cách tỉa cành hợp lý.',
        'Kiểm tra và vệ sinh dụng cụ làm vườn để tránh lây nhiễm.'
    ],
    'Apple___Cedar_apple_rust': [
        'Loại bỏ cây tuyết tùng gần vườn táo vì chúng là vật chủ trung gian.',
        'Phun thuốc trừ nấm như myclobutanil hoặc triadimefon vào đầu mùa.',
        'Tỉa cành để tăng lưu thông không khí và giảm độ ẩm.',
        'Trồng các giống táo kháng bệnh như Freedom hoặc Enterprise.'
    ],
    'Apple___healthy': ['Tiếp tục duy trì chăm sóc tốt, tưới nước đều, bón phân cân đối và kiểm tra định kỳ.'],
    'Blueberry___healthy': ['Duy trì độ pH đất từ 4.5-5.5, tưới nước đều và kiểm tra sâu bệnh thường xuyên.'],
    'Cherry_(including_sour)___Powdery_mildew': [
        'Phun thuốc trừ nấm như sulfur hoặc potassium bicarbonate khi phát hiện dấu hiệu bệnh.',
        'Tỉa cành để cải thiện lưu thông không khí và ánh sáng.',
        'Tránh tưới nước lên lá, tưới trực tiếp vào gốc.',
        'Loại bỏ lá và cành bị nhiễm, tiêu hủy đúng cách.'
    ],
    'Cherry_(including_sour)___healthy': ['Tiếp tục chăm sóc tốt, tỉa cành định kỳ và bón phân hữu cơ.'],
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
        'Sử dụng thuốc trừ nấm như azoxystrobin hoặc pyraclostrobin khi bệnh xuất hiện.',
        'Luân canh cây trồng để giảm tích lũy mầm bệnh trong đất.',
        'Loại bỏ tàn dư cây trồng sau thu hoạch.',
        'Trồng giống ngô kháng bệnh nếu có.'
    ],
    'Corn_(maize)___Common_rust_': [
        'Phun thuốc trừ nấm như triazole hoặc strobilurin vào giai đoạn đầu nhiễm bệnh.',
        'Trồng giống ngô kháng rỉ sắt.',
        'Đảm bảo khoảng cách trồng hợp lý để tăng thông thoáng.',
        'Theo dõi thời tiết ẩm ướt để phun thuốc phòng ngừa.'
    ],
    'Corn_(maize)___Northern_Leaf_Blight': [
        'Sử dụng thuốc trừ nấm như mancozeb hoặc chlorothalonil.',
        'Luân canh cây trồng với cây không phải ngô.',
        'Tiêu hủy tàn dư cây trồng để giảm nguồn bệnh.',
        'Trồng giống ngô kháng bệnh như DeKalb hoặc Pioneer.'
    ],
    'Corn_(maize)___healthy': ['Tiếp tục luân canh cây trồng, bón phân cân đối và kiểm tra sâu bệnh.'],
    'Grape___Black_rot': [
        'Phun thuốc trừ nấm như myclobutanil hoặc captan trước và sau khi ra hoa.',
        'Loại bỏ quả và lá bị nhiễm, tiêu hủy ngay lập tức.',
        'Tỉa cành để tăng thông thoáng và giảm độ ẩm.',
        'Trồng giống nho kháng bệnh như Concord.'
    ],
    'Grape___Esca_(Black_Measles)': [
        'Cắt bỏ cành bị nhiễm và tiêu hủy, tránh để mầm bệnh lây lan.',
        'Sử dụng thuốc trừ nấm như fosetyl-Al vào đầu mùa.',
        'Đảm bảo hệ thống thoát nước tốt để tránh stress cho cây.',
        'Kiểm tra sức khỏe cây định kỳ và bón phân hợp lý.'
    ],
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': [
        'Phun thuốc trừ nấm như captan hoặc myclobutanil khi lá bắt đầu phát triển.',
        'Loại bỏ lá bị nhiễm và vệ sinh vườn sạch sẽ.',
        'Tỉa cành để cải thiện lưu thông không khí.',
        'Tránh tưới nước lên lá, tưới vào gốc cây.'
    ],
    'Grape___healthy': ['Duy trì chăm sóc tốt, tỉa cành và phun thuốc phòng ngừa định kỳ.'],
    'Orange___Haunglongbing_(Citrus_greening)': [
        'Loại bỏ và tiêu hủy cây bị nhiễm nặng để tránh lây lan.',
        'Kiểm soát côn trùng chích hút như rệp sáp bằng thuốc trừ sâu.',
        'Bón phân cân đối để tăng sức đề kháng cho cây.',
        'Hợp tác với cơ quan nông nghiệp để kiểm soát bệnh khu vực.'
    ],
    'Peach___Bacterial_spot': [
        'Phun thuốc chứa đồng (copper-based) vào đầu mùa xuân.',
        'Trồng giống đào kháng bệnh như Redhaven.',
        'Loại bỏ lá và quả bị nhiễm, vệ sinh vườn sạch sẽ.',
        'Đảm bảo cây thông thoáng bằng cách tỉa cành.'
    ],
    'Peach___healthy': ['Tiếp tục chăm sóc tốt, tưới nước đều và kiểm tra bệnh định kỳ.'],
    'Pepper,_bell___Bacterial_spot': [
        'Sử dụng thuốc chứa đồng hoặc streptomycin để kiểm soát vi khuẩn.',
        'Loại bỏ lá và quả bị nhiễm, tiêu hủy đúng cách.',
        'Luân canh cây trồng với cây không phải ớt.',
        'Tưới nước vào gốc, tránh làm ướt lá.'
    ],
    'Pepper,_bell___healthy': ['Duy trì đất tơi xốp, bón phân cân đối và kiểm tra sâu bệnh.'],
    'Potato___Early_blight': [
        'Phun thuốc trừ nấm như chlorothalonil hoặc mancozeb khi bệnh xuất hiện.',
        'Luân canh cây trồng với cây không phải khoai tây.',
        'Loại bỏ tàn dư cây trồng sau thu hoạch.',
        'Trồng giống khoai tây kháng bệnh như Kennebec.'
    ],
    'Potato___Late_blight': [
        'Sử dụng thuốc trừ nấm như metalaxyl hoặc chlorothalonil ngay khi có dấu hiệu bệnh.',
        'Loại bỏ và tiêu hủy cây bị nhiễm nặng.',
        'Tránh tưới nước lên lá, tưới vào gốc cây.',
        'Theo dõi thời tiết ẩm ướt để phun thuốc phòng ngừa.'
    ],
    'Potato___healthy': ['Tiếp tục luân canh cây trồng, bón phân hữu cơ và kiểm tra bệnh.'],
    'Raspberry___healthy': ['Duy trì chăm sóc tốt, tỉa cành và bón phân cân đối.'],
    'Soybean___healthy': ['Tiếp tục luân canh cây trồng, kiểm tra sâu bệnh và bón phân hợp lý.'],
    'Squash___Powdery_mildew': [
        'Phun thuốc trừ nấm như sulfur hoặc myclobutanil khi phát hiện bệnh.',
        'Tỉa cành để tăng lưu thông không khí và ánh sáng.',
        'Tưới nước vào buổi sáng để lá khô nhanh.',
        'Loại bỏ lá bị nhiễm và vệ sinh vườn.'
    ],
    'Strawberry___Leaf_scorch': [
        'Phun thuốc trừ nấm như captan hoặc myclobutanil.',
        'Loại bỏ lá bị nhiễm và tiêu hủy đúng cách.',
        'Đảm bảo đất thoát nước tốt và cây thông thoáng.',
        'Tưới nước vào gốc, tránh làm ướt lá.'
    ],
    'Strawberry___healthy': ['Duy trì đất tơi xốp, b HIGHLIGHTón phân hữu cơ và kiểm tra sâu bệnh.'],
    'Tomato___Bacterial_spot': [
        'Sử dụng thuốc chứa đồng hoặc streptomycin để kiểm soát vi khuẩn.',
        'Loại bỏ lá và quả bị nhiễm, tiêu hủy ngay.',
        'Luân canh cây trồng với cây không phải cà chua.',
        'Tưới nước vào gốc, tránh làm ướt lá.'
    ],
    'Tomato___Early_blight': [
        'Phun thuốc trừ nấm như chlorothalonil hoặc mancozeb.',
        'Loại bỏ lá bị nhiễm ở phần dưới cây.',
        'Tỉa cành để tăng lưu thông không khí.',
        'Luân canh cây trồng và vệ sinh tàn dư sau thu hoạch.'
    ],
    'Tomato___Late_blight': [
        'Sử dụng thuốc trừ nấm như metalaxyl hoặc chlorothalonil ngay khi phát hiện bệnh.',
        'Loại bỏ và tiêu hủy cây bị nhiễm nặng.',
        'Tránh tưới nước lên lá, tưới vào gốc cây.',
        'Theo dõi thời tiết ẩm ướt để phun thuốc phòng ngừa.'
    ],
    'Tomato___Leaf_Mold': [
        'Phun thuốc trừ nấm như chlorothalonil hoặc copper-based.',
        'Tăng thông thoáng bằng cách tỉa cành và giàn cây.',
        'Tưới nước vào gốc và tránh làm ướt lá.',
        'Loại bỏ lá bị nhiễm và vệ sinh vườn.'
    ],
    'Tomato___Septoria_leaf_spot': [
        'Phun thuốc trừ nấm như mancozeb hoặc chlorothalonil.',
        'Loại bỏ lá bị nhiễm ở phần dưới cây.',
        'Tưới nước vào gốc và đảm bảo cây thông thoáng.',
        'Luân canh cây trồng và vệ sinh tàn dư sau thu hoạch.'
    ],
    'Tomato___Spider_mites Two-spotted_spider_mite': [
        'Sử dụng thuốc trừ nhện như abamectin hoặc neem oil.',
        'Tăng độ ẩm bằng cách tưới nước nhẹ lên lá vào sáng sớm.',
        'Loại bỏ lá bị nhiễm nặng và vệ sinh vườn.',
        'Kiểm tra thường xuyên để phát hiện sớm nhện đỏ.'
    ],
    'Tomato___Target_Spot': [
        'Phun thuốc trừ nấm như chlorothalonil hoặc azoxystrobin.',
        'Loại bỏ lá bị nhiễm và vệ sinh tàn dư cây trồng.',
        'Tỉa cành để tăng lưu thông không khí.',
        'Luân canh cây trồng để giảm mầm bệnh.'
    ],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        'Kiểm soát côn trùng truyền bệnh như bọ phấn trắng bằng thuốc trừ sâu.',
        'Loại bỏ và tiêu hủy cây bị nhiễm nặng.',
        'Sử dụng lưới chắn côn trùng để bảo vệ cây non.',
        'Trồng giống cà chua kháng virus nếu có.'
    ],
    'Tomato___Tomato_mosaic_virus': [
        'Loại bỏ và tiêu hủy cây bị nhiễm để tránh lây lan.',
        'Vệ sinh dụng cụ làm vườn bằng dung dịch khử trùng.',
        'Tránh chạm vào cây khỏe sau khi tiếp xúc với cây bệnh.',
        'Trồng giống cà chua kháng virus như Better Boy.'
    ],
    'Tomato___healthy': ['Tiếp tục chăm sóc tốt, tưới nước đều, bón phân cân đối và kiểm tra sâu bệnh.'],
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
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
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
    model_path = './AI/plant-disease-model-complete.pth'
    
    print(f"Image path: {image_path}", file=sys.stderr)
    print(f"Model path: {model_path}", file=sys.stderr)
    
    try:
        result = predict_image(image_path, model_path)
        print(json.dumps(result))  # Output JSON to be parsed by Node.js
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)