
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS

import cv2
import mediapipe as mp
import numpy as np
import requests 

app = Flask(__name__)

# Cấu hình CORS cho toàn bộ ứng dụng
CORS(app)


# Biến toàn cục để điều khiển việc bật/tắt hiệu ứng kính mát ảo
virtual_glasses_enabled = False

# Khởi tạo Mediapipe Face Detection và công cụ vẽ
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Tải hình ảnh kính mát với nền trong suốt
sunglass_image = None  # Khởi tạo biến toàn cục cho ảnh kính mát

def apply_virtual_glasses(image, detection):
    global virtual_glasses_enabled, sunglass_image
    if not virtual_glasses_enabled or sunglass_image is None:
        return image  # Nếu hiệu ứng chưa bật, trả về ảnh gốc

    # Lấy các tọa độ chính từ khuôn mặt phát hiện
    nose_tip = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
    left_ear = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
    right_ear = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)

    if left_ear and right_ear:
        if left_ear.x > right_ear.x:
            left_ear, right_ear = right_ear, left_ear

        ear_distance = right_ear.x - left_ear.x
        if ear_distance <= 0:
            return image

        # Tính toán kích thước kính mát
        sunglass_width = int(ear_distance * image.shape[1] * 1.3)
        sunglass_height = int(sunglass_width * (sunglass_image.shape[0] / sunglass_image.shape[1]))
        sunglasses_resized = cv2.resize(sunglass_image, (sunglass_width, sunglass_height))

        # Điều chỉnh độ trong suốt (opacity) của kính mát
        opacity_level = 0.7  # Độ trong suốt (0 = trong suốt hoàn toàn, 1 = không trong suốt)
        
        if sunglasses_resized.shape[2] == 4:  # Kênh alpha có tồn tại
            alpha_channel = sunglasses_resized[:, :, 3] / 255.0
        else:
            alpha_channel = np.ones((sunglasses_resized.shape[0], sunglasses_resized.shape[1]))

        alpha_channel = alpha_channel * opacity_level  # Điều chỉnh độ trong suốt

        eye_center_x = int(nose_tip.x * image.shape[1])
        eye_center_y = int(nose_tip.y * image.shape[0]) - 80

        x_offset = eye_center_x - (sunglass_width // 2)
        y_offset = eye_center_y - (sunglass_height // 2)

        if (0 <= y_offset < image.shape[0] and
            0 <= x_offset < image.shape[1] and
            y_offset + sunglasses_resized.shape[0] <= image.shape[0] and
            x_offset + sunglasses_resized.shape[1] <= image.shape[1]):

            # Áp dụng kính mát lên ảnh với độ trong suốt đã điều chỉnh
            for c in range(0, 3):
                image[y_offset:y_offset + sunglasses_resized.shape[0], x_offset:x_offset + sunglasses_resized.shape[1], c] = \
                    (sunglasses_resized[:, :, c] * alpha_channel) + \
                    (image[y_offset:y_offset + sunglasses_resized.shape[0], x_offset:x_offset + sunglasses_resized.shape[1], c] * (1.0 - alpha_channel))
    
    return image


def generate_frames():
    """Tạo các khung hình từ camera và áp dụng hiệu ứng nếu được bật."""
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            success, image = cap.read()
            if not success:
                break

            # Lật ảnh để phản chiếu
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            # Nếu phát hiện khuôn mặt, vẽ và áp dụng kính mát
            if results.detections:
                for detection in results.detections:
                    # Không vẽ khung bao quanh khuôn mặt
                    # mp_drawing.draw_detection(image, detection)
                    image = apply_virtual_glasses(image, detection)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Trang chính hiển thị video với hiệu ứng kính mát."""
    return render_template('index.html')

@app.route('/toggle_glasses', methods=['POST'])
def toggle_glasses():
    """API để bật/tắt hiệu ứng kính mát."""
    global virtual_glasses_enabled, sunglass_image 
    
    # Lấy dữ liệu từ yêu cầu POST 
    data = request.get_json()
    image_url = data.get('image_url')  # Lấy URL ảnh từ payload
    
    # Tải ảnh từ URL
    try:
        print(f"Attempting to load image from URL: {image_url}")  # Thêm log để kiểm tra URL
        response = requests.get(image_url)
        if response.status_code == 200:
            print("Image loaded successfully from URL")
            # Đọc ảnh từ phản hồi và chuyển đổi nó thành mảng numpy
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            sunglass_image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)  # Gán ảnh kính mát mới
            if sunglass_image is None:
                raise ValueError("Error: Không thể tải ảnh kính mát từ URL.")
        else:
            raise ValueError(f"Error: Không thể tải ảnh từ URL, mã trạng thái: {response.status_code}")
    except Exception as e:
        print(f"Exception while loading image: {e}")
        return jsonify({'error': str(e)}), 400

    # Đảm bảo rằng hiệu ứng kính mát luôn bật sau khi tải ảnh thành công
    if not virtual_glasses_enabled:
        virtual_glasses_enabled = True  # Bật kính mát nếu chưa bật
    
    # Bật/tắt kính mát
    # virtual_glasses_enabled = not virtual_glasses_enabled
    return jsonify({'glasses_enabled': virtual_glasses_enabled, 'image_url': image_url})

@app.route('/video_feed')
def video_feed():
    """API cung cấp luồng video từ camera."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

