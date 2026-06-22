from pathlib import Path
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import joblib
import base64
import io


# =====================================================
# 1. PATH SETUP
# =====================================================
WEB_DIR = Path(__file__).resolve().parent
MODEL_PATH = WEB_DIR / "model" / "digit_logistic_model.pkl"


# =====================================================
# 2. CREATE FLASK APP
# =====================================================
app = Flask(__name__)


# =====================================================
# 3. LOAD EXPORTED MODEL - GLOBAL SCOPE
# =====================================================
# Load model một lần khi Flask khởi động.
# Không load model bên trong route /predict.

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
image_size = model_package.get("image_size", 28)
input_features = model_package.get("input_features", 784)

print("Model loaded successfully from:", MODEL_PATH)


# =====================================================
# 4. IMAGE PREPROCESSING HELPERS
# =====================================================
def center_by_mass(img):
    """
    Căn giữa ảnh 28x28 theo trọng tâm pixel sáng.
    Input: nền đen, nét trắng.
    """
    y_indices, x_indices = np.indices(img.shape)
    total = img.sum()

    if total <= 0:
        return img

    center_y = (y_indices * img).sum() / total
    center_x = (x_indices * img).sum() / total

    shift_y = int(round(14 - center_y))
    shift_x = int(round(14 - center_x))

    shifted = np.roll(img, shift_y, axis=0)
    shifted = np.roll(shifted, shift_x, axis=1)

    # Xóa vùng bị cuộn vòng
    if shift_y > 0:
        shifted[:shift_y, :] = 0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0

    if shift_x > 0:
        shifted[:, :shift_x] = 0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0

    return shifted


def image_to_ink_array(image):
    """
    Chuyển ảnh bất kỳ về dạng:
    - nền đen
    - nét chữ số sáng/trắng
    - NumPy array uint8

    Hàm này dùng được cho:
    - ảnh canvas: nền trắng, nét đen
    - ảnh upload: thường cũng là nền sáng, nét tối
    - nếu ảnh nền tối, nét sáng thì sẽ tự giữ hướng phù hợp hơn
    """
    image = image.convert("L")
    arr = np.array(image).astype("uint8")

    # Ước lượng nền bằng median.
    # Nếu ảnh sáng nhiều hơn tối => nền trắng, nét đen => đảo màu.
    # Nếu ảnh tối nhiều hơn sáng => nền đen, nét sáng => giữ nguyên.
    if np.median(arr) > 127:
        ink = 255 - arr
    else:
        ink = arr.copy()

    # Lọc nhiễu nền.
    # Ngưỡng 35 ổn cho canvas; vẫn đủ mềm cho ảnh upload đơn giản.
    ink[ink < 35] = 0

    return ink


def crop_all_ink(ink):
    """
    Crop toàn bộ vùng có nét chữ số trong ảnh.
    """
    coords = np.column_stack(np.where(ink > 0))

    if coords.size == 0:
        return None, None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = ink[y_min:y_max + 1, x_min:x_max + 1]

    return cropped, (y_min, x_min, y_max, x_max)


def find_digit_regions(cropped_ink):
    """
    Tách nhiều chữ số theo trục ngang bằng vertical projection.

    Lưu ý chuyên môn:
    - Cách này phù hợp khi các chữ số có khoảng cách rõ ràng.
    - Nếu các chữ số dính liền nhau, Logistic Regression + segmentation đơn giản
      sẽ khó nhận diện chính xác.
    """
    h, w = cropped_ink.shape

    # Đếm số pixel sáng theo từng cột
    projection = (cropped_ink > 0).sum(axis=0)

    # Cột được xem là có nét nếu có đủ pixel sáng
    active_threshold = max(1, int(h * 0.01))
    active = projection > active_threshold

    raw_regions = []
    start = None

    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            raw_regions.append([start, i - 1])
            start = None

    if start is not None:
        raw_regions.append([start, w - 1])

    if not raw_regions:
        return []

    # Gộp các vùng quá gần nhau để tránh tách nhầm 1 chữ số thành nhiều phần
    merge_gap = max(4, int(w * 0.025))
    merged = [raw_regions[0]]

    for region in raw_regions[1:]:
        previous = merged[-1]
        gap = region[0] - previous[1] - 1

        if gap <= merge_gap:
            previous[1] = region[1]
        else:
            merged.append(region)

    # Lọc nhiễu nhỏ
    final_regions = []
    min_width = max(3, int(w * 0.01))

    for x1, x2 in merged:
        if (x2 - x1 + 1) >= min_width:
            final_regions.append((x1, x2))

    return final_regions


def extract_digit_arrays(image):
    """
    Tách ảnh thành danh sách các chữ số riêng lẻ.
    Output: list các ảnh digit dạng ink array, thứ tự trái sang phải.
    """
    ink = image_to_ink_array(image)
    cropped, bbox = crop_all_ink(ink)

    if cropped is None:
        return []

    h, w = cropped.shape
    x_regions = find_digit_regions(cropped)

    # Nếu không tách được region, xem như 1 chữ số
    if not x_regions:
        return [cropped]

    digit_arrays = []

    for x1, x2 in x_regions:
        region = cropped[:, x1:x2 + 1]

        # Lấy lại y theo nét thật của từng vùng
        coords = np.column_stack(np.where(region > 0))

        if coords.size == 0:
            continue

        y_min, x_min_local = coords.min(axis=0)
        y_max, x_max_local = coords.max(axis=0)

        digit = region[y_min:y_max + 1, x_min_local:x_max_local + 1]

        # Padding nhẹ để không cắt sát nét
        dh, dw = digit.shape
        pad = max(2, int(max(dh, dw) * 0.08))
        digit = np.pad(digit, pad, mode="constant", constant_values=0)

        # Bỏ vùng quá nhỏ vì có thể là nhiễu
        if digit.shape[0] * digit.shape[1] < 25:
            continue

        digit_arrays.append(digit)

    return digit_arrays


def preprocess_digit_array(digit, long_side=20):
    """
    Tiền xử lý 1 chữ số đã được tách riêng về vector 1x784.
    """
    h, w = digit.shape

    if h <= 0 or w <= 0:
        return np.zeros((1, input_features), dtype="float32")

    # Resize giữ tỉ lệ, cạnh dài nhất về long_side
    if h > w:
        new_h = long_side
        new_w = max(1, int(round(w * long_side / h)))
    else:
        new_w = long_side
        new_h = max(1, int(round(h * long_side / w)))

    digit_image = Image.fromarray(digit.astype("uint8"))
    digit_image = digit_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    digit_array = np.array(digit_image).astype("float32")

    # Đưa vào giữa ảnh 28x28
    final_image = np.zeros((28, 28), dtype="float32")

    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    final_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_array

    # Căn giữa theo trọng tâm pixel
    final_image = center_by_mass(final_image)

    # Normalize 0-1
    final_image = final_image / 255.0

    return final_image.reshape(1, input_features).astype("float32")


def predict_one_digit(digit):
    """
    Dự đoán 1 chữ số bằng nhiều kích thước tiền xử lý rồi lấy trung bình xác suất.
    """
    vectors = np.vstack([
        preprocess_digit_array(digit, long_side=18),
        preprocess_digit_array(digit, long_side=20),
        preprocess_digit_array(digit, long_side=22),
    ])

    probabilities = model.predict_proba(vectors)
    avg_probabilities = probabilities.mean(axis=0)

    class_index = int(np.argmax(avg_probabilities))
    prediction = int(model.classes_[class_index])
    confidence = float(avg_probabilities[class_index] * 100)

    return prediction, confidence


def predict_number_from_image(image):
    """
    Dự đoán một ảnh có thể chứa:
    - 1 chữ số
    - nhiều chữ số, ví dụ 12, 305, 2026

    Kết quả được ghép theo thứ tự trái sang phải.
    """
    digit_arrays = extract_digit_arrays(image)

    if not digit_arrays:
        return {
            "number": "",
            "digits": [],
            "average_confidence": 0.0
        }

    digit_results = []

    for digit_array in digit_arrays:
        digit, confidence = predict_one_digit(digit_array)
        digit_results.append({
            "digit": digit,
            "confidence": confidence
        })

    number = "".join(str(item["digit"]) for item in digit_results)
    average_confidence = sum(item["confidence"] for item in digit_results) / len(digit_results)

    return {
        "number": number,
        "digits": digit_results,
        "average_confidence": average_confidence
    }


# =====================================================
# 5. HOME PAGE
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# 6. PREDICT API
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None or "image" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing image data"
            }), 400

        image_data = data["image"]

        # Bỏ phần đầu: data:image/png;base64,...
        image_data = image_data.split(",")[1]

        # Decode base64 thành ảnh
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Dự đoán số, có thể là 1 chữ số hoặc nhiều chữ số
        result = predict_number_from_image(image)

        if result["number"] == "":
            return jsonify({
                "status": "error",
                "message": "No handwritten digit detected"
            }), 400

        return jsonify({
            "status": "success",
            "number": result["number"],
            "digit": result["number"],
            "digits": [
                {
                    "digit": item["digit"],
                    "confidence": f"{item['confidence']:.2f}%"
                }
                for item in result["digits"]
            ],
            "confidence": f"{result['average_confidence']:.2f}%",
            "digit_count": len(result["digits"])
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =====================================================
# 7. RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
