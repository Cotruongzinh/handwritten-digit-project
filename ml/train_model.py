from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =====================================================
# 1. SET PATH
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "data" / "train.csv"

# Theo Lecture 06, file .pkl nên nằm trong Flask root workspace
MODEL_DIR = PROJECT_DIR / "web" / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "digit_logistic_model.pkl"


# =====================================================
# 2. CONFIG
# =====================================================
USE_AUGMENTATION = True

# Nếu máy yếu, để 10000 như hiện tại.
# Nếu sau này có dataset lớn hơn và muốn train nhanh, có thể giảm LIMIT_ROWS.
LIMIT_ROWS = None


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================
def clean_kaggle_dataframe(df):
    """
    File CSV của bạn có 1 dòng thừa dạng:
    label, 1x1, 1x2, ..., 28x28

    Hàm này xóa dòng thừa và ép dữ liệu về số.
    """
    df = df[df["label"].astype(str).str.lower() != "label"].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df["label"] = df["label"].astype(int)
    return df


def center_by_mass(img):
    """
    Căn giữa ảnh 28x28 theo trọng tâm pixel sáng.
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

    if shift_y > 0:
        shifted[:shift_y, :] = 0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0

    if shift_x > 0:
        shifted[:, :shift_x] = 0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0

    return shifted


def normalize_digit_vector(vector, long_side=20):
    """
    Chuẩn hóa ảnh Kaggle 28x28 theo cùng logic với ảnh canvas:
    - Lọc nền
    - Crop chữ số
    - Resize giữ tỉ lệ
    - Đặt vào ảnh 28x28
    - Căn giữa theo trọng tâm
    - Normalize 0-1
    """
    img = vector.reshape(28, 28).astype("float32")

    # Lọc nền
    img[img < 30] = 0

    coords = np.column_stack(np.where(img > 0))

    if coords.size == 0:
        return np.zeros(784, dtype="float32")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit = img[y_min:y_max + 1, x_min:x_max + 1]
    h, w = digit.shape

    if h > w:
        new_h = long_side
        new_w = max(1, int(round(w * long_side / h)))
    else:
        new_w = long_side
        new_h = max(1, int(round(h * long_side / w)))

    digit_image = Image.fromarray(digit.astype("uint8"))
    digit_image = digit_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    digit_array = np.array(digit_image).astype("float32")

    final_image = np.zeros((28, 28), dtype="float32")

    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    final_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_array
    final_image = center_by_mass(final_image)

    return (final_image / 255.0).reshape(784).astype("float32")


def shift_image(vector, dx, dy):
    """
    Tạo phiên bản chữ số bị lệch nhẹ trái/phải/lên/xuống.
    Đây là data augmentation để model hiểu tốt hơn ảnh vẽ trên web.
    """
    img = vector.reshape(28, 28)
    shifted = np.zeros_like(img)

    if dy >= 0:
        src_y1, src_y2 = 0, 28 - dy
        dst_y1, dst_y2 = dy, 28
    else:
        src_y1, src_y2 = -dy, 28
        dst_y1, dst_y2 = 0, 28 + dy

    if dx >= 0:
        src_x1, src_x2 = 0, 28 - dx
        dst_x1, dst_x2 = dx, 28
    else:
        src_x1, src_x2 = -dx, 28
        dst_x1, dst_x2 = 0, 28 + dx

    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    return shifted.reshape(784).astype("float32")


def augment_dataset(X, y):
    """
    Tăng dữ liệu bằng cách dịch ảnh nhẹ.
    Giữ Logistic Regression nhưng giúp model bớt nhạy với vị trí chữ số.
    """
    shifts = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]

    X_aug = []
    y_aug = []

    for img, label in zip(X, y):
        for dx, dy in shifts:
            X_aug.append(shift_image(img, dx, dy))
            y_aug.append(label)

    return np.array(X_aug, dtype="float32"), np.array(y_aug)


# =====================================================
# 4. LOAD KAGGLE DATASET
# =====================================================
print("Loading Kaggle Digit Recognizer dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)
df = clean_kaggle_dataframe(df)

if LIMIT_ROWS is not None:
    df = df.head(LIMIT_ROWS)

print("Dataset shape after cleaning:", df.shape)
print(df.head())


# =====================================================
# 5. SPLIT FEATURES X AND TARGET y
# =====================================================
y = df["label"].values
X_raw = df.drop("label", axis=1).values.astype("float32")

print("Raw X shape:", X_raw.shape)
print("y shape:", y.shape)

if X_raw.shape[1] != 784:
    raise ValueError(f"Expected 784 pixel columns, but got {X_raw.shape[1]}")


# =====================================================
# 6. NORMALIZE IMAGES LIKE WEB INPUT
# =====================================================
print("Normalizing images with the same preprocessing style as the web canvas...")

X = np.array([normalize_digit_vector(row, long_side=20) for row in X_raw], dtype="float32")

print("Normalized X shape:", X.shape)


# =====================================================
# 7. AUGMENT DATASET
# =====================================================
if USE_AUGMENTATION:
    print("Creating augmented dataset...")
    X, y = augment_dataset(X, y)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)


# =====================================================
# 8. SHOW SAMPLE IMAGE
# =====================================================
sample_image = X[0].reshape(28, 28)

plt.imshow(sample_image, cmap="gray")
plt.title(f"Sample label: {y[0]}")
plt.axis("off")
plt.show()


# =====================================================
# 9. TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =====================================================
# 10. TRAIN LOGISTIC REGRESSION
# =====================================================
print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=500,
    solver="lbfgs",
    C=1.0,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# =====================================================
# 11. EVALUATE MODEL
# =====================================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =====================================================
# 12. EXPORT / FREEZE MODEL
# =====================================================
# Đây là phần quan trọng theo yêu cầu của cô:
# Train xong thì export model ra file vật lý .pkl.
# Web sẽ load file này để predict, không train lại.

model_package = {
    "model": model,
    "image_size": 28,
    "input_features": 784,
    "normalize": True,
    "invert_color": True,
    "classes": [int(c) for c in model.classes_],
    "preprocessing": "crop_resize_center_by_mass_multiscale",
    "augmentation": USE_AUGMENTATION
}

joblib.dump(model_package, MODEL_PATH)

print("Export model successfully!")
print("Frozen AI model saved to:", MODEL_PATH)
