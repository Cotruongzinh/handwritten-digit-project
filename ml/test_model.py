from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# 1. SET PATH
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = PROJECT_DIR / "web" / "model" / "digit_logistic_model.pkl"


# =====================================================
# 2. LOAD EXPORTED MODEL
# =====================================================
model_package = joblib.load(MODEL_PATH)
model = model_package["model"]

print("Frozen model loaded successfully!")
print("Classes:", model_package["classes"])


# =====================================================
# 3. LOAD AND CLEAN KAGGLE DATA
# =====================================================
df = pd.read_csv(DATA_PATH, low_memory=False)

# Xóa dòng thừa nếu file có dòng: label, 1x1, 1x2, ...
df = df[df["label"].astype(str).str.lower() != "label"].copy()

# Ép toàn bộ dữ liệu về dạng số
df = df.apply(pd.to_numeric, errors="coerce")

# Xóa các dòng còn bị lỗi nếu có
df = df.dropna()

# Đưa label về số nguyên
df["label"] = df["label"].astype(int)

print("Dataset shape after cleaning:", df.shape)
print(df.head())


# =====================================================
# 4. GET ONE SAMPLE
# =====================================================
true_label = df.loc[df.index[0], "label"]

sample = df.drop("label", axis=1).iloc[0].values.astype("float32")

# Chuẩn hóa pixel từ 0-255 về 0-1
sample = sample / 255.0

# Chuyển thành vector 1x784
sample_vector = sample.reshape(1, 784)


# =====================================================
# 5. PREDICT
# =====================================================
predicted_label = model.predict(sample_vector)[0]

print("True label:", true_label)
print("Predicted label:", predicted_label)


# =====================================================
# 6. SHOW IMAGE
# =====================================================
plt.imshow(sample.reshape(28, 28), cmap="gray")
plt.title(f"True: {true_label} | Predicted: {predicted_label}")
plt.axis("off")
plt.show()