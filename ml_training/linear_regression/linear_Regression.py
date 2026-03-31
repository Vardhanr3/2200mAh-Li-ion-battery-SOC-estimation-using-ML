# ================================
# Battery SOC - Linear Model (FINAL FIXED VERSION)
# ================================

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
data = pd.read_csv("hybrid_training_dataset.csv")

# -----------------------------
# 2️⃣ Data Cleaning
# -----------------------------
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
data["soc"] = data["soc"].clip(0, 100)

# -----------------------------
# 3️⃣ Feature Selection (NO CURRENT)
# -----------------------------
X = data[["voltage", "temperature"]].values
y = data["soc"].values.reshape(-1, 1) / 100.0

# -----------------------------
# 4️⃣ Scaling
# -----------------------------
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5️⃣ Linear Model
# -----------------------------
model = Sequential([
    Input(shape=(2,)),
    Dense(1, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"]
)

# -----------------------------
# 6️⃣ Training
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=300,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 7️⃣ Evaluation
# -----------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test MAE (SOC): {mae * 100:.2f} %")

# -----------------------------
# 8️⃣ Save Models & Scaler
# -----------------------------
os.makedirs("exports", exist_ok=True)

model.save("exports/soc_model.keras")
model.save("exports/soc_model.h5")

joblib.dump(scaler_X, "exports/scaler_X.pkl")

print("✅ Models & scaler saved")

# ==========================================================
# 9️⃣ TFLITE EXPORT (FINAL FIX - NO GRADIENT BUG)
# ==========================================================

# Save model WITHOUT gradients (critical fix)
tf.saved_model.save(
    model,
    "exports/saved_model",
    options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
)

# ---------- FLOAT32 ----------
converter = tf.lite.TFLiteConverter.from_saved_model("exports/saved_model")
tflite_float = converter.convert()

float_path = "exports/soc_model_float32.tflite"
with open(float_path, "wb") as f:
    f.write(tflite_float)

print("✅ FLOAT32 TFLite saved")

# ---------- FLOAT16 ----------
converter = tf.lite.TFLiteConverter.from_saved_model("exports/saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_float16 = converter.convert()

float16_path = "exports/soc_model_float16.tflite"
with open(float16_path, "wb") as f:
    f.write(tflite_float16)

print("✅ FLOAT16 TFLite saved")

# -----------------------------
# 🔍 Verify Models
# -----------------------------
def check_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    print(f"🔍 {os.path.basename(path)} OK")

check_tflite(float_path)
check_tflite(float16_path)

# -----------------------------
# 📦 Size Comparison
# -----------------------------
def size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

print("\n📦 Model Sizes:")
print(f"FLOAT32 : {size_mb(float_path):.4f} MB")
print(f"FLOAT16 : {size_mb(float16_path):.4f} MB")

# -----------------------------
# DONE
# -----------------------------
print("\n🎯 FINAL FILES GENERATED IN /exports")
