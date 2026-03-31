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
cols = ["voltage", "temperature"]
X = data[cols].values
y = data["soc"].values.reshape(-1, 1) / 100.0

# -----------------------------
# 4️⃣ Scaling
# -----------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# -----------------------------
# 5️⃣ Model Architecture
# -----------------------------
model = Sequential([
    Input(shape=(2,)),
    Dense(32, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
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
    batch_size=8,
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
# 8️⃣ Save Model & Scalers
# -----------------------------
os.makedirs("exports", exist_ok=True)

model.save("exports/soc_model.keras")

joblib.dump(scaler_X, "exports/scaler_X.pkl")
joblib.dump(scaler_y, "exports/scaler_y.pkl")

print("✅ Model & scalers saved")

# -----------------------------
# 9️⃣ Simple TFLite Export
# -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("exports/soc_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved")

# -----------------------------
# DONE
# -----------------------------
print("\n🎯 OUTPUT FILES:")
print(" - soc_model.keras")
print(" - soc_model.tflite")
print(" - scaler_X.pkl")
print(" - scaler_y.pkl")
