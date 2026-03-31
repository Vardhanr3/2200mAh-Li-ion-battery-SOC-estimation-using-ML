import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------
# CREATE EXPORT FOLDER
# -------------------------------------------------
EXPORT_DIR = "export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
df = pd.read_csv("hybrid_training_dataset.csv")

# -------------------------------------------------
# CLEAN DATA
# -------------------------------------------------
df = df.dropna(subset=["soc"]).reset_index(drop=True)

print("📂 Dataset Loaded")
print("Columns:", list(df.columns))
print("Samples:", len(df))

# -------------------------------------------------
# SELECT FEATURES
# -------------------------------------------------
FEATURES = [
    "voltage",
    "temperature",
]

TARGET = "soc"

# Validate columns
for col in FEATURES + [TARGET]:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

X = df[FEATURES]
y = df[TARGET]

print("\n✅ Features used:", FEATURES)

# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
print("\n🚀 Training XGBoost...")

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    reg_alpha=0.2,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Performance:")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.4f}")

# -------------------------------------------------
# SAVE FILES TO EXPORT/
# -------------------------------------------------
model_path = os.path.join(EXPORT_DIR, "soc_xgboost_model.pkl")
features_path = os.path.join(EXPORT_DIR, "feature_order.pkl")
metrics_path = os.path.join(EXPORT_DIR, "metrics.txt")

joblib.dump(model, model_path)
joblib.dump(FEATURES, features_path)

# Save metrics (optional but useful)
with open(metrics_path, "w") as f:
    f.write(f"MAE  : {mae:.4f}\n")
    f.write(f"RMSE : {rmse:.4f}\n")
    f.write(f"R2   : {r2:.4f}\n")

print("\n💾 Files saved in /export:")
print(" - soc_xgboost_model.pkl")
print(" - feature_order.pkl")
print(" - metrics.txt")

print("\n✅ Training Complete")
