import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# ================= LOAD DATA =================
df = pd.read_csv("hybrid_training_dataset.csv")

# ================= SELECT FEATURES =================
# DO NOT use current for training
features = ["voltage", "temperature"]   # modify if needed
target = "soc"

X = df[features].values
y = df[target].values

# ================= NORMALIZATION =================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# ================= CREATE SEQUENCES =================
def create_sequences(X, y, time_steps=50):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 50
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.3, shuffle=False
)

# ================= BUILD MODEL =================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, len(features))),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ================= TRAIN =================
model.fit(X_train, y_train, epochs=40, batch_size=10, validation_data=(X_test, y_test))

# ================= SAVE MODEL =================
model.save("lstm_soc_model.h5")

print("Model training complete and saved!")
