# Li-Ion Battery State of Charge (SOC) Estimation using Machine Learning

## Overview
This project presents an IoT-enabled battery monitoring system that estimates the State of Charge (SOC) of a lithium-ion battery using machine learning models. Real-time data is collected using an ESP32-based hardware setup and transmitted to a server where trained ML models predict SOC.

The system combines embedded sensing, wireless communication, and data-driven modeling to achieve accurate and scalable battery monitoring.

---

## Features
- Real-time voltage, current, and temperature monitoring
- ESP32-based wireless data acquisition
- Machine Learning models for SOC prediction:
  - Linear Regression
  - Support Vector Regression (SVR)
  - XGBoost Regression
  - Multi-Layer Perceptron (MLP)
- Server-side SOC prediction using FastAPI
- Coulomb counting for validation
- Excel logging for performance analysis

---

## Machine Learning Approach
- Supervised learning using battery datasets
- Feature set:
  - Voltage
  - Current
  - Temperature
  - Derived features (power, delta Ah, etc.)
- Comparative study of multiple regression models
- MLP selected for deployment due to:
  - High accuracy
  - Stable performance
  - Real-time suitability

---

## Hardware Components
- ESP32 Microcontroller
- Voltage Sensor Module (0–25V divider)
- ACS712 Current Sensor (±5A)
- DHT11 Temperature Sensor
- Li-Ion Battery (3.7V, 2200mAh)
- Load (DC motor / fan)

---

## System Workflow
1. Sensors measure battery parameters (V, I, T)
2. ESP32 sends data via Wi-Fi (HTTP POST)
3. Server receives and preprocesses data
4. ML model predicts SOC
5. Results are logged and displayed


---

### Upload ESP32 Code
- Open Arduino IDE
- Install required libraries:
  - WiFi.h
  - HTTPClient.h
  - DHT.h
- Upload code to ESP32

---

## Results
- MLP model achieved highest accuracy among all models
- Low MAE and RMSE with strong generalization
- Real-time predictions closely match Coulomb counting

---

## Future Scope
- Hybrid ML models for improved accuracy
- State of Health (SOH) estimation
- Support for multiple battery types
- Compact hardware design with display module
- Online learning for adaptive prediction

---

## Applications
- Battery Management Systems (BMS)
- Electric Vehicles (EVs)
- Energy Storage Systems
- IoT-based monitoring solutions


## Note
This project is developed as part of a final-year engineering project focusing on integrating IoT and Machine Learning for practical battery monitoring.---
