#include <WiFi.h>
#include <HTTPClient.h>
#include "DHT.h"

// ================= WiFi =================
const char* ssid = "esp32network";
const char* password = "12345678";
const char* serverURL = "http://10.42.0.1:3000/data";

// ================= DHT11 =================
#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// ================= ADC Pins ==============
const int voltagePin = 34;
const int currentPin = 35;

// ================= ADC Configuration =====
const float ADC_REF = 3.3;
const int ADC_RES = 4095;

// ================= Voltage Sensor =========
const float voltageSensorScaling = 5.01;

// ================= ACS712-05B =============
const float sensitivity = 0.185;
float zeroCurrent = 1.65;


// =====================================================
// Zero-current calibration
// =====================================================
void calibrateZeroCurrent() {
  Serial.println("Calibrating current sensor...");
  float sum = 0;
  for (int i = 0; i < 800; i++) {
    sum += analogRead(currentPin) * (ADC_REF / ADC_RES);
    delay(2);
  }
  zeroCurrent = sum / 800.0;

  Serial.print("Zero current voltage = ");
  Serial.println(zeroCurrent, 4);
}


// =====================================================
// Read current
// =====================================================
float readCurrent() {
  float sum = 0;
  for (int i = 0; i < 300; i++) {
    sum += analogRead(currentPin) * (ADC_REF / ADC_RES);
    delayMicroseconds(150);
  }
  float avgV = sum / 300.0;
  float current = (avgV - zeroCurrent) / sensitivity;

  if (abs(current) < 0.03) current = 0;
  return current;
}


// =====================================================
// WiFi Connect (STABLE VERSION)
// =====================================================
void connectWiFi() {
  Serial.println("\nStarting WiFi connection...");

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(1000);

  Serial.print("Connecting to SSID: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);   // improves stability

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}


// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  dht.begin();

  analogReadResolution(12);
  analogSetPinAttenuation(voltagePin, ADC_11db);
  analogSetPinAttenuation(currentPin, ADC_11db);

  connectWiFi();

  calibrateZeroCurrent();

  Serial.println("Sensors calibrated. System ready.\n");
}


// =====================================================
// LOOP
// =====================================================
void loop() {

  // If WiFi dropped → reconnect
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost. Reconnecting...");
    connectWiFi();
  }

  // -------- Voltage --------
  float adcV = analogRead(voltagePin) * (ADC_REF / ADC_RES);
  float batteryVoltage = adcV * voltageSensorScaling;

  // -------- Current --------
  float current = readCurrent();

  // -------- Temperature --------
  float temperature = dht.readTemperature();

  // -------- Serial --------
  Serial.println("------ Battery Monitoring ------");
  Serial.print("Battery Voltage (V): "); Serial.println(batteryVoltage, 3);
  Serial.print("Battery Current (A): "); Serial.println(current, 3);
  Serial.print("Temperature (°C): "); Serial.println(temperature, 2);
  Serial.println("--------------------------------\n");

  // -------- JSON --------
  String payload = "{";
  payload += "\"voltage\":" + String(batteryVoltage, 3) + ",";
  payload += "\"current\":" + String(current, 3) + ",";
  payload += "\"temperature\":" + String(temperature, 2);
  payload += "}";

  // -------- POST --------
  HTTPClient http;
  http.begin(serverURL);
  http.addHeader("Content-Type", "application/json");

  int code = http.POST(payload);

  Serial.print("HTTP Response: ");
  Serial.println(code);

  http.end();

  delay(1000);
}
