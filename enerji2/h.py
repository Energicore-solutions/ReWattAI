import pandas as pd
import numpy as np
import xgboost as xgb
import json
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 📌 1. TRAIN AND SAVE THE MODEL
def train_and_save_model(data_path="Location1.csv", model_path="xgboost_model.json", scaler_path="scaler.pkl"):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    # 📌 CONVERT DATE TO NUMERIC VALUES
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Year"] = df["Time"].dt.year
    df["Month"] = df["Time"].dt.month
    df["Day"] = df["Time"].dt.day
    df["Hour"] = df["Time"].dt.hour
    df.drop(columns=["Time"], inplace=True)

    # 📌 REMOVE NON-NUMERIC COLUMNS
    for col in df.columns:
        if df[col].dtype == "object":
            df.drop(columns=[col], inplace=True)

    # 📌 DEFINE FEATURES
    X = df.drop(columns=["Power"])
    y = df["Power"]

    # 📌 SAVE FEATURE NAMES
    feature_names = X.columns.tolist()
    with open("features.json", "w") as f:
        json.dump(feature_names, f)

    # 📌 SPLIT THE DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 📌 SCALE THE DATA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 📌 XGBoost Model
    model = xgb.XGBRegressor(
        colsample_bytree=0.99,
        learning_rate=0.24,
        max_depth=9,
        n_estimators=550,
        subsample=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    # 📌 Train the Model
    model.fit(X_train_scaled, y_train)

    # 📌 Save the Model
    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)

    # 📌 Model Performance
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"🔥 Model Saved! - MSE: {mse:.4f}, R²: {r2:.4f}")
    # Modeli yeniden kaydedin
    model.save_model("xgboost_model.json")


# 📌 2. GET DATA FROM OPENWEATHER API & MAKE PREDICTIONS
def predict_power(api_key, city, model_path="xgboost_model.json", scaler_path="scaler.pkl"):
    # 📌 Get data from OpenWeather API
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        print("❌ Error: Unable to fetch API data.")
        return None

    # 📌 Read feature names
    with open("features.json", "r") as f:
        feature_names = json.load(f)

    # 📌 OpenWeather API data
    weather_data = {
        "temperature_2m": data["main"]["temp"],
        "relativehumidity_2m": data["main"]["humidity"],
        "windspeed_10m": data["wind"]["speed"],
        "winddirection_10m": data["wind"]["deg"],
    }

    # 📌 Time data
    timestamp = pd.Timestamp.utcnow()
    weather_data["Year"] = timestamp.year
    weather_data["Month"] = timestamp.month
    weather_data["Day"] = timestamp.day
    weather_data["Hour"] = timestamp.hour

    # 📌 Fill in missing columns
    for feature in feature_names:
        if feature not in weather_data:
            weather_data[feature] = 0  # Default value (can be adjusted)

    # 📌 Prepare data for the model
    input_data = np.array([[weather_data[feature] for feature in feature_names]])

    # 📌 Load the model and scaler
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # 📌 Scale the data
    input_scaled = scaler.transform(input_data)

    # 📌 Make prediction
    predicted_power = model.predict(input_scaled)[0]

    print(f"⚡ Predicted Power Output: {predicted_power:.2f} MW")
    return predicted_power


# 📌 3. EXAMPLE USAGE
if __name__ == "__main__":
    # 📌 Train and save the model
    train_and_save_model()

    # 📌 Make prediction with OpenWeather API
    API_KEY = "601c0bdb3709bf5472c7a721ef457c42"
    CITY = "Bursa"  # Change the city
    predict_power(API_KEY, CITY)
