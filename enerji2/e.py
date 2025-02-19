import streamlit as st
import pandas as pd
import pickle
import requests
import json
import joblib
import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

def get_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        weather_data = response.json()
        temp = weather_data["main"]["temp"]
        pressure = weather_data["main"]["pressure"]
        humidity = weather_data["main"]["humidity"]
        wind_speed = weather_data["wind"]["speed"]
        wind_direction = weather_data["wind"]["deg"]
        sunshine = 0  # Varsayılan değer
        
        return {
            "Temperature": temp,
            "AirPressure": pressure,
            "Humidity": humidity,
            "WindSpeed": wind_speed,
            "WindDirection": wind_direction,
            "Sunshine": sunshine
        }
    else:
        return None

def predict_solar(api_key, city, model_path="model2.pkl", scaler_path="scaler2.pkl"):
    weather_data = get_weather_data(api_key, city)
    if weather_data is None:
        return "Hava durumu verisi alınamadı!"
    
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    
    trained_columns = ["WindSpeed", "Sunshine", "AirPressure", "AirTemperature", "RelativeAirHumidity", "Year", "Month", "Day", "Hour", "Minute", "Sunshine_Estimate"]
    
    new_data = {
        "WindSpeed": weather_data["WindSpeed"],
        "Sunshine": weather_data["Sunshine"],
        "AirPressure": weather_data["AirPressure"],
        "AirTemperature": weather_data["Temperature"],
        "RelativeAirHumidity": weather_data["Humidity"],
        "Year": 2025,
        "Month": 2,
        "Day": 19,
        "Hour": 12,
        "Minute": 30
    }
    
    new_data_df = pd.DataFrame([new_data])
    new_data_df["Sunshine_Estimate"] = new_data_df["Sunshine"] * (new_data_df["AirPressure"] / 1013.25)
    
    for col in trained_columns:
        if col not in new_data_df.columns:
            new_data_df[col] = 0
    
    new_data_df = new_data_df[trained_columns]
    new_data_scaled = scaler.transform(new_data_df)
    prediction = model.predict(new_data_scaled)
    
    return prediction[0]

def predict_wind(api_key, city, model_path="xgboost_model.json", scaler_path="scaler.pkl"):
    weather_data = get_weather_data(api_key, city)
    if weather_data is None:
        return "Hava durumu verisi alınamadı!"
    
    with open("wind_features.json", "r") as f:
        feature_names = json.load(f)
    
    input_data = {
        "temperature_2m": weather_data["Temperature"],
        "relativehumidity_2m": weather_data["Humidity"],
        "windspeed_10m": weather_data["WindSpeed"],
        "winddirection_10m": weather_data["WindDirection"],
        "Year": 2025,
        "Month": 2,
        "Day": 19,
        "Hour": 12
    }
    
    for feature in feature_names:
        if feature not in input_data:
            input_data[feature] = 0
    
    input_array = np.array([[input_data[feature] for feature in feature_names]])
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    input_scaled = scaler.transform(input_array)
    predicted_power = model.predict(input_scaled)[0]
    
    return predicted_power

st.title("Enerji Üretim Tahmini")
city = st.selectbox("Bir şehir seçin:", ["Ankara", "İstanbul", "İzmir", "Bursa", "Antalya", "Atina", "Patras", "Heraklion", "Larisa", "Adana", "Konya", "Gaziantep", "Mersin", "Eskişehir", "Samsun", "Trabzon", "Kayseri", "Denizli", "Kocaeli", "Volos", "Chania", "Rhodos", "Kozani", "Ioannina", "Kalamata", "Kavala", "Alexandroupoli", "Chalkida", "Patmos"])
energy_type = st.radio("Tahmin Türü Seçin:", ["Güneş", "Rüzgar"])

if st.button("Hesapla"):
    api_key = "601c0bdb3709bf5472c7a721ef457c42"
    
    if energy_type == "Güneş":
        prediction = predict_solar(api_key, city)
    else:
        prediction = predict_wind(api_key, city)
    if isinstance(prediction, str) and "Hava durumu verisi alınamadı!" in prediction:
     st.error(prediction)  # Hata mesajını göster
    else:
    # Prediction değerini float'a dönüştür ve göster
     prediction_value = float(prediction)
     st.success(f"Tahmin Edilen Üretim: {prediction_value:.2f} MW")
