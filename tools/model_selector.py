import pandas as pd
import os
import joblib
import tensorflow as tf

def buy_model_selector():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_log = pd.read_csv(os.path.join(base_dir, "../models/models_log.csv"))
    
    buy_log = model_log[model_log["model"] == "buy_model"]
    buy_model_name = buy_log.loc[buy_log["r2"] == buy_log["r2"].max(), "model_name"].tolist()[0]

    if buy_model_name == "buy_model_ML.joblib":
        model = joblib.load(os.path.join(base_dir, "../models/buy_model_ML.joblib"))
    else:
        buy_model_name = "modelo_compras_dl.keras"
        model = tf.keras.models.load_model(os.path.join(base_dir, "../models/modelo_compras_dl.keras"))

    return model, buy_model_name

def rent_model_selector():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_log = pd.read_csv(os.path.join(base_dir, "../models/models_log.csv"))
    
    # Filtrar primero por rent_model, luego buscar el máximo r2 dentro de ese subconjunto
    rent_log = model_log[model_log["model"] == "rent_model"]
    rent_model_name = rent_log.loc[rent_log["r2"] == rent_log["r2"].max(), "model_name"].tolist()[0]

    if rent_model_name == "rent_model_ML.joblib":
        model = joblib.load(os.path.join(base_dir, "../models/rent_model_ML.joblib"))
    else:
        rent_model_name = "modelo_alquiler_dl.keras"
        model = tf.keras.models.load_model(os.path.join(base_dir, "../models/modelo_alquiler_dl.keras"))

    return model, rent_model_name