from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import io
import os
import gdown

app = FastAPI()

# =========================
# Download model dari Google Drive (jika belum ada)
# =========================
MODEL_PATH = "model/smartbite.h5"
DRIVE_FILE_ID = "1-gHySlrAYDXkF5ssENqJSzsGOkN9ZC_j"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# =========================
# Load model dan data nutrisi
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = [
    "Alpukat", "Ayam Goreng", "Bakso", "Bawang Bombai", "Bawang Merah",
    "Bawang Putih", "Cabe Rawit", "Mangga", "Nasi Putih", "Pisang",
    "Sate Ayam", "Tahu Goreng", "Telur Mata Sapi", "Tempe Goreng", "rendang"
]

df_nutrition = pd.read_csv("nutrition.csv")

# =========================
# Endpoint utama
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    pred_class = np.argmax(prediction, axis=1)[0]
    food_name = class_labels[pred_class]

    nutri = df_nutrition[df_nutrition['name'].str.lower() == food_name.lower()]
    if not nutri.empty:
        nutrition_info = nutri.iloc[0].to_dict()
        return {
            "food": food_name,
            "nutrition": nutrition_info
        }
    else:
        return {
            "food": food_name,
            "nutrition": "not found"
        }
