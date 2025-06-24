from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import io

app = FastAPI()
model = tf.keras.models.load_model("model/smartbite.h5")
class_labels = ["Alpukat", "Ayam Goreng", "Bakso", "Bawang Bombai", "Bawang Merah", 
                "Bawang Putih", "Cabe Rawit", "Mangga", "Nasi Putih", "Pisang", 
                "Sate Ayam", "Tahu Goreng", "Telur Mata Sapi", "Tempe Goreng", "rendang"]

# Load nutrisi
df_nutrition = pd.read_csv("nutrition.csv")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    pred_class = np.argmax(prediction, axis=1)[0]
    food_name = class_labels[pred_class]

    nutri = df_nutrition[df_nutrition['name'].str.lower() == food_name.lower()]
    if not nutri.empty:
        nutrition_info = nutri.iloc[0].to_dict()
        return {"food": food_name, "nutrition": nutrition_info}
    else:
        return {"food": food_name, "nutrition": "not found"}
