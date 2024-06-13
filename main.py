import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pandas as pd
import json
from google.cloud.sql.connector import Connector, IPTypes
import pg8000
import sqlalchemy

from flask import Flask, request, jsonify

# Load model
model = keras.models.load_model("model.h5")

# Load grading data
df_grading = pd.read_csv('Data Minuman.csv')

# Load labels from JSON
with open('class_indices.json', 'r') as f:
    labels = json.load(f)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((150, 150))
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_nutrifacts(drink_name):
    drink = df_grading[df_grading['Produk'] == drink_name]
    drink = drink[['Gula/Sajian(g)', 'Gula/100ml(g)', 'Grade']].iloc[0]
    return drink

def predict(img_tensor):
    predictions = model.predict(img_tensor)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = labels[str(predicted_class_index)]
    return predicted_class_label

app = Flask(__name__)

def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
    db_user = os.environ["DB_USER"]
    db_pass = os.environ["DB_PASS"]
    db_name = os.environ["DB_NAME"]

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
    connector = Connector()

    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )
    return pool

engine = connect_with_connector()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            tensor = preprocess_image(image_bytes)
            prediction = predict(tensor)        
            if prediction != "random images": 
                nutrifacts = get_nutrifacts(prediction)
                data = {
                    "Gula/Sajian(g)": str(nutrifacts["Gula/Sajian(g)"]),
                    "Gula/100ml(g)": str(nutrifacts["Gula/100ml(g)"]),
                    "Grade": nutrifacts["Grade"],
                    "Product": prediction
                }
                with engine.connect() as connection:
                    connection.execute(
                        "INSERT INTO produks (produk, gula, takaran, grade) VALUES (%s, %s, %s, %s)",
                        (prediction, nutrifacts["Gula/Sajian(g)"], nutrifacts["Gula/100ml(g)"], nutrifacts["Grade"])
                    )
            else:
                data = {"Product": "Product tidak ditemukan"}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"

if __name__ == "__main__":
    app.run(debug=True)