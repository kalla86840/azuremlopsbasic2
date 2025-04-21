import os
import joblib
import json
import numpy as np

def init():
    global model
    try:
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "linear_model.pkl")
        print("🔍 Loading model from:", model_path)
        model = joblib.load(model_path)
        print("✅ Model loaded.")
    except Exception as e:
        print("❌ Failed to load model:", str(e))
        raise

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        input_array = np.array(data)
        predictions = model.predict(input_array)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
