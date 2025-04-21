import os
import joblib
import json
import numpy as np

def init():
    global model
    try:
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "linear_model.pkl")
        model = joblib.load(model_path)
        print("✅ Model loaded from:", model_path)
    except Exception as e:
        print("❌ Failed to load model:", str(e))
        raise

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        inputs = np.array(data)
        predictions = model.predict(inputs)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
