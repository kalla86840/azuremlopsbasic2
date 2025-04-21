import os
import joblib
import json
import numpy as np

# This method is called once when the service is started
def init():
    global model
    # Path where Azure ML mounts the model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "linear_model.pkl")
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")

# This method is called for every request to the endpoint
def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        input_array = np.array(data)
        predictions = model.predict(input_array)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
