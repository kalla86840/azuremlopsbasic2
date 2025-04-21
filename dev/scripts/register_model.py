from azureml.core import Workspace, Model

ws = Workspace.create(
    name="mlops-basic-ws",
    subscription_id="2735eac8-e053-4d04-ad5f-83d5f236217b",
    resource_group="mlops-basic-rg",
    location="westus2",
    exist_ok=True,
    show_output=True
)

model = Model.register(
    workspace=ws,
    model_path="outputs/linear_model.pkl",
    model_name="linear_model"
)

print(f"✅ Registered model: {model.name}, version: {model.version}")
