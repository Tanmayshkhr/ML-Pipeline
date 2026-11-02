from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import os


app = FastAPI(title="Iris Model API")

PROJECT_DIR = Path(__file__).resolve().parent

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}",
)

MODEL_URI = "models:/iris-classifier@champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    client = mlflow.MlflowClient()
    champion_model = client.get_model_version_by_alias(
        name="iris-classifier",
        alias="champion"
    )
    model_version = champion_model.version
    print(f"loaded model: iris-classifier:{model_version}")
except Exception as ex:
    raise RuntimeError(
        f"Failed to load model from MLFlow model registry: {ex}"
    )

model_path = PROJECT_DIR / "models" / "iris_model.joblib"
model = joblib.load(model_path)

class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/")
def health_check():
    return {"message": "API is ready to serve..."}

@app.get("/health")
def health_check():
    return {"status": "healthy",
            "model_name": "iris-classifier",
            "model_version": model_version
            }


@app.post("/predict")
def predict(data: IrisRequest):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction]
    }
