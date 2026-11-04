# Import TestClient so we can send simulated HTTP requests
from fastapi.testclient import TestClient

# Import application object defined in main.py
from main import create_app


class FakeIrisModel:
    def predict(self, features):
        # Always return class 0
        return [0]


def fake_model_loader():
    return FakeIrisModel()


app = create_app(model_loader=fake_model_loader)


def test_health_endpoint_reports_model_metadata():
    test_app = create_app(
        model_loader=fake_model_loader,
        version_loader=lambda: "test-version"
    )

    with TestClient(test_app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_name": "iris-classifier",
        "model_version": "test-version"
    }


def test_predict_endpoint_returns_setosa_for_fake_model():
    test_app = create_app(
        model_loader=fake_model_loader,
        version_loader=lambda: "test-version"
    )

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    with TestClient(test_app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 0,
        "class_name": "setosa"
    }