# we are importing TestClient from fastapi.testclient to simulate HTTP requests to our FastAPI application

from fastapi.testclient import TestClient

# Import application object defined from main.py
from main import app

client = TestClient(app)

# now we have to test the endpoint
# name of function should start with test_ to be recognized by pytest
def test_health_check(): 
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# we will send a post request to the /predict endpoint with sample input data
def test_predict_iris_prediction():
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    #send a simulated POST request to the /predict endpoint with the input data and check if the response is as expected
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "class_name" in response.json()

    assert response.json()["class_name"] in ["setosa","versicolor","virginica"] 
    assert response.json()["prediction"] in [0,1,2]
