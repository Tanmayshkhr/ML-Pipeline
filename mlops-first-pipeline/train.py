from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ,precision_score , recall_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Configure mlflow tracking part
# mlflow server stores experiment metadata and model records in a backend store (like a database) and artifacts (like model files) in an artifact store (like a file system or cloud storage).
mlflow.set_tracking_uri("http://localhost:6001")

# Group related training runs under one exeriment name, which can be useful for organizing and comparing different runs of the same experiment.
mlflow.set_experiment("Iris_Classification")

n_estimators = 100
random_state = 42

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model_parameters = {"model_type": "RandomForestClassifier", "n_estimators": n_estimators, "random_state": random_state , "test_size": 0.2}
with mlflow.start_run() as run:
    mlflow.log_params(model_parameters)
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # 3. Create and train the model
    model.fit(X_train, y_train)
    # 4. Make predictions
    y_pred = model.predict(X_test)
    # 5. Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy: {accuracy:.2f}")
    precision = precision_score(y_test, y_pred, average='macro', zero_division = 0)
    mlflow.log_metric("precision", precision)
    recall = recall_score(y_test, y_pred, average='macro', zero_division = 0)
    mlflow.log_metric("recall", recall)

    print("MLFlow run ID:", run.info.run_id)
    print("Test accuracy:", accuracy)
    print("Test precision:", precision)
    print("Test recall:", recall)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
        registered_model_name="iris_classifier",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    print("Registered Model: iris_classifier")

# 6. Create the models directory
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

# 7. Save the trained model
model_path = model_dir / "model.pkl"
joblib.dump(model, model_path)

print(f"Model saved to {model_path.resolve()}")