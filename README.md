# Iris ML Pipeline

An MLOps example that trains an Iris classifier, tracks experiments and model
versions with MLflow, and serves the champion model through FastAPI.

## Project layout

- `mlops-first-pipeline/train.py` - trains, evaluates, and registers the model
- `mlops-first-pipeline/main.py` - FastAPI application
- `mlops-first-pipeline/test/` - API tests
- `mlops-first-pipeline/Dockerfile` - API container image
- `.github/workflows/ci.yaml` - test and Docker CI workflow

## Local setup

From the repository root, create and activate the virtual environment:

```powershell
python -m venv kiet-env
.\kiet-env\Scripts\Activate.ps1
pip install -r .\mlops-first-pipeline\requirements.txt
```

## Start MLflow

MLflow uses port `5000` for tracking and the model registry. Start it in a
separate terminal:

```powershell
.\kiet-env\Scripts\mlflow.exe server `
  --host 127.0.0.1 `
  --port 5000 `
  --workers 1 `
  --backend-store-uri "sqlite:///C:/path/to/ML-Pipeline/mlops-first-pipeline/mlflow.db"
```

Keep the server running, then open <http://127.0.0.1:5000>.

## Train the model

Run the command from `mlops-first-pipeline` so the model artifacts are written
to the expected directories:

```powershell
cd .\mlops-first-pipeline
python .\train.py
```

## Run the API

From `mlops-first-pipeline`:

```powershell
python -m uvicorn main:app --reload
```

The API is available at <http://127.0.0.1:8000>. The model is loaded from the
MLflow `iris-classifier@champion` alias.

## Test

```powershell
python -m pytest -v .\mlops-first-pipeline\test
```

## Docker

Build and run the API container:

```powershell
docker build -t iris-ml-api .\mlops-first-pipeline
docker run --rm -p 8000:8000 iris-ml-api
```

## Continuous integration

GitHub Actions runs the API tests on pushes and pull requests targeting
`main`. On pushes to `main`, it also builds and publishes the Docker image to
GitHub Container Registry.
