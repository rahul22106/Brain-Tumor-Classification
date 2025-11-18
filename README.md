# Brain-Tumor-Classification


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml




## 🚀 Deployment & CI/CD

### Python Virtual Environment (venv)
- It is recommended to use a virtual environment for local development:
```bash
  python -m venv venv
```

 ```bash
   brain/scripts/activate  
 ```

```bash   
  pip install -r requirements.txt
  ```

### DockerHub Integration
- The CI/CD pipeline uses two required GitHub secrets:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`
- These are used to build and push Docker images automatically as part of the workflow.

### Render Deployment (via GitHub Actions)
- Automatic deployment to Render is triggered on every push to `main` using a GitHub Actions workflow.
- Required secret: `RENDER_DEPLOY_HOOK_URL`
- Render will pull the latest Docker image and deploy the updated Streamlit app.



## MLflow


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI= https://dagshub.com/rahul22106/Brain-Tumor-Classification.mlflow
MLFLOW_TRACKING_USERNAME= rahul22106

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI= https://dagshub.com/rahul22106/Brain-Tumor-Classification.mlflow

export MLFLOW_TRACKING_USERNAME= rahul22106

```



### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)





