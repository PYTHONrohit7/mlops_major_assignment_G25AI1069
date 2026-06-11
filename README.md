# End-to-End MLOps Pipeline: Olivetti Faces Classification

This project implements an automated, end-to-end MLOps pipeline for classifying the **Olivetti faces dataset** using a scikit-learn `DecisionTreeClassifier`. 

The project showcases model development, automated testing via GitHub Actions CI/CD, containerization with Docker, and container orchestration with Kubernetes.

---

## Project Architecture & Branches

The project follows a strict branching strategy:
* **`main`**: Initial project setup and metadata.
* **`dev`**: Model development, testing scripts, and local execution configurations.
* **`docker_cicd`**: Flask web application implementation, Docker configuration, and Kubernetes orchestration deployment manifests.

---

##  Step 2: Model Development & CI/CD (`dev` branch)

The model training and evaluation are handled in the `dev` branch:
* **`train.py`**: Loads the Olivetti faces dataset from `sklearn.datasets`, splits it into 70% train and 30% test sets, trains a `DecisionTreeClassifier`, and saves the model using `joblib` to `savedmodel.pth`.
* **`test.py`**: Loads the saved model (`savedmodel.pth`) and evaluates it on the test set, outputting the test accuracy.
* **`.github/workflows/ci.yml`**: Configured to run on every `push` to the `dev` branch. It sets up Python, installs dependencies, and runs `train.py` and `test.py`.

### Local Execution (Step 2)
```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Run evaluation
python test.py
```

---

##  Step 3: Containerization & Deployment (`docker_cicd` branch)

The application is containerized and deployed using Kubernetes in the `docker_cicd` branch:
* **`app.py`**: A Flask web application that serves a simple HTML page to upload images, preprocesses them (grayscale, resize to 64x64, flatten), and outputs the predicted face class.
* **`Dockerfile`**: Packages the Flask application with Python 3.10-slim.
* **`deployment.yaml`**: Kubernetes manifest defining a Deployment (configured with **3 replicas**) and a **NodePort Service** mapping container port `5000` to port `30007`.

### Docker Build & Push
```bash
docker build -t rohitpaliwal4/mlops-app:latest .
docker push rohitpaliwal4/mlops-app:latest
```

### Kubernetes Deployment
```bash
# Apply deployment and service
kubectl apply -f deployment.yaml

# Check pods status
kubectl get pods

# Accessing the web application (port forwarding if using kind cluster)
kubectl port-forward svc/mlops-app-service 30007:80
```

---

## Self-Healing Demonstration

To ensure that 3 replicas are always running, a pod can be deleted using `kubectl delete pod <pod-name>`. Kubernetes will immediately spin up a replacement pod to maintain the desired count of 3 replicas.

---

## Links
* **GitHub Repository**: [https://github.com/PYTHONrohit7/mlops_major_assignment_G25AI1069](https://github.com/PYTHONrohit7/mlops_major_assignment_G25AI1069)
* **Docker Hub Repository**: [https://hub.docker.com/r/rohitpaliwal4/mlops-app](https://hub.docker.com/r/rohitpaliwal4/mlops-app)
