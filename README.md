# 📰 News Classification System

This project is a **news classification system** that processes and classifies news headlines into predefined categories using **machine learning models**. It is composed of three **Dockerized services** that ensure replicability and modularity.

## 📌 Project Overview

The system consists of the following components:

1. **Data Preparation** - Downloads and preprocesses the dataset, cleaning and structuring it for training.
2. **Model Training** - Trains machine learning models, including fine-tuning a **DistilBERT** model and training an **XGBoost** classifier.
3. **API Service** - Deploys the trained model as a REST API for real-time news classification.

Additionally, a **client script** fetches live news headlines and sends them to the API for classification.

## 🏗️ Setup & Execution

Each service is containerized with **Docker** and contains a `Makefile` to simplify execution. Docker images are available in the following DockerHub repository: https://hub.docker.com/r/josecaloca/multiclass-text-classification

Using `make` from the `Makefile`, we can simplify the execution of the project. It is recommended to see the available `make` commands by running:

```bash
make help
```
In overall, for building the Docker images and running the containers, 1 command is needed:

```bash
make build run
```

Alternatively, we can build the Docker images and run the containers of each service as follows:

### 1️⃣ Run Data Preparation
```bash
make build-data-prep run-data-prep
```

### 2️⃣ Run Model Training
```bash
make build-model-train run-model-train
```

### 3️⃣ Run API Service
```bash
make build-api run-api
```

**Note**: Once the API is running, it will be available at: ```http://127.0.0.1:8000```

## 🚀 Using the System

### 🔍 Classify News Headlines
A client script (`./client.py`) fetches real-time news headlines from an external news API and classifies them using the deployed model.

Run the client script:
```cd
python client.py
```

### 🛠️ API Endpoints

The API exposes the following endpoint:

- POST /predict
    - Description: Classifies a news headline into a category.
    - Payload: `{"title": "Some news headline"}`
    - Response:
    ```json
    {"category": "business"}
    ```

## 📂 Project Structure (High-Level)

```bash
.
├── README.md               # Project documentation
├── client.py               # Fetches news headlines and classifies them
├── services                # The three main services
│   ├── api                 # API service for model inference
│   ├── data_preparation    # Data preprocessing pipeline
│   └── model_training      # Training pipeline for machine learning models

```

## 🔧 Key Technologies

This project utilizes a modern **MLOps pipeline** with well-structured **Dockerized services** to ensure reproducibility and efficient deployment. Below are the key technologies used:

### 📌 **Machine Learning & Model Training**
- **Hugging Face Transformers** – Used for **fine-tuning a DistilBERT model** to classify news headlines.
- **XGBoost** – A gradient boosting algorithm used for training an alternative classifier based on embeddings extracted from DistilBERT.
- **Scikit-Learn** – Used for additional preprocessing and evaluation metrics.
- **CometML** – Integrated to **track experiments**, log training metrics during model development and used also as a secondary model registry.

### 📦 **Model Registry & Feature Store (Hugging Face Hub + CometML)**
- **Hugging Face Hub** acts as:
  - A **Model Registry** for storing and versioning **fine-tuned DistilBERT models**
  - A **Feature Store** where the **pre-processed dataset** (tokenized) is stored and retrieved for training and inference.
- The trained XGBoost model is stored in **CometML**.

### 🚀 **Model Deployment & API**
- **Litserve** – A high-performance framework built on **FastAPI**, optimized specifically for **ML model deployment**.
- **Requests** – Used in `client.py` to fetch live news headlines and send them to the API for classification.

### 🐳 **Containerization & Automation**
- **Docker** – Ensures each service runs in an isolated and reproducible environment.
- **Makefiles** – Automates the **build & run** process for each service (`data_preparation`, `model_training`, and `api`).
- **UV** – Used as the **dependency manager**, ensuring reproducibility and fast installations (instead of Poetry).

### 📊 **Experiment Tracking & Logging**
- **CometML** – Used to log metrics such as **F1-score, precision, recall, and confusion matrices** during training.
- **Loguru** – A modern logging library used for structured logging and error tracking across the project.

### 📜 **Code Quality & Linting**
- **Ruff** – A fast and efficient Python linter that enforces best practices and keeps the code clean.
- **Pre-commit Hooks** – Ensures code quality by automatically running linting, formatting, and security checks before commits.
