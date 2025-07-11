# 🔁 Automated MLOps Pipeline on AWS

An end-to-end MLOps pipeline that trains, deploys, and monitors a machine learning model using AWS SageMaker. This project demonstrates best practices in CI/CD for ML systems, leveraging the power of automation for robust model management in production environments.

---

## 🚀 Project Overview

This project sets up a fully automated MLOps pipeline using:

- **AWS SageMaker** for model training and hosting  
- **GitHub Actions** for CI/CD  
- **S3** for data and artifact storage  
- **SageMaker Model Monitor** for automated model quality tracking  
- **IAM & OIDC** for secure role-based access control

---

## 🧠 Problem Statement

We build a binary classification model using Scikit-learn to solve a supervised ML task, and:

1. Train the model via **SageMaker training jobs**
2. Automatically deploy the model to an endpoint
3. Continuously monitor the model’s prediction drift and performance degradation using **Model Monitor**

---

## 🛠️ Tech Stack

| Layer           | Tools                                         |
|----------------|-----------------------------------------------|
| Model Training | `scikit-learn`, `SageMaker Python SDK`        |
| Deployment     | `SageMaker Hosting`, `Boto3`, `S3`            |
| Monitoring     | `SageMaker Model Monitor`                     |
| CI/CD          | `GitHub Actions`, `OIDC`, `IAM Roles`         |
| Infrastructure | `AWS S3`, `IAM`, `SageMaker`, `CloudWatch`    |

---

## 📁 Project Structure

```
mlops-pipeline-aws/
│
├── data/
│ └── titanic.csv # Sample dataset
├── train.py # Local training script
├── sagemaker_train_deploy.py # Launch training job on SageMaker
├── sagemaker_deploy_endpoint.py # Deploy trained model to SageMaker endpoint
├── sagemaker_monitor.py (WIP) # Model monitoring script
├── requirements.txt # Dependencies
└── README.md # Project overview and instructions

```

---

## 🔧 How to Run

### 1. 🔑 Prerequisites

- AWS Account with SageMaker, S3, and IAM permissions
- S3 bucket created (e.g., `apoorvawsbucket321`)
- AWS CLI configured locally (`aws configure`)
- Python 3.8+ with `boto3`, `sagemaker`, `joblib`

### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

3. 🚂 Start Training

```bash
python sagemaker_train_deploy.py
```

4. 🌐 Deploy Endpoint

Edit sagemaker_deploy_endpoint.py and set:

```bash
python sagemaker_deploy_endpoint.py
```

5. 🛡️ (Optional) Enable Monitoring
COMING SOON: sagemaker_monitor.py to activate Model Monitor.


✅ Results
The trained model achieved:

Accuracy: ~76%

F1-Score: 0.75 (macro avg)

Model stored in: model.joblib and uploaded to S3

Endpoint: Created and ready for inference via REST API
