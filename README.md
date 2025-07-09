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
