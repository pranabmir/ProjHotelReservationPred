Predict hotel reservation outcomes using a deployed machine learning API.

---

## Overview

**ProjHotelReservationPred** is an end-to-end Machine Learning project focused on predicting hotel reservation outcomes, such as whether a booking will be cancelled or confirmed.  

The project demonstrates the **complete ML lifecycle**:  
- Data ingestion and preprocessing  
- Model training, validation, and saving artifacts  
- API development for real-time predictions  
- Deployment on **Google Cloud Run** using **Docker**  
- CI/CD integration with **Jenkins**  

This makes it not only a machine learning use case, but also a practical example of how to take a model from research to production.

---

## API Endpoint

A live API has been deployed here: https://ml-project-190505162245.us-central1.run.app

Example output would be whether customer is likely to cancel booking or not. This basically predict fraud and spams bookings.

LGBM is used for the model with with about 93% accuracy. However, model can be better if trained longer or random forest or  XGBoost is used. But due to resource limiration, LGBM is used.

## Tech Stack

Language: Python  
ML libraries: lgbm  
Web framework: Flask  
Deployment: Google Cloud Run  
Containerization: Docker  
CI/CD: Jenkins  
Configuration management: YAML / JSON configs  
Experiments and Exploratory analysis: Jupyter notebooks  

---

## Repo structure

ProjHotelReservationPred/  
├── application.py              # Entry point to run the prediction API  
├── Dockerfile                  # Docker image builder  
├── Jenkinsfile                 # Jenkins pipeline config for CI/CD  
├── requirements.txt            # Python dependencies  
├── setup.py                    # script for install prerequisites   
├── setuptest.py                # Just a testfile-please ignore

├── artifacts/                  # Stores outputs of ML pipeline  
│   ├── data/                   # Raw / processed datasets  
│   ├── model/                  # Trained model files  
│   └── scaler/                 # Preprocessing scalers, encoders, etc.  

├── config/                     # Configurations for pipeline, paths, parameters  
│   └── config.yaml             # all config paths

├── custom_jenkins/             # Jenkins customization, helpers, or scripts  

├── logs/                       # Contains logs of development

├── notebook/                   # Jupyter notebooks for EDA and experimentation  
│   └── hotel_reservation.ipynb # notebook for experiments

├── pipeline/                   # Pipeline scripts  
│   ├── data_ingestion.py       # Code for fetching and preparing data  
│   ├── data_preprocessing.py   # Feature engineering, scaling, encoding  
│   ├── model_trainer.py        # Model training and evaluation  
│   └── prediction_pipeline.py  # Pipeline for serving predictions  

├── src/                        # Core ML source code  
│   ├── components/             # Reusable ML components  
│   ├── entity/                 # Data classes and entities  
│   ├── exception.py            # Custom exception handling  
│   ├── logger.py               # Custom logging
│   └── utils                   # utility function such as read_yaml and load_yaml


