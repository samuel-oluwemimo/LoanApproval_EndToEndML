# Loan Approval Prediction Project

## Overview
This project aims to predict loan approval using various machine learning models and Principal Component Analysis (PCA). The project includes data preprocessing, model training, and evaluation.

Data
The dataset used for this project is Loan.csv, located in the data directory.

### Notebooks
- EDA_Loan_Approval.ipynb: Exploratory Data Analysis
- Model_Training.ipynb: Model training and evaluation
### Scripts
- Preprocessing
- Preprocessing.py: Script for data preprocessing
### Components
- data_ingestion.py: Handles data ingestion
- data_transformation.py: Applies PCA and other transformations
- model_training.py: Trains various models
### Pipeline
- predict_pipeline.py: Pipeline for making predictions
- train_pipeline.py: Pipeline for training models
### Utilities
- exception.py: Custom exception handling
- logger.py: Logging utility
- utils.py: General utilities
### Models

The trained models are saved in the artifacts directory:
- best_model.pkl: The best performing model
- pca_model.pkl: PCA model
