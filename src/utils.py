import logging
import os
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves an object to the specified file path using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Error while saving object to {file_path}: {str(e)}")
        raise CustomException(e)

def evaluate_model_with_cv(model, X_train, y_train, cv_folds=5):
    """
    Performs cross-validation on a given model and returns the mean F1 score.
    """
    try:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, scoring='f1', cv=skf)
        mean_cv_score = np.mean(cv_scores)
        return mean_cv_score
    except Exception as e:
        logging.error(f"Error during cross-validation: {str(e)}")
        raise CustomException(e)


def get_test_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics for the test set predictions.
    """
    try:
        metrics = {
            "f1_score": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred),
            "log_loss": log_loss(y_true, y_pred),
        }
        return metrics
    except Exception as e:
        logging.error(f"Error during metric evaluation: {str(e)}")
        raise CustomException(e)
