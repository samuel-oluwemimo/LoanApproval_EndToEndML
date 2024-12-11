import os
import sys
import logging
import pandas as pd
from dataclasses import dataclass
from src.utils import save_object, evaluate_model_with_cv, get_test_metrics
from src.exception import CustomException
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

@dataclass
class ModelTrainerConfig:
    model_save_path: str = os.path.join("artifacts", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def model_training(self, processed_train, processed_test):
        try:
            logging.info("Starting model training process.")

            # Split train and test data into features and labels
            processed_train, processed_test = pd.read_csv(processed_train),  pd.read_csv(processed_test)
            X_train, y_train = processed_train.iloc[:, :-1], processed_train.iloc[:, -1]
            X_test, y_test = processed_test.iloc[:, :-1], processed_test.iloc[:, -1]

            # Define models
            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'SVC': SVC(probability=True),
                'CatBoostClassifier': CatBoostClassifier(verbose=0),
                'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                'LGBMClassifier': LGBMClassifier()
            }
            #parameters for hyperparameter tuning
            param_grids = {
                'LogisticRegression': {
                    'C': [0.1, 1.0],
                    'solver': ['liblinear']
                },
                'DecisionTreeClassifier': {
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5]
                },
                'RandomForestClassifier': {
                    'n_estimators': [100, 150],
                    'max_depth': [10, 15],
                    'min_samples_split': [2],
                    'min_samples_leaf': [1, 2]
                },
                'KNeighborsClassifier': {
                    'n_neighbors': [3, 5, 7],
                    'metric': ['minkowski']
                },
                'SVC': {
                    'C': [0.1, 1.0],
                    'kernel': ['linear', 'rbf']
                },
                'CatBoostClassifier': {
                    'iterations': [200, 500],
                    'learning_rate': [0.01, 0.1]
                },
                'XGBClassifier': {
                    'n_estimators': [100, 150],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1]
                },
                'LGBMClassifier': {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.01, 0.1]
                }
            }


            best_model_name = None
            best_model = None
            best_cv_score = 0
            model_results = []

            # Train and evaluate models
            for name, model in models.items():
                logging.info(f"Training model: {name}")

                # Cross-validation
                cv_score = evaluate_model_with_cv(model, X_train, y_train)
                logging.info(f"{name} cross-validation F1 score: {cv_score:.4f}")

                # Hyperparameter tuning
                param_grid = param_grids[name]
                rs_cv = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3, n_jobs=-1)
                rs_cv.fit(X_train, y_train)

                # Set the best parameters for the model
                model.set_params(**rs_cv.best_params_)
                model.fit(X_train,y_train)

                # Train and evaluate on test set
                # model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                test_metrics = get_test_metrics(y_test, y_test_pred)

                # Log metrics
                model_results.append({
                    "model": name,
                    "cv_f1_score": cv_score,
                    **test_metrics
                })

                # Update the best model based on cross-validation score
                if cv_score > best_cv_score:
                    best_cv_score = cv_score
                    best_model_name = name
                    best_model = model

            logging.info(f"Best model selected: {best_model_name} with cross-validation F1 score: {best_cv_score:.4f}")

            # Save the best model
            save_object(self.config.model_save_path, best_model)
            logging.info(f"Best model '{best_model_name}' saved at {self.config.model_save_path}.")

            return {
                "best_model": best_model_name,
                "best_cv_score": best_cv_score,
                "results": model_results
            }

        except Exception as e:
            logging.error("Error occurred during model training.")
            raise CustomException(e, sys)
