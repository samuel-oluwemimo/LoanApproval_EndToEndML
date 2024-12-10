import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from dataclasses import dataclass
import logging
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessed_train_path: str = os.path.join('artifacts', 'preprocessed_train.csv')
    preprocessed_test_path: str = os.path.join('artifacts', 'preprocessed_test.csv')
    pca_model_path: str = os.path.join('artifacts', 'pca_model.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def preprocess_data(self, train_path, test_path, target_column, apply_pca=True, pca_variance=0.95):
        """
        Preprocesses the data by transforming features and applying optional PCA.
        Saves preprocessed train and test datasets to the artifacts directory.
        """
        logging.info("Started preprocessing data.")
        try:
            # Load datasets
            logging.info(f"Reading train dataset from {train_path}")
            train_df = pd.read_csv(train_path)
            logging.info(f"Reading test dataset from {test_path}")
            test_df = pd.read_csv(test_path)

            # Drop unnecessary columns
            if 'ApplicationDate' in train_df.columns:
                logging.info("Dropping 'ApplicationDate' column.")
                train_df = train_df.drop('ApplicationDate', axis=1)
                test_df = test_df.drop('ApplicationDate', axis=1)

            # Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Identify categorical and numerical columns
            cat_columns = X_train.select_dtypes(include='object').columns
            num_columns = X_train.select_dtypes(include=['number']).columns

            # Define preprocessing pipelines
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('1hot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            pipeline = ColumnTransformer([
                ('num', num_pipeline, num_columns),
                ('cat', cat_pipeline, cat_columns)
            ], remainder='passthrough')

            # Apply transformation
            logging.info("Fitting and transforming train and test data.")
            X_train_transformed = pipeline.fit_transform(X_train)
            X_test_transformed = pipeline.transform(X_test)


            if apply_pca:
                logging.info("Applying PCA.")
                pca = PCA(n_components=pca_variance)
                X_train_transformed = pca.fit_transform(X_train_transformed)
                X_test_transformed = pca.transform(X_test_transformed)
                
                # Save the PCA model using the utility function
                save_object(self.transformation_config.pca_model_path, pca)
                logging.info(f"PCA model saved to {self.transformation_config.pca_model_path}.")


            # Save preprocessed datasets
            logging.info("Saving preprocessed datasets.")
            train_transformed_df = pd.DataFrame(np.c_[X_train_transformed, np.array(y_train)])
            test_transformed_df = pd.DataFrame(np.c_[X_test_transformed, np.array(y_test)])
            train_transformed_df.to_csv(self.transformation_config.preprocessed_train_path, index=False)
            test_transformed_df.to_csv(self.transformation_config.preprocessed_test_path, index=False)
            
            logging.info(f"Preprocessed train data saved to {self.transformation_config.preprocessed_train_path}.")
            logging.info(f"Preprocessed test data saved to {self.transformation_config.preprocessed_test_path}.")

            return (
                self.transformation_config.preprocessed_train_path,
                self.transformation_config.preprocessed_test_path
            )
        except Exception as e:
            logging.error("Error occurred during data preprocessing.")
            raise CustomException(e, sys)


