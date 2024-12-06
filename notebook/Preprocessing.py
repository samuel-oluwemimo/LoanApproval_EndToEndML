import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def preprocess_data(file_path, target_column, test_size=0.33, random_state=42, apply_pca=True, pca_variance=0.95):
    """
    Preprocess the dataset for training and testing.
    
    Parameters:
        file_path (str): Path to the dataset file.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        apply_pca (bool): Whether to apply PCA for dimensionality reduction.
        pca_variance (float): Explained variance ratio for PCA.
    
    Returns:
        X_train_reduced, X_test, y_train, y_test: Preprocessed and split datasets.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    if 'ApplicationDate' in df.columns:
        df = df.drop('ApplicationDate', axis=1)
    
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical and numerical columns
    cat_columns = X.select_dtypes(include='object').columns
    num_columns = X.select_dtypes(include=['number']).columns
    
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
    
    # Transform the features
    X_transformed = pipeline.fit_transform(X)
    X = pd.DataFrame(X_transformed, columns=pipeline.get_feature_names_out())
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Apply PCA (if required)
    if apply_pca:
        pca = PCA(n_components=pca_variance)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)
        return X_train_reduced, X_test_reduced, y_train, y_test
    
    return X_train, X_test, y_train, y_test