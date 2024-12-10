import pickle
import os
import logging
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves an object to the specified file path using pickle.
    
    Parameters:
        file_path (str): Path to save the pickle file.
        obj (object): Python object to be serialized and saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directories if not existing
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Error while saving object to {file_path}: {str(e)}")
        raise CustomException(e)
