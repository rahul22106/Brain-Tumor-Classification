import os
import sys
import yaml
import json
import joblib
from pathlib import Path
from typing import Any
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
from BT_Classification import logger
from BT_Classification import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns ConfigBox object
    
    Args:
        path_to_yaml (Path): path to yaml file
        
    Returns:
        ConfigBox: ConfigBox type object
        
    Raises:
        ValueError: if yaml file is empty
        CustomException: for any other errors
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories
    
    Args:
        path_to_directories (list): list of paths of directories
        verbose (bool, optional): ignore if multiple directories are to be created. Defaults to True.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save json data
    
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"json file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json files data
    
    Args:
        path (Path): path to json file
        
    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def save_model(data: Any, path: Path):
    """
    Save model using joblib
    
    Args:
        data (Any): model object to be saved
        path (Path): path to save model
    """
    try:
        joblib.dump(value=data, filename=path)
        logger.info(f"Model saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def load_model(path: Path) -> Any:
    """
    Load model using joblib
    
    Args:
        path (Path): path to model file
        
    Returns:
        Any: model object
    """
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded from: {path}")
        return model
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size of file in KB
    
    Args:
        path (Path): path to file
        
    Returns:
        str: size in KB
    """
    try:
        size_in_kb = round(os.path.getsize(path)/1024)
        return f"~ {size_in_kb} KB"
    except Exception as e:
        raise CustomException(e, sys)