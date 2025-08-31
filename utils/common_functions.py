import os
import pandas
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found in given path")
        with open(file_path,'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info('Succesfully read YAML file')
            return config
    except Exception as e:
        logger.error("Error while reading YAML file")
        raise CustomException('Failed to read YAML file', e)