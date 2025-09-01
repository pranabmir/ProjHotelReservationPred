import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self,train_path,test_path,processed_dir,config_path):
        self.train_path =train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def preprocess_data(self,df):
        try:
            logger.info('Starting Data Processing Step')
            logger.info('Dropping the Columns')
            drop_list = ["Unnamed: 0","Booking_ID"]
            df = df.drop(columns=drop_list)
            df.drop_duplicates(inplace=True)
            cat_cols = self.config['data_processing']['categorical_columns']
            # num_cols = self.config['data_processing']['numerical_columns']
            logger.info('applying label encoding')
            le = LabelEncoder()
            mappings = {}
            for col in cat_cols:
                df[col] = le.fit_transform(df[col])
                mappings[col] = {label: code for label,code in zip(le.classes_,le.transform(le.classes_))}
            logger.info("Label mappings are")
            for col,mapping in mappings.items():
                logger.info(f"{col}: {mapping}")
            return df
        except Exception as e:
            logger.error(f"Error during preprocess step {e}")
            raise CustomException("Error while preprocess data",e)
    
    def balance_data(self,df):
        try:
            logger.info('Handing Imabalanced data')
            x = df.drop(columns = 'booking_status')
            y = df['booking_status']
            smote = SMOTE(random_state=42)
            x_res,y_res = smote.fit_resample(x,y)
            balanced_df = pd.DataFrame(x_res,columns = x.columns)
            balanced_df['booking_status'] = y_res
            logger.info('Data Balanced success')
            return balanced_df
        except Exception as e:
            logger.error(f"Error during balancing step {e}")
            raise CustomException("Error while balancing data",e)
    
    def select_features(self,df):
        try:
            logger.info('Starting Feature Selection step')

            x = df.drop(columns = 'booking_status')
            y = df['booking_status']
            model = RandomForestClassifier(random_state=42)
            model.fit(x,y)
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({'features':x.columns,
                                      'importance':feature_importance})
            top_features_count = self.config["data_processing"]['top_features_count']
            top_features = feature_importance_df.sort_values(by ='importance',ascending=False).iloc[:top_features_count,0].to_list()
            logger.info(f"Top {top_features_count} Features are {top_features} ")
            top_features_df = df[top_features+['booking_status']]
            logger.info("Feature Selection Completed")
            return top_features_df
        except Exception as e:
            logger.error(f"Error during Feature Selection step {e}")
            raise CustomException('Error while Feature selection',e)
    
    def save_data(self,df,file_path):
        try:
            logger.info("Save data in processed folder")
            df.to_csv(file_path,index = False)
            logger.info(f'Data Save successfully to {file_path}')
        except Exception as e:
            logger.error(f"Error during aata saving step {e}")
            raise CustomException('Error while saving data',e)
    
    def process(self):
        try:
             logger.info('Loading Data from Raw directory')
             train_df = load_data(self.train_path)
             test_df = load_data(self.test_path)
             train_df = self.preprocess_data(train_df)
             test_df = self.preprocess_data(test_df)
             train_df = self.balance_data(train_df)
             test_df = self.balance_data(test_df)
             train_df = self.select_features(train_df)
             test_df = test_df[train_df.columns]
             self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
             self.save_data(test_df,PROCESSED_TEST_DATA_PATH)
             logger.info("Data processing done successfully")
        except Exception as e:
            logger.error(f"Error during data processing {e}")
            raise CustomException("Error while processing data pipeline",e)

if __name__=="__main__":
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()