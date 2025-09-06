import os
import pandas
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params =RANDOM_SEARCH_PARAMS
    
    def load_and_split(self):
        try:
            logger.info(f"loading training data from {self.train_path}")
            train_df = load_data(self.train_path)
            logger.info(f"loading test data from {self.test_path}")
            test_df = load_data(self.test_path)

            x_train = train_df.drop(columns = ['booking_status'])
            y_train = train_df['booking_status']

            x_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info('Data splitted successfully for Model Training')
            return x_train,y_train,x_test,y_test
        except Exception as e:
            logger.info(f'Error while loading data {e}')
            raise CustomException('Error while loading the data',e)
        
    def train_lgbm(self,x_train,y_train):
        try:
            logger.info('Initializing model')
            lgbm_model = lightgbm.LGBMClassifier(random_state =self.random_search_params['random_state'])
            logger.info('starting hyperparams tuning')
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter = self.random_search_params['n_iter'],
                n_jobs = self.random_search_params['n_jobs'],
                verbose = self.random_search_params['verbose'],
                random_state= self.random_search_params['random_state'],
                scoring = self.random_search_params['scoring']
            )
            random_search.fit(x_train,y_train)
            logger.info('hyparam tuning completed')
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f"Best Params are: {best_params}")
            return best_lgbm_model
        except Exception as e:
            logger.info(f"Error while training data {e}")
            raise CustomException("Error while training data",e)

    def evaluate_model(self,model,x_test,y_test):
        try:
            logger.info('Evaluating model')
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            logger.info(f"Accuracy score: {accuracy}")
            logger.info(f"Precision score: {precision}")
            logger.info(f"Recall score: {recall}")
            logger.info(f"F1 score: {f1}")
            return {"accuracy":accuracy,
                    "precision":precision,
                    "recall":recall,
                    "f1":f1}
        except Exception as e:
            logger.info(f"Error while evaluating model {e}")
            raise CustomException("Error while evaluating model",e)
    
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            logger.info('Output Directory created')
            joblib.dump(model,self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save model",e)
        
    def run(self):
        try:
            logger.info("Starting model training pipeline")
            x_train,y_train,x_test,y_test = self.load_and_split()
            best_lgbm_model = self.train_lgbm(x_train,y_train)
            metrics = self.evaluate_model(best_lgbm_model,x_test,y_test)
            self.save_model(best_lgbm_model)
            logger.info("Model training successfully completed")
        except Exception as e:
            logger.error(f"Error in the model training pipeline  {e}")
            raise CustomException("error in the model training pipeline",e)

if __name__==  "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.run()