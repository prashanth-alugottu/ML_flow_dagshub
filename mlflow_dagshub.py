import os
import warnings
import sys

import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import train_test_split 
from urllib.parse import urlparse 
import mlflow
import mlflow.tensorflow
# from tensorflow.keras import models,layer,Dense,Sequential

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# from tensorflow.keras.layers import Dense



import dagshub 
import logging
import certifi

mlflow.tensorflow.autolog()

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

dagshub.init(repo_owner='chinnuprashanth868', repo_name='ML_flow_dagshub', mlflow=True)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
        # print(data)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        
    train, test = train_test_split(data)
    train_x = train.drop(["quality"],axis=1)
    test_x = test.drop(["quality"],axis=1)
    
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    with mlflow.start_run():
        
        model = Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(train_x, train_y, epochs=10)
        
        predictions = model.predict(test_x)
                
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        mae = mean_absolute_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)

        print("TensorFlow Model Metrics:")
        print("  RMSE:", rmse)
        print("  MAE :", mae)
        print("  R2  :", r2)
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        
        remote_server_url="https://github.com/prashanth-alugottu/ML_flow_dagshub.git"
        mlflow.set_tracking_uri(remote_server_url)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
         # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.keras.log_model(
                model, "model", registered_model_name="TensorFlowWineModel")
        else:
            mlflow.keras.log_model(model, "model")

        
        
        



