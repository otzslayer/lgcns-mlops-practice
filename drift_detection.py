import os
import sys
import joblib
import warnings
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import train_test_validation, model_evaluation

from src.common.constants import (
    ARTIFACT_PATH,
    DATA_PATH,
    LOG_FILEPATH,
)
from src.common.logger import handle_exception, set_logger
from src.preprocess import CAT_FEATURES, preprocess_pipeline


logger = set_logger(os.path.join(LOG_FILEPATH, "logs.log"))
sys.excepthook = handle_exception
warnings.filterwarnings(action="ignore")


def main(detect_type: str):
    DATE = datetime.now().strftime("%Y%m%d")
    model = joblib.load(os.path.join(ARTIFACT_PATH, "model.joblib"))
    label = "rent"

    if detect_type[0] == 'type01':
        train_df = pd.read_csv(
            os.path.join(DATA_PATH, "house_rent_train.csv"), 
            usecols=lambda x: x not in ["area_locality", "posted_on", "id"]
            )
        test_df = pd.read_csv(
            os.path.join(DATA_PATH, "house_rent_test.csv"), 
            usecols=lambda x: x not in ["area_locality", "posted_on", "id"]
            )
        
        logger.debug(f"{train_df.info()}")
        logger.debug(f"{test_df.info()}")

        train_ds = Dataset(
            train_df, 
            label=label,
            cat_features=CAT_FEATURES,
            )
        test_ds = Dataset(
            test_df, 
            label=label,
            cat_features=CAT_FEATURES,
            )
        validation_suite = train_test_validation()
        suite_result = validation_suite.run(train_ds, test_ds)

        logger.info(f"Data Validaton Done")

        save_path = os.path.join(
            ARTIFACT_PATH, 
            f"{DATE}_drift_detection_type01.html"
            )
        logger.info(suite_result.save_as_html(save_path))

        logger.info(
            "Result can be found in the following path:\n" f"{save_path}"
        )
    elif detect_type[0] == 'type02':
        train_df = pd.read_csv(
            os.path.join(DATA_PATH, "house_rent_train.csv"), 
            usecols=lambda x: x not in ["area_locality", "posted_on", "id"]
            )
        test_df = pd.read_csv(
            os.path.join(DATA_PATH, "house_rent_test.csv"), 
            usecols=lambda x: x not in ["area_locality", "posted_on", "id"]
            )
        
        _X_train = train_df.drop([label], axis=1)
        y_train = np.log1p(train_df[label])
        X_train = preprocess_pipeline.fit_transform(X=_X_train, y=y_train)

        _X_test = test_df.drop([label], axis=1)
        y_test = np.log1p(test_df[label])
        X_test = preprocess_pipeline.fit_transform(X=_X_test, y=y_test)
        
        train_ds = Dataset(
            X_train,
            label=y_train,
            cat_features=CAT_FEATURES,
            )
        test_ds = Dataset(
            X_test,
            label=y_test,
            cat_features=CAT_FEATURES,
            )
        evaluation_suite = model_evaluation()
        suite_result = evaluation_suite.run(train_ds, test_ds, model)
        
        logger.info(f"Model Validaton Done")

        save_path = os.path.join(
            ARTIFACT_PATH, 
            f"{DATE}_drift_detection_type02.html"
            )
        suite_result.save_as_html(save_path, show_additional_outputs=False)

        logger.info(
            "Result can be found in the following path:\n" f"{save_path}"
        )

    else:
        logger.warning(f"Detection Type `{detect_type}` is not valid!")


def get_arguments():
    """
    arguments 파싱 함수
    """ 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        nargs='+',
        dest='detect_type',
        help=': `type01` for data validation | `type02` for model validation'
        )
    detect_type = parser.parse_args().detect_type
    return detect_type


if __name__ == '__main__':
    detect_type = get_arguments()
    main(detect_type)