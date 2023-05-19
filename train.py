import os
import sys

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from common.constants import DATA_PATH, LOG_FILEPATH
from common.logger import handle_exception, set_logger
from common.metrics import rmse_cv_score
from common.utils import get_param_set
from preprocess import preprocess_pipeline


logger = set_logger(os.path.join(LOG_FILEPATH, "logs.log"))
sys.excepthook = handle_exception

if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_PATH, "house_rent_train.csv"))

    _X = train_df.drop(["Rent", "Area Locality", "Posted On"], axis=1)
    y = np.log1p(train_df["Rent"])
    
    X = preprocess_pipeline.fit_transform(X=_X, y=y)
    
    params_candidates = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "max_features": [1.0, 0.9, 0.8, 0.7]
    }
    
    param_set = get_param_set(params=params_candidates)
    
    for i, params in enumerate(param_set):
        logger.debug(f"Run {i}: {params}")
        regr = GradientBoostingRegressor(**params)
        
        regr.fit(X, y)

        # get evaluations scores
        score_cv = rmse_cv_score(regr, X, y)
        
        # TODO: 결과 처리하는 코드 추가
        # TODO: Artifact 저장하도록
        
    # TODO: Best hp 찾아서 저장하도록