import os
import sys

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            parameters = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model, parameters, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object_from_cloudinary(public_id):
    """Download a serialized dill object from Cloudinary.

    Args:
        public_id: Either a full Cloudinary URL or a raw public_id path.
    """
    try:
        import requests
        public_id = str(public_id).strip()
        if not public_id:
            raise ValueError("Cloudinary asset reference is empty")

        if public_id.startswith(("http://", "https://")):
            url = public_id
        else:
            cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
            if not cloud_name:
                raise ValueError("CLOUDINARY_CLOUD_NAME is required when using Cloudinary public_id")
            url = f"https://res.cloudinary.com/{cloud_name}/raw/upload/{public_id.lstrip('/')}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return dill.loads(response.content)
    except Exception as e:
        raise CustomException(e, sys)