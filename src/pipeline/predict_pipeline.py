import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, load_object_from_cloudinary

# Local artifacts root (used when USE_CLOUDINARY is False)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cloudinary defaults. These can be overridden with environment variables:
# CLOUDINARY_MODEL_ASSET and CLOUDINARY_PREPROCESSOR_ASSET.
DEFAULT_CLOUDINARY_MODEL_ASSET = "model_tu4c9k.pkl"
DEFAULT_CLOUDINARY_PREPROCESSOR_ASSET = "preprocessor_j3sriu.pkl"


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            use_cloudinary = os.environ.get("USE_CLOUDINARY", "False").lower() == "true"

            if use_cloudinary:
                model_asset = os.environ.get("CLOUDINARY_MODEL_ASSET", DEFAULT_CLOUDINARY_MODEL_ASSET)
                preprocessor_asset = os.environ.get("CLOUDINARY_PREPROCESSOR_ASSET", DEFAULT_CLOUDINARY_PREPROCESSOR_ASSET)
                model = load_object_from_cloudinary(model_asset)
                preprocessor = load_object_from_cloudinary(preprocessor_asset)
            else:
                model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'model.pkl')
                preprocessor_path = os.path.join(PROJECT_ROOT, 'artifacts', 'preprocessor.pkl')
                model = load_object(file_path=model_path)
                preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
            self,
            gender: str,
            race_ethnicity: str,
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)