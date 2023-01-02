# make prediction function
from typing import Union

import pandas as pd
from classification_model1.processing.validation import validate_inputs
from classification_model1 import __version__ as _version
from classification_model1.config.core import config
from classification_model1.processing.data_manager import load_pipeline


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
print(pipeline_file_name)
_titanic_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    df = pd.DataFrame(input_data)
    validated_df, errors = validate_inputs(input_data=df)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _titanic_pipe.predict(validated_df[config.model_config.features])
        results = {"predictions": predictions, "version": _version, "errors": errors}

    return results
