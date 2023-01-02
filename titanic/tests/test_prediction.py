import numpy as np
from sklearn.metrics import accuracy_score

from classification_model1.predict import make_prediction


def test_make_prediction(sample_input_data):

    expected_number_of_predictions = 262
    result = make_prediction(input_data=sample_input_data)
    predictions = result.get('predictions')

    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert result.get('errors') is None
    assert len(predictions) == expected_number_of_predictions

    _predictions = list(predictions)
    y_true = sample_input_data['survived']
    acc_score = accuracy_score(y_true, _predictions)

    assert acc_score > 0.7
