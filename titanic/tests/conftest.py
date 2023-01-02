import pytest
from classification_model1.processing.data_manager import _load_raw_dataset
from classification_model1.config.core import config
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_input_data():
    data = _load_raw_dataset(file_name=config.app_config.raw_data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        data, data[config.model_config.target],
        test_size=config.model_config.test_size, random_state=config.model_config.random_state)

    return X_test
