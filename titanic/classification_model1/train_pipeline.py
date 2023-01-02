# run training function
from classification_model1.pipeline import titanic_pipe
from sklearn.model_selection import train_test_split
from classification_model1.config.core import config
from classification_model1.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    data = load_dataset(file_name=config.app_config.raw_data_file)
    X = data[config.model_config.features]
    y = data[config.model_config.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.model_config.test_size, random_state=config.model_config.random_state)

    titanic_pipe.fit(X_train, y_train)
    save_pipeline(pipeline_to_persist=titanic_pipe)


if __name__ == "__main__":
    run_training()
