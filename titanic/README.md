# Productionized Titanic Classification Model Package

## Run With Tox (Recommended)
- `pip install tox`
- Make sure you are in the titanic directory (where the tox.ini file is) then run the command: `tox` (this runs the tests and typechecks, trains the model under the hood). The first time you run this it creates a virtual env and installs
dependencies, so takes a few minutes.

## Run Without Tox
- Add titanic *and* classification_model1 paths to your system PYTHONPATH
- `pip install -r requirements/test_requirements`
- Train the model: `python classification_model1/train_pipeline.py`
- Run the tests `pytest tests`
