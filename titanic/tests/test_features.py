from classification_model1.processing.features import ExtractLetterTransformer
from classification_model1.config.core import config


def test_extract_letter_transformer(sample_input_data):

    transformer = ExtractLetterTransformer(variables=config.model_config.cabin_vars)

    assert sample_input_data['cabin'].iat[6] == 'E12'

    output = transformer.fit_transform(X=sample_input_data)

    assert output['cabin'].iat[6] == 'E'
