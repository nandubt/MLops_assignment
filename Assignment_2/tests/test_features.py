from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_temporal_variable_transformer(input_data):
    # Given
    extracter = ExtractLetterTransformer(
        variables=config.model_config.cabin_var_imputation
    )

    SAMPLE_INDEX = 5
    SAMPLE_VALUE = "G"
    SAMPLE_RAW_VALUE = "G6"
    FEATURE = "cabin"
    # print(input_data[:10])
    assert input_data[FEATURE].iat[SAMPLE_INDEX] == SAMPLE_RAW_VALUE
    # When
    subject = extracter.fit_transform(input_data)

    # Then
    assert subject[FEATURE].iat[SAMPLE_INDEX] == SAMPLE_VALUE
