import math

import numpy as np

from classification_model.predict import make_prediction


def test_make_prediction(input_data):
    # Given
    expected_row = 4
    expected_fourth_prediction_value = 1.0
    expected_no_predictions = 262

    # When
    result = make_prediction(input_data=input_data)

    # Then
    predictions = result.get("predictions")

    assert isinstance(predictions, list)
    assert isinstance(predictions[expected_row], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(
        predictions[expected_row], expected_fourth_prediction_value, abs_tol=100
    )
