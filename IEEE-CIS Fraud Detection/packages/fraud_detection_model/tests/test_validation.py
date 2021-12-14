import json

from fraud_detection_model.processing.validation import validate_inputs


def test_validate_inputs(sample_test_data):
    # Given
    dataset = sample_test_data

    # When
    validated_inputs, errors = validate_inputs(inputs=dataset)
    # Then
    assert not errors

    # we expect that 68 rows
    assert len(dataset) == 50
    assert len(validated_inputs) == 50


def test_validate_inputs_identifies_errors(sample_test_data):
    # Given
    dataset = sample_test_data

    dataset.loc[0, "TransactionAmt"] = "Kurtis"  # we expect a float

    # When
    validated_inputs, errors = validate_inputs(inputs=dataset)
    errors = json.loads(errors)

    # Then
    assert errors
    assert errors[0]["msg"] == "value is not a valid float"
