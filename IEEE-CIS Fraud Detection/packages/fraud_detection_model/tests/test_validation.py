from fraud_detection_model.processing.validation import validate_inputs


def test_validate_inputs(sample_test_data):
    # Given
    transaction, identity = sample_test_data

    # When
    validated_inputs, errors = validate_inputs(
        transaction=transaction, identity=identity
    )
    # Then
    assert not errors

    # we expect that 68 rows
    assert len(transaction) == 68
    assert len(identity) == 68
    assert len(validated_inputs) == 68


def test_validate_inputs_identifies_errors(sample_test_data):
    # Given
    transaction, identity = sample_test_data

    transaction.loc[0, "TransactionAmt"] = "Kurtis"  # we expect a float

    # When
    validated_inputs, errors = validate_inputs(
        transaction=transaction, identity=identity
    )

    # Then
    assert errors
    assert errors[0]["TransactionAmt"] == ["Not a valid number."]
