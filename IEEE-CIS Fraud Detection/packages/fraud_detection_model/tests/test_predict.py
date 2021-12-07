from sklearn.metrics import roc_auc_score

from fraud_detection_model import pipeline
from fraud_detection_model.predict import make_prediction


def test_make_predict(sample_test_data):
    # Given
    transaction, identity = sample_test_data

    results = make_prediction(transaction=transaction, identity=identity)

    assert len(results["predictions"]) == len(transaction)


def test_prediction_quality_against_benchmark(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    benchmark = 0.90

    _pipe = pipeline.fraud_detection_pipe.fit(X_train, y_train)

    # When
    predictions = _pipe.predict_proba(X_test)
    score = roc_auc_score(y_test, predictions[:, 1])

    # Then
    assert predictions is not None
    assert score > benchmark
