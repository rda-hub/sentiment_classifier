import pytest

from api.model import predict


@pytest.mark.parametrize("payload", [
    "I love this movie!",
    "I hate this so much",
    "can you open this"
])
def test_predict_text(payload):
    response = predict(payload)
    print(response)


def test_predict_malformed_json():
    response = predict('')
    print(response)


@pytest.mark.parametrize("payload", [
    "12345",
    "  ",
    "!@#$%",
    None
])
def test_predict_non_string_text(client, payload):
    print(f"Testing with payload: {payload}")
    response = predict(str(payload))
    print(f"Non-string input test: {response}")
