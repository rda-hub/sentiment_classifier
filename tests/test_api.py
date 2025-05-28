def test_positive(client):
    response = client.post("/predict", json={"text": "I love this movie!"})
    assert response.status_code == 200
    data = response.json()
    score = data["score"]
    inference_time = data["inference_time"]
    label = data["label"]
    print(f"Label: {label}, Score: {score:.4f}, Inference Time: {inference_time:.4f}s")

    assert score > 0.90
    assert inference_time < 1.0
    assert data["label"] == "POSITIVE"


def test_negative(client):
    response = client.post("/predict", json={"text": "I hate this so much"})
    assert response.status_code == 200
    data = response.json()
    score = data["score"]
    inference_time = data["inference_time"]
    label = data["label"]
    print(f"Label: {label}, Score: {score:.4f}, Inference Time: {inference_time:.4f}s")

    assert score > 0.90
    assert inference_time < 1.0
    assert data["label"] == "NEGATIVE"


def test_predict_missing_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200
    data = response.json()
    print(f"Empty input test: Label={data['label']}, Score={data['score']:.4f}")
    assert data['score'] < 0.8