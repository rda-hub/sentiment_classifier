import requests
resp = requests.post("http://localhost:8000/predict", json={"text": "I hate this so much"})
print(resp.json())
