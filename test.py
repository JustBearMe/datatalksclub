import requests

url = 'http://localhost:9696/predict'

data = {'url': 'https://static.insales-cdn.com/images/products/1/5430/67409206/large_mgm-750.jpg'}

result = requests.post(url, json=data).json()
print(result)