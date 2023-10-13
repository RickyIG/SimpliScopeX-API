import requests

url = "https://github.com/RickyIG/SimpliScopeX-API/raw/main/final_simpliscopex_model.h5"
response = requests.get(url)
open("latest_model.h5", "wb").write(response.content)

print("Model is successfully pulled.")