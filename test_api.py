import requests

# Define API URL
url = "http://127.0.0.1:5000/predict"

# Sample test data
test_data = {"Sales": [1200], "Discount": [0.15], "Profit": [400]}

# Send POST request
response = requests.post(url, json=test_data)

# Print API response
print("Prediction Response:", response.json())
