import requests
import pandas as pd

# Load Tapestry result
tapestry_result = pd.read_csv('/predictions/tapestry_result.csv').to_dict(orient='records')[0]

# SmartThings API credentials and endpoint
smartthings_endpoint = "https://api.smartthings.com/v1/devices/{device_id}/commands"
smartthings_token = "SMARTTHINGS_API_TOKEN"
device_id = "DEVICE_ID"

# Send result to SmartThings
headers = {
    "Authorization": f"Bearer {smartthings_token}",
    "Content-Type": "application/json"
}

data = {
    "commands": [
        {
            "component": "main",
            "capability": "custom.capability",
            "command": "setPrediction",
            "arguments": [tapestry_result]
        }
    ]
}

response = requests.post(smartthings_endpoint.format(device_id=device_id), json=data, headers=headers)

print(f"Response from SmartThings: {response.status_code} {response.text}")
