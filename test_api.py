import requests
import json

# API endpoint - note the trailing slash
url = "http://127.0.0.1:8000/api/summarize/"

# Test data
data = {
    "text": """Introduction
This privacy policy ("Privacy Policy") applies to all visitors and users of the Cursor desktop app and websites (collectively, "Cursor," "App" or "Apps"), which are offered by Anysphere Inc. and/or any of its affiliates ("Anysphere" or "we" or "us") and describes how we process your personal information in connection with those Apps, and how we collect information through the use of cookies and related technologies. It also tells you how you can access and update your personal information and describes the data protection rights that may be available under your country's or state's laws. Please read this Privacy Policy carefully. By accessing or using any part of the App, you acknowledge you have been informed of and consent to our practices with regard to your personal information and data.

Applicability of this Privacy Policy
If you are a customer of Anysphere, this Privacy Policy does not apply to personal information or other data and information that we process on your behalf (if any) as your service provider (collectively, "Customer Data"). Our use of your Customer Data shall instead be governed by the terms and conditions of the separate customer agreement or terms of service that you have agreed to with us."""
}

# Send POST request
print("Sending request to API...")
response = requests.post(url, json=data)

# Print results
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")