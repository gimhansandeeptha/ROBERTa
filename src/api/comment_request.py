import requests
from fastapi import HTTPException
import json
import os 
from dotenv import load_dotenv
from servicenow_access import service_now_refresh_token

base_url = "https://wso2sndev.service-now.com/api/wso2/customer_health/get_customer_comments"
environment_variable_file_path = ".env"
load_dotenv(environment_variable_file_path)

def request_customer_comments(query_params=None):
    # Retrieve access token and token type from environment variables
    access_token = os.getenv("ACCESS_TOKEN")
    token_type = os.getenv("TOKEN_TYPE")

    payload = {}
    headers = {
        'Authorization': f"{token_type} {access_token}",
    }
    #pagination 
    response = requests.get(base_url, headers=headers, data=payload, params=query_params)
    return response

def get_customer_comments(query_params:dict=None)->json:
    response = request_customer_comments(query_params)
    if response.status_code==200:
        pass
    elif response.status_code==401:
        service_now_refresh_token()
        response=request_customer_comments(query_params)
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response

query_params = {
        "startDate": "2024-03-11",
        "endDate": "2024-03-11"
    }

# comment = get_customer_comments(query_params)
# print(comment.json())
