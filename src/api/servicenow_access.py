from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
import asyncio

from dotenv import load_dotenv
import os
load_dotenv(".env")

app = FastAPI()

class TokenRequest(BaseModel):
    client_id: str
    client_secret: str
    username: str
    password: str

@app.post("/get_data")
async def get_data(token_request: TokenRequest, background_tasks: BackgroundTasks):
    # Define your token endpoint URL
    token_url = "https://wso2sndev.service-now.com/oauth_token.do"

    # Define the grant type
    grant_type = "password"

    # Define any additional parameters needed for token retrieval
    token_params = {
        "grant_type": grant_type,
        "client_id": token_request.client_id,
        "client_secret": token_request.client_secret,
        "username": token_request.username,
        "password": token_request.password
    }

    # Make a POST request to the token endpoint to get the access token
    token_response = requests.post(token_url, data=token_params)
    print(token_response.text)
    # Check if the request was successful
    if token_response.status_code == 200:
        # Extract the access token from the response
        access_token = token_response.json().get("access_token")

        if access_token:
            # Now you can use the access token to make authorized requests to the API
            # For example, make a GET request to your API endpoint
            # Replace "YOUR_API_ENDPOINT" with your actual API endpoint
            api_endpoint = "https://wso2sndev.service-now.com/api/wso2/customer_health/get_customer_comments?startDate=2024-03-11&endDate=2024-03-11"
            headers = {
                "Authorization": f"Bearer {access_token}"
            }
            api_response = requests.get(api_endpoint, headers=headers)

            # Return the API response
            return api_response.json()
        else:
            raise HTTPException(status_code=500, detail="Failed to obtain access token")
    else:
        raise HTTPException(status_code=token_response.status_code, detail=token_response.text)

# Run the get_data function once and wait for one minute before shutting down
async def run_get_data_once():
    # Define your client credentials
    client_id = os.getenv("CLIENT_ID")  # Replace with your client ID generated in ServiceNow
    client_secret = os.getenv("CLIENT_SECRET")  # Replace with your client secret generated in ServiceNow

    # Define your username and password
    username = os.getenv("SERVICENOW_USERNAME")  # Replace with your ServiceNow username
    password = os.getenv("SERVICENOW_PASSWORD")  # Replace with your ServiceNow password

    # Define the grant type
    grant_type = "password"

    # Define any additional parameters needed for token retrieval
    token_params = {
        "grant_type": grant_type,
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password
    }

    await get_data(TokenRequest(**token_params), BackgroundTasks())
    # Wait for one minute
    await asyncio.sleep(60)

# Start the server
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_get_data_once())

