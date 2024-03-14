from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
import asyncio

from dotenv import load_dotenv
import os
import tracemalloc
tracemalloc.start()

token_url = "https://wso2sndev.service-now.com/oauth_token.do"
environment_variable_file_path = ".env"

load_dotenv(environment_variable_file_path)

app = FastAPI()

class AuthorizationRequest(BaseModel):
    client_id: str
    client_secret: str
    username: str
    password: str

class TokenRequest(BaseModel):
    client_id: str
    client_secret: str
    access_token: str
    refresh_token: str


@app.post("/authorize")
async def authorize(authorization_request: AuthorizationRequest, background_tasks: BackgroundTasks):
    # Define any additional parameters needed for token retrieval
    grant_type = "password"
    authorization_params = {
        "grant_type": grant_type,
        "client_id": authorization_request.client_id,
        "client_secret": authorization_request.client_secret,
        "username": authorization_request.username,
        "password": authorization_request.password
    }

    # POST request to the token endpoint to get the access token
    authorization_response = requests.post(token_url, data=authorization_params)

    # Check if the request was successful
    if authorization_response.status_code == 200:
        # Extract the access token from the response
        access_token = authorization_response.json().get("access_token")
        refresh_token = authorization_response.json().get("refresh_token")
        
        if access_token and refresh_token:
            await update_env_variables({"ACCESS_TOKEN":access_token, "REFRESH_TOKEN":refresh_token}, environment_variable_file_path)
        else:
            raise HTTPException(status_code=500, detail="Failed to obtain access token & refersh token")
    else:
        raise HTTPException(status_code=authorization_response.status_code, detail=authorization_response.text)


@app.post("/reniew_token")
async def reniew_token(token_request: TokenRequest, background_tasks: BackgroundTasks):
    grant_type = "refresh_token"
    token_params = {
        "grant_type": grant_type,
        "client_id": token_request.client_id,
        "client_secret": token_request.client_secret,
        "access_token": token_request.access_token,
        "refresh_token": token_request.refresh_token
    }
    token_response = requests.post(token_url, data=token_params)
    # Check if the request was successful
    if token_response.status_code == 200:
        # Extract the access token from the response
        access_token = token_response.json().get("access_token")
        refresh_token = token_response.json().get("refresh_token")
        
        if access_token and refresh_token:
            await update_env_variables({"ACCESS_TOKEN":access_token, "REFRESH_TOKEN":refresh_token}, environment_variable_file_path)
        else:
            raise HTTPException(status_code=500, detail="Failed to obtain access token & refersh token")
    else:
        raise HTTPException(status_code=token_response.status_code, detail=token_response.text)
    
        
async def service_now_authorize():
    # Get the client credentials
    client_id = os.getenv("CLIENT_ID")  
    client_secret = os.getenv("CLIENT_SECRET")

    # Get username and password
    username = os.getenv("SERVICENOW_USERNAME")
    password = os.getenv("SERVICENOW_PASSWORD")  # Replace with your ServiceNow password

    # Additional parameters needed for token retrieval
    authorization_params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password
    }

    await authorize(AuthorizationRequest(**authorization_params), BackgroundTasks())
    await asyncio.sleep(60)

async def service_now_refresh_token():
    # Get the client credentials
    client_id = os.getenv("CLIENT_ID")  
    client_secret = os.getenv("CLIENT_SECRET")

    # Get access and referesh tokens (Assuming that access and refresh tokens have obtained earlier and they are avilable in environmental variables)
    access_token = os.getenv("ACCESS_TOKEN")
    referesh_token = os.getenv("REFRESH_TOKEN") 

    # Additional parameters needed for token retrieval
    token_params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "access_token": access_token,
        "refresh_token": referesh_token
    }

    await reniew_token(TokenRequest(**token_params), BackgroundTasks())
    await asyncio.sleep(60)

async def update_env_variables(env_variable_dict:dict, env_file_path:str):
    with open(env_file_path, 'r') as file:
        lines = file.readlines()
    
    with open(env_file_path, 'w') as file:
        for line in lines:
            key_val = line.strip().split('=')
            key = key_val[0]
            if key in env_variable_dict:
                file.write(f'{key}="{env_variable_dict[key]}"\n')
                del env_variable_dict[key]
            else:
                file.write(line)
        
        for key, val in env_variable_dict.items():
            file.write(f'{key}="{val}"')


# Start the server
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(service_now_refresh_token())
