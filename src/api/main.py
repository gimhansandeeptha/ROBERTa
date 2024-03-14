

from fastapi import FastAPI
import httpx

app = FastAPI()

@app.get("/get_data")
async def get_data_from_external_api():
    external_api_url = "https://example.com/api/data"  # Replace with the actual URL of the external API

    async with httpx.AsyncClient() as client:
        response = await client.get(external_api_url)

    if response.status_code == 200:
        data = response.json()
        return {"message": "Data retrieved successfully", "data": data}
    else:
        return {"message": f"Failed to retrieve data. Status code: {response.status_code}"}
