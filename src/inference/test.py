from fastapi import FastAPI, HTTPException, Request
from app import App
from model import RobertaClass

robertaApp = App()
roberta_model = RobertaClass()
roberta_model = robertaApp.start_model()

appapi = FastAPI()

@appapi.post("/send_message/{message_id}")
async def send_message(message_id: str, request: Request):
    data = await request.json()
    new_message = data.get("text")
    if not new_message:
        raise HTTPException(status_code=400, detail="Message not provided in request")

    prediction = robertaApp.predict([data])
    return {"status": "Message sent successfully", "prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(appapi, host="127.0.0.1", port=8000)
