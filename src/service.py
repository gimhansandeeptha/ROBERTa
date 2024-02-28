from fastapi import FastAPI, HTTPException, Request
from app import App
from model import RobertaClass
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Replace the models file path in the models directory. 
model_file_path = "D:\Gimhan Sandeeptha\Gimhan\Sentiment-Email\ROBERTa_production\models\pytorch_roberta_sentiment_3_classes_0.1.3.bin"
robertaApp = App(model_file_path)
roberta_model = RobertaClass()
roberta_model = robertaApp.start_model()

appapi = FastAPI()

# CORS configuration
appapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],  # Adjust this to the HTTP methods you want to allow
    allow_headers=["*"],  # Adjust this to the headers you want to allow
)

@appapi.post("/send_message/{message_id}")
async def send_message(message_id: str, request: Request):
    data = await request.json()
    new_message = data.get("message")
    if not new_message:
        raise HTTPException(status_code=400, detail="Message not provided in request")

    prediction = robertaApp.predict([new_message])[0]

    if prediction == 0:
        sentiment = 'Negative'
    elif prediction == 1:
        sentiment = 'Neutral'
    elif prediction == 2:
        sentiment = 'Positive'
    else:
        sentiment = 'unknown'
    
    return {"status": "Message sent successfully", "prediction": [sentiment]}

if __name__ == "__main__":
    uvicorn.run(appapi, host="127.0.0.1", port=8000)

