from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
import uvicorn
from .servicenow.main import API

# Changing the order of following two imports leads to an error (dependency conflict) #check
from .model.roberta import RobertaClass
from .inference.main import get_sentiments

from .model.app import App
from .database.main import Database

from .api.main import router

# Replace the models file path in the models directory. 
robertaApp = App(metadata_path = "metadata\\roberta.json") # C:\\Users\\gimhanSandeeptha\\Gimhan Sandeeptha\\Sentiment Project\\ROBERTa\\metadata\\roberta.json
robertaApp.start_model()

@asynccontextmanager
async def lifespan(lifespan):
    print('app started...')
    schedular = BackgroundScheduler()
    schedular.add_job(func=process, trigger='cron', hour=10, minute=35, second=0)
    schedular.start()
    yield
    print("app stopped...")
    schedular.shutdown(wait=False)

def process():
    api = API()
    comments = api.get_comments()
    comments_with_sentiment = get_sentiments(comments)
    database = Database()
    database.insert(comments_with_sentiment)

app=FastAPI(lifespan=lifespan)
app.include_router(router)
if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
