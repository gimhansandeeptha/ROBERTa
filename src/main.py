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

# Replace the models file path in the models directory. 
robertaApp = App(metadata_path = "metadata\\roberta.json") # C:\\Users\\gimhanSandeeptha\\Gimhan Sandeeptha\\Sentiment Project\\ROBERTa\\metadata\\roberta.json
robertaApp.start_model()

@asynccontextmanager
async def lifespan(lifespan):
    print('app started...')
    schedular = BackgroundScheduler()
    schedular.add_job(func=process, trigger='cron', hour=14, minute=52, second=0)
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
    # print(comments_with_sentiment)

app=FastAPI(lifespan=lifespan)
if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# response_list =[["Acc01",["Hi how are you?", "it is not a fair deal. i am dissapointed"]],["Acc2",["It is great keep it up", "Hi this is a normal day","can i have your pen for a moment"]]]
# print(get_sentiments(response_list))
    
