from fastapi import FastAPI
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
import uvicorn
from src.servicenow.main import API

# Changing the order of following two imports leads to an error (dependency conflict) #check
from src.model.roberta import RobertaClass
from src.inference.main import ModelPrediction
from src.model.app import App
from src.database.main import Database
from src.api.main import router
from src.preprocess.main import DataCleaner
from src.finetune.open_ai.main import APICall
from src.servicenow.data_object import SentimentData

# Replace the models file path in the models directory. 
robertaApp = App(metadata_path = "metadata\\roberta.json")
robertaApp.start_model()

@asynccontextmanager
async def lifespan(lifespan):
    print('app started...')
    schedular = BackgroundScheduler()
    schedular.add_job(func=process, trigger='cron', hour=11, minute=12, second=0)
    schedular.start()
    yield
    print("app stopped...")
    schedular.shutdown(wait=False)

def process():
    api = API()
    sentiment_data = SentimentData()
    api.get_comments(sentiment_data)

    data_cleaner = DataCleaner(sentiment_data)
    data_cleaner.clean()

    model_prediction = ModelPrediction()
    api_call = APICall()
    # model_prediction.get_sentiments(sentiment_data)
    loop = asyncio.new_event_loop()
    task1 = loop.create_task(model_prediction.get_sentiments(sentiment_data))
    task2 = loop.create_task(api_call.get_sentiments(sentiment_data))

    # Wait for both tasks to complete
    loop.run_until_complete(asyncio.gather(task2, task1))

    # loop.run_until_complete(model_prediction.get_sentiments(sentiment_data))
    print(sentiment_data.cases)
    # database = Database()
    # database.insert_cases(sentiment_data)

app=FastAPI(lifespan=lifespan)
app.include_router(router)
if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
