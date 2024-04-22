from fastapi import FastAPI
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
import uvicorn
from src.servicenow.main import API
import pickle

# Changing the order of following two imports leads to an error (dependency conflict) #check
from src.model.roberta import RobertaClass
from src.inference.main import ModelPrediction
from src.model.app import App
from src.database.main import Database
from src.api.main import router
from src.preprocess.main import DataCleaner
from src.finetune.open_ai.main import APICall
from src.servicenow.data_object import SentimentData
from src.finetune.main import Handler

# Replace the models file path in the models directory. 
# robertaApp = App(metadata_path = "metadata\\roberta.json")
# robertaApp.start_model()

@asynccontextmanager
async def lifespan(lifespan):
    print('app started...')
    schedular = BackgroundScheduler()

    # ---------------------- For testing ---------------------------------
    from datetime import datetime, timedelta
    time = datetime.now()+timedelta(seconds=30)
    hour = time.hour
    minute = time.minute
    second = time.second
    # --------------------------------------------------------------------

    schedular.add_job(func=process, trigger='cron', hour=hour, minute=minute, second=second)
    schedular.add_job(func=finetune, trigger='cron', hour=hour, minute=minute+2, second=second)
    schedular.start()
    yield
    print("app stopped...")
    schedular.shutdown(wait=False)

def process():
    '''Run periodically with following tasks:

    * Fetch comment from the service-now
    * Clean the data 
    * predict the sentiment by local model
    * Predict the sentiment by GPT
    * store the local model results in the database
    * store the GPT sentiment in the database
    '''
    # from src.test import get_mock_data
    # sentiment_data: SentimentData = get_mock_data()

    # ----------------------------- the following part should be uncommented in actual setting ------------------------
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
    # -------------------------------------------------------------------------------------------------------------------

    database = Database()
    database.insert_cases(sentiment_data)

    # Insert GPT sentiments to the database
    database.insert_gpt_sentiment(sentiment_data)


def finetune():
    handler = Handler()
    handler.finetune()

app=FastAPI(lifespan=lifespan)
app.include_router(router)
if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
