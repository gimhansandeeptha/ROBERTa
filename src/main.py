from fastapi import FastAPI
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
import uvicorn
from src.servicenow.main import API
import pickle
from src.utils.data_model import ServicenowData, DatabaseData
# Changing the order of following two imports leads to an error (dependency conflict) #check

from src.model.main import ModelProcess

from src.database.main import Database
from src.api.main import router
from src.preprocess.main import DataCleaner
from src.open_ai.main import APICall

# Replace the models file path in the models directory. 

@asynccontextmanager
async def lifespan(lifespan):
    print('app started...')
    schedular = BackgroundScheduler()

    # ---------------------- For testing ---------------------------------
    from datetime import datetime, timedelta
    time = datetime.now()+timedelta(seconds=15)
    hour = time.hour
    minute = time.minute
    second = time.second
    # --------------------------------------------------------------------

    # schedular.add_job(func=process, trigger="cron", hour=hour, minute=minute, second=second)
    schedular.add_job(func=finetune, trigger="cron", hour=hour, minute=minute, second=second)
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
    sn_data = ServicenowData()
    api.get_comments(sn_data)

    data_cleaner = DataCleaner()
    data_cleaner.clean(sn_data)

    model_process= ModelProcess()
    api_call = APICall()

    loop = asyncio.new_event_loop()
    inference_task = loop.create_task(model_process.inference_process(sn_data))
    gpt_prediction_task = loop.create_task(api_call.set_gpt_sentiments(sn_data))
    loop.run_until_complete(asyncio.gather(inference_task, gpt_prediction_task))

    # loop = asyncio.new_event_loop()
    # task1 = loop.create_task(model_prediction.get_sentiments(sentiment_data))
    # task2 = loop.create_task(api_call.get_sentiments(sentiment_data))

    # # Wait for both tasks to complete
    # loop.run_until_complete(asyncio.gather(task2, task1))
    # -------------------------------------------------------------------------------------------------------------------

    database = Database()
    database.insert_cases(sn_data)

    # Insert GPT sentiments to the database
    database.insert_gpt_sentiment(sn_data)

# from src.finetune.main import Handler

def finetune():
    database = Database()
    gpt_entries = database.get_gpt_entries(count=10) # Use large value e.g. 500
    print(gpt_entries.head(30))

    model_process = ModelProcess()
    model_process.finetune_process(gpt_entries)

app=FastAPI(lifespan=lifespan)
app.include_router(router)
if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
