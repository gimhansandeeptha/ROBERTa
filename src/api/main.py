from .comment_request import get_customer_comments
from .query_param import get_query_param
from .servicenow_access import service_now_authorize, service_now_refresh_token
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
from .preprocess import extract_messages
import json

start = 0
page_size= 5

def get_comments():
    global start
    global page_size
    processed_messages = []
    while  start != "null":
        query_param = get_query_param(start, page_size)
        response = get_customer_comments(query_param)
        headers = response.headers
        start = headers.get("nextPageStart") 
        messages = extract_messages(response)
        processed_messages = processed_messages + messages
    print (processed_messages)

@asynccontextmanager
async def lifespan(lifespan):
    print('app started...')
    service_now_authorize()           # comment for testing
    schedular = BackgroundScheduler()
    schedular.add_job(func=get_comments, trigger='cron', hour=11, minute=24, second=0)
    schedular.start()
    yield
    print("app stopped...")
    schedular.shutdown(wait=False)
