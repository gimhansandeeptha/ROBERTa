from .comment_request import get_customer_comments
from .query_param import get_query_param
from .servicenow_access import service_now_authorize, service_now_refresh_token
from .preprocess import extract_messages
# from .shared_data import SharedResource
import json

start = 0
page_size= 5

class API():
    def __init__(self) -> None:
        self.start = 0
        self.page_size = 5
        service_now_authorize() 

    def get_comments(self):
        processed_messages = []
        # shared_resource = SharedResource()
        while  self.start != "null":
            query_param = get_query_param(self.start, self.page_size)
            response = get_customer_comments(query_param)
            headers = response.headers
            self.start = headers.get("nextPageStart") 
            messages = extract_messages(response)
            processed_messages = processed_messages + messages
        # await shared_resource.set_data(processed_messages)
        # print(processed_messages)
        return processed_messages

