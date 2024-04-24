from src.servicenow.comment_request import get_customer_comments
from src.servicenow.query_param import get_query_param
from src.servicenow.servicenow_access import service_now_authorize, service_now_refresh_token
from src.servicenow.preprocess import extract_messages
from src.utils.data_model import ServicenowData
# from .shared_data import SharedResource

start = 0
page_size= 5

class API():
    def __init__(self) -> None:
        self.start = 0
        self.page_size = 5
        service_now_authorize() 

    def get_comments(self, sn_data: ServicenowData) -> ServicenowData:
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
        sn_data.load_data(processed_messages)
        return sn_data


# api = API()
# sn_data = ServicenowData()
# a=api.get_comments(sn_data)

# sn_data.reset_params()
# while sn_data.next_case():
#     while sn_data.next_comment():
#         print(sn_data.get_case_id(), sn_data.get_account(), sn_data.get_comment())