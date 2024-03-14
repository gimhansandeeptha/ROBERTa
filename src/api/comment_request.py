import requests
import json

def get_new_token():
    pass

def get_customer_comments(query_params=None):
    base_url = "https://wso2sndev.service-now.com/api/wso2/customer_health/get_customer_comments"
    payload = {}
    headers = {
  'Authorization': '', # token
  'Cookie': 'BIGipServerpool_wso2sndev=a54e6a419eb6227f51cefa66eeb46a63; JSESSIONID=C6A8A9CE55FC366667886C11829BCE56; glide_session_store=368A542F1B78C250264C997A234BCBC7; glide_user_activity=U0N2M18xOmxaVFd0TldxZll1S2wzWGllWHdGVFVDS1NDcXF3Y09nUkJ6aE9Lb0p5bkk9OlltaktKbm5GM2dVNm9McnB6VFpmcG1VbDlVUzhzY3pIcm5lQzFsMUt1Q1U9; glide_user_route=glide.7aba0aa922f21deed5cfb2b44291036d'
    }

    response = requests.get(base_url, headers=headers, data=payload, params=query_params)
    # print(response.text)
    print(json.dumps(response.json(), indent=4))

query_params = {
        "startDate": "2024-03-11",
        "endDate": "2024-03-11"
    }
get_customer_comments(query_params)
