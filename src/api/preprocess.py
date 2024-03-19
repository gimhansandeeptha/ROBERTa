from .comment_request import get_customer_comments

def extract_messages(response):
    ''' Return two dimentional array, fist dimention is the account name and the second dimention is liat of comments
        Both the dimentions can be empty.
    '''
    case_comments=[]

    try:
        data = response.json()
    except ValueError:
        raise ValueError("The object cannot be parsed")
    
    print(data)
    cases = data['result']['cases']
    for case in cases:
        values=[]
        account_name = case.get('account', None)
        if account_name==None:
            continue
        for comment in case.get('entries', []):
            value = comment.get('value', None)
            if value:
                values.append(value)
        
        case_comments.append([account_name,values])
    return case_comments

# query_params = {
#         "startDate": "2024-03-11",
#         "endDate": "2024-03-11"
#     }

# print(extract_messages(get_customer_comments(query_params)))
