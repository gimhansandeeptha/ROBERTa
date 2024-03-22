from ..model.app import App
from ..model.roberta import RobertaClass
from .preprocess import html_to_text, normalize_texts
# Replace the models file path in the models directory. 
robertaApp = App(metadata_path = "metadata\\roberta.json") # C:\\Users\\gimhanSandeeptha\\Gimhan Sandeeptha\\Sentiment Project\\ROBERTa\\metadata\\roberta.json
robertaApp.start_model()

def get_one_sentiment(text):
    prediction = robertaApp.predict([text])[0]
    sentiment = "unknown"
    if prediction == 0:
        sentiment = 'Negative'
    elif prediction == 1:
        sentiment = 'Neutral'
    elif prediction == 2:
        sentiment = 'Positive'
    return sentiment

def get_sentiments(cases):
    for case in cases:
        for entry in case.get('entries'):
            comment = entry.get('value')
            comment = normalize_texts(html_to_text(comment))
            sentiment = get_one_sentiment(comment)
            entry['value'] = comment
            entry['sentiment'] = sentiment
    return cases


# response_list =[{'case_id': 'CS0431996', 'account': 'Test Customer Account', 'sys_created_on': '2024-03-11 02:16:41', 'entries': [{'value': 'Closed by contributor : Samitha Rathnayake (Intern)  ⓦ', 'created_on': '2024-03-11 02:17:05'}, {'value': '<br><b> <u>Incident impact description</u> </b><br><p>Testing</p><br><b> <u>Description</u> </b><br><p></p><p>Test Description</p><p></p>', 'created_on': '2024-03-11 02:16:41'}]}, {'case_id': 'CS0432040', 'account': 'DemoCloud', 'sys_created_on': '2024-03-11 08:41:08', 'entries': [{'value': 'Solution Accepted \n', 'created_on': '2024-03-11 08:45:05'}, {'value': '<p>agent commented in here with different proposal,</p>', 'created_on': '2024-03-11 08:44:46'}, {'value': 'Proposed Solution Rejected \nI prefer different solution...', 'created_on': '2024-03-11 08:44:18'}, {'value': 'Case moved to Waiting on WSO2 \n', 'created_on': '2024-03-11 08:43:16'}, {'value': '<p>Customer commented here..</p>', 'created_on': '2024-03-11 08:43:05'}, {'value': '<br><b> <u>Description</u> </b><br><p></p><p>Test goes here</p><p></p>', 'created_on': '2024-03-11 08:41:08'}]}]
# print(get_sentiments(response_list))
