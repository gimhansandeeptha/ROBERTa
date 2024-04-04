from src.model.app import App
from src.model.roberta import RobertaClass
from src.servicenow.data_object import SentimentData
# Replace the models file path in the models directory. 
# robertaApp = App(metadata_path = "metadata\\roberta.json") # C:\\Users\\gimhanSandeeptha\\Gimhan Sandeeptha\\Sentiment Project\\ROBERTa\\metadata\\roberta.json
# robertaApp.start_model()

class ModelPrediction:
    def __init__(self) -> None:
        self.roberta_app = App(metadata_path = "metadata\\roberta.json")
        self.roberta_app.start_model()
        
    def _get_one_sentiment(self, text):
        prediction = self.roberta_app.predict([text])[0]
        sentiment = "unknown"
        if prediction == 0:
            sentiment = 'Negative'
        elif prediction == 1:
            sentiment = 'Neutral'
        elif prediction == 2:
            sentiment = 'Positive'
        return sentiment

    async def get_sentiments(self, sentiment_data: SentimentData):
        for case in sentiment_data.cases:
            for entry in case.get('entries'):
                comment = entry.get('value')
                sentiment = self._get_one_sentiment(comment)
                async with sentiment_data.lock:
                    entry['sentiment'] = sentiment
        # return cases

# response_list =[{'case_id': 'CS0431996', 'account': 'Test Customer Account', 'sys_created_on': '2024-03-11 02:16:41', 'entries': [{'value': 'Closed by contributor : Samitha Rathnayake (Intern)  ⓦ', 'created_on': '2024-03-11 02:17:05'}, {'value': '<br><b> <u>Incident impact description</u> </b><br><p>Testing</p><br><b> <u>Description</u> </b><br><p></p><p>Test Description</p><p></p>', 'created_on': '2024-03-11 02:16:41'}]}, {'case_id': 'CS0432040', 'account': 'DemoCloud', 'sys_created_on': '2024-03-11 08:41:08', 'entries': [{'value': 'Solution Accepted \n', 'created_on': '2024-03-11 08:45:05'}, {'value': '<p>agent commented in here with different proposal,</p>', 'created_on': '2024-03-11 08:44:46'}, {'value': 'Proposed Solution Rejected \nI prefer different solution...', 'created_on': '2024-03-11 08:44:18'}, {'value': 'Case moved to Waiting on WSO2 \n', 'created_on': '2024-03-11 08:43:16'}, {'value': '<p>Customer commented here..</p>', 'created_on': '2024-03-11 08:43:05'}, {'value': '<br><b> <u>Description</u> </b><br><p></p><p>Test goes here</p><p></p>', 'created_on': '2024-03-11 08:41:08'}]}]
# print(get_sentiments(response_list))
