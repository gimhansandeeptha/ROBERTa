## The logic to fetch the data from GPT and store them to the database/delete data ans all that(gpt table)
## also delete data if want 
from src.servicenow.data_object import SentimentData
from src.open_ai.gpt import GPT
class APICall:
    def __init__(self) -> None:
        self.gpt = GPT()

    async def _get_one_sentiment(self,comment):
        return await self.gpt.get_response(comment)

    async def get_sentiments(self, sentiment_data: SentimentData):
        for case in sentiment_data.cases:
            for entry in case.get('entries'):
                comment = entry.get('value')
                sentiment = await self._get_one_sentiment(comment)
                async with sentiment_data.lock:
                    entry['gpt_sentiment'] = sentiment

