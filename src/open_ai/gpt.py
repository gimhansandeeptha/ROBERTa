# API call implementation to the GPT
from openai import OpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import asyncio
import time

class GPT:
    def __init__(self) -> None:
        self.client = OpenAI(api_key="") 
        self.instruction = """You will be provided with a text, and your task is to classify its sentiment as Positive, Neutral, or Negative.
        only give either Positive, Negative or Neutral"""
        self.model = "gpt-3.5-turbo"
        self.instruction_role = "system"
        self.user_role = "user"
        self.temperature = 0.7
        self.max_tokens = 64
        self.top_p = 1

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_response(self, text:str):
        try:
            response = self.client.chat.completions.create(
                    model = self.model,
                    messages = [
                        {
                            "role" : self.instruction_role,
                            "content" : self.instruction
                        },
                        {
                            "role" : self.user_role,
                            "content" : text
                        }
                    ],
                    temperature = self.temperature,
                    max_tokens = self.max_tokens,
                    top_p = self.top_p
            )
            return response.choices[0].message.content
        
        except RateLimitError as e:
            print("Rate limit exceeded. Please try again later.")
            return ""
        except Exception as e:
            print(f"An error has occured when calling the openai api: {e}")
            return ""
        # import random
        # sentiments = ['Negative', 'Positive', 'Neutral']
        # return random.choice(sentiments)

# Unit Testing
# text = "The case can be closed now. it is success and resolved. Thank you for your solution"
# gpt = GPT()
# sentiment = gpt.get_response(text)
# print(sentiment)
