from dotenv import load_dotenv
import os

load_dotenv(".env")

user_name = os.getenv("SERVICENOW_USERNAME")
print(user_name)
