from .connectdb import DatabaseConnection
from .createdb import CreateDB
from dotenv import load_dotenv
import os

environment_variable_file_path = ".\\src\\database\\.env"
load_dotenv(environment_variable_file_path)

username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")
hostname = 'localhost'
username = 'root'
database = 'sentiment'

class Database():
    def __init__(self) -> None:
        pass

    def create(self):
        db=CreateDB(hostname=hostname,
                    database_name=database,
                    username=username,
                    password=password
                    )
        db.create()
        db.create_schema()
        return 
    
    def insert(self, value):
        db=DatabaseConnection(hostname=hostname,
                              database=database,
                              username=username,
                              password=password
                              )
        db.query()
        
