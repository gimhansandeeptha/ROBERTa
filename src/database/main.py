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
    
    def insert(self, value:list):
        ''' Value is assumed to be a list with the following format:
        value = [["case_id 01", "Account_name1",["comment 01","comment 02", ...],["sentiment 01","sentiment 2", ...]], 
                 ["case_id 02", "Account_name2",["comment 01","comment 02", ...],["sentiment 01","sentiment 2", ...]],... 
                ]
        '''
        db=DatabaseConnection(hostname=hostname,
                              database=database,
                              username=username,
                              password=password
                              )
        db.connect()
        try:
            for item in value:
                case_id, account_name, comments, sentiments = item

                insert_account = f"INSERT INTO account (case_id, account_name) VALUES ('{case_id}','{account_name}')"
                db.query(insert_account)
                # Insert into comment table
                for comment, sentiment in zip(comments, sentiments):
                    insert_comment = f"INSERT INTO comment (id, comment, sentiment, account_case_id) VALUES (NULL, '{comment}', '{sentiment}', '{case_id}')"
                    db.query(insert_comment)
            
        finally:
            db.disconnect

    def delete_all(self):
        db=DatabaseConnection(hostname=hostname,
                              database=database,
                              username=username,
                              password=password
                              )
        db.connect()
        try:
            query_account = "DELETE FROM account"
            query_comment = "DELETE FROM comment"
            db.query(query_comment)
            db.query(query_account)
        finally:
            db.disconnect()


# # Unit Testing
# response = [
#     ['CS0432550', 'ParaTestAccount6', ['Solution Accepted \n', 'Proposed Solution Rejected \nI prefer different solution...', '<p>Customer commented here..</p>', '<br><b> <u>Description</u> </b><br><p></p><p>Test Description goes here</p><p></p>'], ['Positive', 'Negative', 'Neutral', 'Neutral']], 
#     ['CS0432551', 'ParaTestAccount6', ['Closed by customer.', '<br><b> <u>Description</u> </b><br><p>-- This is the previous description (Edit or Delete if you want to alter) --</p><p>Test Description goes here</p><p></p>'], ['Neutral', 'Neutral']], 
#     ['CS0432552', 'ParaTestAccount6', ['Solution Accepted \n', 'Proposed Solution Rejected \nI prefer different solution...', '<p>Customer commented here..</p>', '<br><b> <u>Description</u> </b><br><p></p><p>Test Description goes here</p><p></p>'], ['Positive', 'Negative', 'Neutral', 'Neutral']], 
#     ['CS0432553', 'ParaTestAccount6', ['Closed by customer.', '<br><b> <u>Description</u> </b><br><p>-- This is the previous description (Edit or Delete if you want to alter) --</p><p>Test Description goes here</p><p></p>'], ['Neutral', 'Neutral']], 
#     ['CS0432555', 'DemoCloud', ['Solution Accepted \n', 'Proposed Solution Rejected \nI prefer different solution...', '<p>agent set notes here with solutions configured</p>', '<p>agent commented in here with different proposal,</p>', 'Case moved to Waiting on WSO2 \n', '<p>Customer commented here..</p>', '<br><b> <u>Description</u> </b><br><p></p><p>Test goes here</p><p></p>'], ['Positive', 'Negative', 'Neutral', 'Negative', 'Neutral', 'Neutral', 'Neutral']]
#     ]

# db = Database()
# db.create()
# db.insert(response)
            
# # Unit testing : Delete
# db = Database()
# db.delete_all()
