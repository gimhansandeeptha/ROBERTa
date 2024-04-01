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
    
    def insert_gpt_sentiment(self, gpt_sentiments:list):
        """ gpt sentiment is a list of dictionaries each conatain "text" and "sentiment" as keys
        """
        db=DatabaseConnection(hostname=hostname,
                              database=database,
                              username=username,
                              password=password
                              )
        db.connect()
        try:
            for comment in gpt_sentiments:
                text = comment.get("text")
                sentiment= comment.get("sentiment")
                insert_query = f"INSERT INTO gpt (text, sentiment) VALUES ('{text}','{sentiment}')"
                db.query(insert_query)
        finally:
            db.disconnect

    
    def insert_cases(self, cases:list):
        ''' Value is assumed to be a list with the following format: Incorrect now 
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
            for case in cases:
                case_id = case.get('case_id')
                account_name = case.get('account')
                sys_created_on = case.get('sys_created_on')
                entries = case.get('entries')

                insert_account = f"INSERT INTO account (case_id, sys_created_on, account_name) VALUES ('{case_id}','{sys_created_on}','{account_name}')"
                db.query(insert_account)
                # Insert into comment table
                for entry in entries:
                    comment = entry.get('value')
                    sentiment = entry.get('sentiment')
                    insert_comment = f"INSERT INTO comment (id, comment, sentiment, account_case_id) VALUES (NULL, '{comment}', '{sentiment}', '{case_id}')"
                    db.query(insert_comment)
            
        finally:
            db.disconnect

    def delete_all_gpt_entries(self):
        db=DatabaseConnection(hostname=hostname,
                              database=database,
                              username=username,
                              password=password
                              )
        db.connect()
        try:
            query_gpt="DELETE FROM gpt"
            db.query(query_gpt)
        finally:
            db.disconnect()

    def delete_all_cases(self):
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

    def get_gpt_entries(self):
        result =''
        query = f"""
                    SELECT 
                        text,
                        sentiment
                    FROM gpt
                """
        db=DatabaseConnection(hostname=hostname,
                              database=database,
                              username=username,
                              password=password
                              )
        db.connect()
        try:
            result = db.query(query)
        finally:
            db.disconnect()
        return result
        

    def get_cases_by_date(self, start_date, end_date):
        result = ''
        query_by_date = f"""
                            SELECT 
                                a.account_name, 
                                a.case_id, 
                                a.sys_created_on, 
                                c.comment, 
                                c.sentiment
                            FROM account a
                            JOIN comment c ON a.case_id = c.account_case_id
                            WHERE Date(sys_created_on) BETWEEN '{start_date}' AND '{end_date}';
                        """
        db=DatabaseConnection(hostname=hostname,
                              database=database,
                              username=username,
                              password=password
                              )
        db.connect()
        try:
            result = db.query(query_by_date)
        finally:
            db.disconnect()
        return result


# # Unit Testing
# response = [{'case_id': 'CS0431996', 'account': 'Test Customer Account', 'sys_created_on': '2024-03-11 02:16:41', 'entries': [{'value': 'Closed by contributor : Samitha Rathnayake (Intern)  ⓦ', 'created_on': '2024-03-11 02:17:05', 'sentiment': 'Positive'}, {'value': '<br><b> <u>Incident impact description</u> </b><br><p>Testing</p><br><b> <u>Description</u> </b><br><p></p><p>Test Description</p><p></p>', 'created_on': '2024-03-11 02:16:41', 'sentiment': 'Positive'}]}, {'case_id': 'CS0432040', 'account': 'DemoCloud', 'sys_created_on': '2024-03-11 08:41:08', 'entries': [{'value': 'Solution Accepted \n', 'created_on': '2024-03-11 08:45:05', 'sentiment': 'Positive'}, {'value': '<p>agent commented in here with different proposal,</p>', 'created_on': '2024-03-11 08:44:46', 'sentiment': 'Negative'}, {'value': 'Proposed Solution Rejected \nI prefer different solution...', 'created_on': '2024-03-11 08:44:18', 'sentiment': 'Negative'}, {'value': 'Case moved to Waiting on WSO2 \n', 'created_on': '2024-03-11 08:43:16', 'sentiment': 'Neutral'}, {'value': '<p>Customer commented here..</p>', 'created_on': '2024-03-11 08:43:05', 'sentiment': 'Neutral'}, {'value': '<br><b> <u>Description</u> </b><br><p></p><p>Test goes here</p><p></p>', 'created_on': '2024-03-11 08:41:08', 'sentiment': 'Neutral'}]}]

# db = Database()
# db.create()
# db.insert(response)
            
# # Unit testing : Delete
# db = Database()
# db.delete_all()
    
# # Unit testing: get_cases_by_date
# db = Database()
# result = db.get_cases_by_date("2024-03-19","2024-03-19")
# print(result)
