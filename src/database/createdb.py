from .connectdb import DatabaseConnection

class CreateDB():
    def __init__(self, hostname, database_name, username, password) -> None:
        self.database_name=database_name
        self.hostname=hostname
        self.username=username
        self.password=password

    def create(self):
        databaseConnection = DatabaseConnection(self.hostname,None, self.username, self.password)
        databaseConnection.connect()
        databaseConnection.query("CREATE DATABASE IF NOT EXISTS {}".format(self.database_name))
        databaseConnection.disconnect
    def create_schema(self):

        account_table = """
            CREATE TABLE IF NOT EXISTS account (
                case_id VARCHAR(16) PRIMARY KEY,
                account_name VARCHAR(50) NOT NULL
            )
        """
        comment_table = """
            CREATE TABLE IF NOT EXISTS comment (
                sys_id VARCHAR(50) PRIMARY KEY,
                comment VARCHAR(1024),
                sentiment VARCHAR(16),
                account_case_id VARCHAR(16),
                FOREIGN KEY (account_case_id) REFERENCES account(case_id)
            )
        """
        databaseConnection2 = DatabaseConnection(self.hostname,self.database_name, self.username, self.password)
        databaseConnection2.connect()

        databaseConnection2.query(account_table)
        databaseConnection2.query(comment_table)
        databaseConnection2.disconnect()