import psycopg2
import logging


class DatabaseService(logging.Logger):
    # Define connection parameters
    DB_HOST = "ep-hidden-sound-d8ygik36.database.us-east-2.cloud.databricks.com"
    DB_NAME = "databricks_postgres"
    DB_USER = "prakhyatkarri@gmail.com"
    DB_PASS = "eyJraWQiOiI2NDZiZWZkNGY5NjYwMTdiNjk1MjRjOTRlMjcxNzljY2YyZmRlZDU1ZGJiMzQ5N2UwZjEwM2EwMzljZjI2ODU3IiwidHlwIjoiYXQrand0IiwiYWxnIjoiUlMyNTYifQ.eyJjbGllbnRfaWQiOiJkYXRhYnJpY2tzLXNlc3Npb24iLCJzY29wZSI6ImlhbS5jdXJyZW50LXVzZXI6cmVhZCBpYW0uZ3JvdXBzOnJlYWQgaWFtLnNlcnZpY2UtcHJpbmNpcGFsczpyZWFkIGlhbS51c2VyczpyZWFkIiwiaWRtIjoiRUFBWXJwdjYwclRZaGdFPSIsImlzcyI6Imh0dHBzOi8vZGJjLTYwZTgzZjc0LTczYzcuY2xvdWQuZGF0YWJyaWNrcy5jb20vb2lkYyIsImF1ZCI6Ijc0NzQ2NTk2NzM4MDEyNzkiLCJzdWIiOiJwcmFraHlhdGthcnJpQGdtYWlsLmNvbSIsImlhdCI6MTc4Mzg4NzI4MSwiZXhwIjoxNzgzODkwODgxLCJqdGkiOiI1NjI1Nzg1MS1mMTgxLTQyMjctYjY2MS0yMmFkMjgzODI2NzAifQ.CkZP7Oc8jFGkMvjOfDxxVqnaG9P1CZCf8wk2sZ-HNg-tW3o7ZG2U0031wJtTWXGhw0wif902H4WK9McjhcBmuiz29Y-t4OSu5-iJSwmt7qk38ID-RqbyBNA0zTb6XYAxQUDb-qJLhSOqiVRIm-V4J6xXtvMtB2HXaYwZMZiDt1wJq8M8zwvZ0UctYj9wRCpg3OgNLJv3xUR1nga5_yz2EVDc67Q7uom3Mt_fiFolxEXO-tejA9wEPLjCUhF-wrU_czP-K8NGPr1qmN-TCBbCiXq3EVp6o--UC8T02xX-0xKz31wCTd0E5BMTsXkeXBreJnQ0O2i3Cj1ympv53XOmWQ"
    DB_PORT = "5432"  # 5432 is the default PostgreSQL port

    connection = None

    def __init__(self, logger):
        self.logger = logger

    def get_connection(self):
        if self.connection:
            return self.connection

        from databricks.sdk import WorkspaceClient

        # Initialize the Databricks Workspace Client
        # (It automatically reads DATABRICKS_HOST, Token, or M2M environment variables)
        w = WorkspaceClient()

        # Get a temporary short-lived OAuth token for database authentication
        db_token = w.config.oauth_token().access_token
        # db_token = self.DB_PASS

        # Connection details
        # You can find these values in the "Connect" modal of your Lakebase Project UI
        db_host = self.DB_HOST
        db_user = self.DB_USER
        db_name = self.DB_NAME

        # Connect using standard PostgreSQL wire protocol
        conn = psycopg2.connect(
            host=db_host,
            port=5432,
            dbname=db_name,
            user=db_user,
            password=db_token,
            sslmode="require",
        )

        # Execute queries
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            print(cursor.fetchone())

        self.connection = conn

        return self.connection

    def close_connection(self):
        if self.connection:
            self.connection.close()

        return self.connection.closed

    def execute(self, query: str, commit: bool):
        if self.connection is None:
            self.get_connection()
        # Execute queries
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            if commit:
                self.connection.commit()
                return
            return cursor.fetchall()


# db_service = DatabaseService(logging.getLogger(__name__))
# connection = db_service.get_connection()
# results = db_service.execute("select * from playing_with_lakebase;")
# for i in results:
#     print(i)
# from databricks.sdk import WorkspaceClient

# Initialize the Databricks Workspace Client
# (It automatically reads DATABRICKS_HOST, Token, or M2M environment variables)
# print(WorkspaceClient().token_management.get("cc9e9369ed1e2ec1d1005f85d4c5bece9f64c1d4e4932a2015f9ecdfe41fa297"))
