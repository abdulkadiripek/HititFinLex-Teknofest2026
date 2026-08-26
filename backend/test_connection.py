import os

import psycopg
from dotenv import load_dotenv


load_dotenv()

connection = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

with connection.cursor() as cursor:
    cursor.execute(
        "SELECT current_database(), current_user, version();"
    )
    database_name, database_user, database_version = cursor.fetchone()

print("Database connection successful")
print("Database:", database_name)
print("User:", database_user)
print("Version:", database_version)

connection.close()