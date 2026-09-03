import os
import duckdb
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("DATABASE_URL")

if not db_url:
    print("Fel: DATABASE_URL saknas i .env")
    exit(1)

con = duckdb.connect()
con.sql("INSTALL postgres; LOAD postgres;")
con.sql(f"ATTACH '{db_url}' AS pg (TYPE postgres);")

print("Ansluten till molndatabasen via DuckDB!")
print("Skriv valfritt SQL-kommando mot tabeller i 'pg' (t.ex. SELECT * FROM pg.stg__elpris LIMIT 5;).")
print("Skriv 'exit' eller 'quit' för att avsluta.\n")

while True:
    try:
        query = input("sql> ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break
        
        con.sql(query).show()
    except Exception as e:
        print(f"Fel: {e}\n")