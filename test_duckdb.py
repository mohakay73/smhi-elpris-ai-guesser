import os
import duckdb
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("DATABASE_URL")

con = duckdb.connect()
con.sql("INSTALL postgres; LOAD postgres;")
con.sql(f"ATTACH '{db_url}' AS pg (TYPE postgres);")

print("\n--- 5 rader från stg__elpris ---")
con.sql("SELECT starts_at, sek_per_kwh, eur_per_kwh FROM pg.stg__elpris LIMIT 5;").show()

print("\n--- 5 rader från stg__weather ---")
con.sql("SELECT station, parameter, observed_at, value, quality FROM pg.stg__weather LIMIT 5;").show()

print("\n--- Total radmängd i staging vs raw ---")
con.sql("""
    SELECT 
        (SELECT count(*) FROM pg.raw__elpris) AS raw_elpris_days,
        (SELECT count(*) FROM pg.stg__elpris) AS stg_elpris_intervals,
        (SELECT count(*) FROM pg.raw__weather) AS raw_weather_rows,
        (SELECT count(*) FROM pg.stg__weather) AS stg_weather_rows;
""").show()