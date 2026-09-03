import duckdb

con = duckdb.connect()
con.sql("INSTALL postgres; LOAD postgres;")
con.sql("ATTACH 'postgresql://neondb_owner:npg_84pgDFkPURGO@ep-super-hall-b1t6ldgh-pooler.c-5.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require' AS pg (TYPE postgres)")

print(con.sql("SELECT count(*) FROM pg.raw__weather").df())
print(con.sql("SELECT count(*) FROM pg.raw__elpris").df())
print(con.sql("SELECT * FROM pg.raw__weather ORDER BY ingestion_timestamp DESC LIMIT 5").df())