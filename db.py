from sqlalchemy import create_engine
# db_engine = create_engine(f"postgresql+psycopg2://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")

db_user_name = 'hbch'
db_pwd = 'hbch2711'
db_host = '192.168.100.160'
port = 3306
db_name = 'RDYS_PUBLIC_TBS_WU'
db_engine = create_engine(f"mysql+pymysql://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")

from schema_engine import SchemaEngine

schema_engine = SchemaEngine(engine=db_engine, db_name=db_name)
mschema = schema_engine.mschema
mschema_str = mschema.to_mschema()
print(mschema_str)
mschema.save(f'./{db_name}.json')