# dependencies.py
import json
import redis
from qdrant_client import QdrantClient
from openai import OpenAI
from sqlalchemy import create_engine
from core.matcher import KeywordMatcher
from cols_mapping import cols_mapping
from config import (
    API_CONFIG, DB_USER_NAME, DB_PWD, DB_HOST, DB_PORT, DB_NAME, 
    REDIS_HOST, REDIS_PORT, REDIS_DB, QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
)

class AppResources:
    def __init__(self):
        self.redis = None
        self.client = None
        self.chat = None
        self.db_engine = None
        self.public_data = {}
        self.use_table_info = {}
        self.words_match = None
        self.col_match = None
        self.zone_match = None
        self.zone_dict = {}
        self.collection_name = COLLECTION_NAME

# 单例模式
res = AppResources()

def init_resources():
    res.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, encoding="utf-8", decode_responses=True)
    res.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    res.chat = OpenAI(api_key=API_CONFIG["api_key"], base_url=API_CONFIG["base_url"], max_retries=2, timeout=API_CONFIG["timeout"])
    res.db_engine = create_engine(f"mysql+pymysql://{DB_USER_NAME}:{DB_PWD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    with open('text2sql_project/RDYS_PUBLIC_TBS_WU.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for x, y in data['tables'].items():
            res.public_data.update({y['comment']: {"db_id": data['db_id'], "schema": data['schema'], "tables": {x: data['tables'][x]}}})

    with open('text2sql_project/table_info.json', 'r', encoding='utf-8') as f4:
        res.use_table_info = json.load(f4)

    with open('text2sql_project/project_words.txt', 'r', encoding='utf-8') as f2:
        res.words_match = KeywordMatcher([x.strip() for x in f2.readlines()])

    # with open('text2sql_project/col_words.txt', 'r', encoding='utf-8') as f3:
    #     res.col_match = KeywordMatcher([x.strip() for x in f3.readlines()])
    res.col_match = KeywordMatcher(cols_mapping)

    res.zone_dict = {'河北省': '130000000', '河北省本级': '130000000', '邢台市': '130500000', '唐山市': '130200000', '邢台': '130500000', '南和区': '130506000', '曹妃甸区': '130209000', '滦州市': '130284000', '平乡县': '130532000', '柏乡县': '130524000', '迁西县': '130227000', '广宗县': '130531000', '巨鹿县': '130529000', '临西县': '130535000', '沙河市': '130582000', '遵化市': '130281000', '清河县': '130534000', '路北区': '130203000', '丰南区': '130207000', '路南区': '130202000', '丰润区': '130208000', '玉田县': '130229000', '南宫市': '130581000', '滦南县': '130224000', '临城县': '130522000'}
    res.zone_match = KeywordMatcher(list(res.zone_dict.keys()))

def close_resources():
    if res.redis:
        res.redis.close()

def get_res():
    return res