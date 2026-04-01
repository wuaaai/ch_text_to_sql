# config.py

BASE_URL = "http://192.168.100.160:8991"

API_CONFIG = {
    "api_key": "sk-d507bd835e174d99b57757f3010dfd02",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "max_tokens": 8192,
    "context_length": 130000,
    "plat": "deepseek",
    "timeout": 60,
}

DB_USER_NAME = 'hbch'
DB_PWD = 'hbch2711'
DB_HOST = '192.168.100.160'
DB_PORT = 3306
DB_NAME = 'RDYS_PUBLIC_TBS_WU'

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 11

QDRANT_HOST = "192.168.100.160"
QDRANT_PORT = 6333

COLLECTION_NAME = "db_words_wu"