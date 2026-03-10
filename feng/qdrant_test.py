# pip install qdrant-client
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.http import models
import requests
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue


def get_embedding1(text_list):
    url = "http://192.168.100.160:8686/embed"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=text_list)
    return response.json()


# 连接本地 Qdrant
client = QdrantClient(host="192.168.100.160", port=6333)
def create1():
    # 创建一个名为 "articles" 的集合
    # vector_size=384 是所用嵌入模型的维度
    res = client.create_collection(
        collection_name="test_feng_table_words",
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    print(res)

def md5_word(word: str) -> str:
    return hashlib.md5(word.encode('utf-8')).hexdigest()


def create2():

    documents = [
        {"id": md5_word("向量数据库"), "text": "Qdrant 是一个用 Rust 写的向量数据库", "meta": {"topic": "tech", "views": 100}},
        {"id": md5_word("红烧肉"), "text": "如何制作美味的红烧肉", "meta": {"topic": "food", "views": 500}},
        {"id": md5_word("Rust"), "text": "Rust 语言的安全性特点", "meta": {"topic": "tech", "views": 200}},
    ]

    points = []
    for doc in documents:
        # 生成向量
        vector = get_embedding1([doc["text"]])['embeddings'][0]
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector=vector,
                payload=doc["meta"]  # 存入元数据
            )
        )

    client.upsert(collection_name="test_feng_table_words", points=points)



def create3():
    query_vector = get_embedding1(["Rust 编程"])['embeddings'][0]
    search_result = client.query_points(
               collection_name = "test_feng_table_words",
          query = query_vector,
        query_filter = models.Filter(
               must = [
                      models.FieldCondition(
                          key = "views",
                      range = models.Range(gte=150),
                  )
    ]
          ),
          limit = 2
      )

    for hit in search_result.points:
        print(hit)


def delete_collection():
    """删除集合"""
    result = client.delete_collection(collection_name="db_words_feng_260304")
    print(f"删除集合结果：{result}")
delete_collection()

def id_search(point_id,words,payload,collection_name):
    existing_points = client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=True,
        with_vectors=False  # 不需要向量，节省带宽
    )

    if not existing_points:
        point_to_upsert = PointStruct(
            id=point_id,
            vector= get_embedding1([words])['embeddings'][0],
            payload=payload
        )

    else:
        payload = existing_points[0].payload
        # 根据格式修改此处逻辑

        payload['views']+=300
        point_to_upsert = PointStruct(
            id=existing_points[0].id,
            vector= get_embedding1([words])['embeddings'][0],
            payload=payload
        )

    client.upsert(
        collection_name=collection_name,
        points=[point_to_upsert]
    )

def del_many():

    # 定义过滤条件：删除所有 cate='news' 且 words='old_topic' 的数据
    delete_filter = Filter(
        must=[
            FieldCondition(
                key="cate",
                match=MatchValue(value="news")
            ),
            FieldCondition(
                key="words",
                match=MatchValue(value="old_topic")
            )
        ]
    )

    # 执行条件删除
    response = client.delete_points(
        collection_name="my_collection",
        points_selector=delete_filter,  # 这里传入 Filter 对象而不是 ID 列表
        wait=True
    )

    print(f"删除状态: {response.status}")


# --构建表字典，标明表的关键字段，项目字段都是什么，合计在什么时候触发，默认单位？


# payload = {
#     "words": "文化旅游体育与传媒支出", # 唯一对应
#     "table_select":["十一、文化旅游体育与传媒支出",""], # 表中的检索项
#     "table":["xxxx_xxxx",""], # 所属表
#     "level":1,
#     "status":1
# }
# id_search(md5_word("红烧肉"),"红烧肉",{},"test_feng_table_words")
# 文化旅游





