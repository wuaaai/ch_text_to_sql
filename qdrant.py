from qdrant_client import QdrantClient

# 连接到你的 Qdrant 服务器
client = QdrantClient("http://192.168.100.160:6333")

# 1. 查看有哪些集合
collections = client.get_collections()
print("当前集合列表:")
for collection in collections.collections:
    print(f"- {collection.name}")

# 2. 查看特定集合的内容（提取前 5 条数据）
collection_name = "db_words_wu"
records, next_page_offset = client.scroll(
    collection_name=collection_name,
    limit=100,
    with_payload=True,
    with_vectors=True # 若需查看向量数据，改为 True
)
print(f"\n[{collection_name}] 的前 100 条数据 Payload:")
# for record in records:
#     print(record.payload)
# print(f"\n[{collection_name}] 的前 30 条数据")
# for record in records:
#     print(record)

# client.delete_collection(collection_name=collection_name)
# print(f"已删除集合: {collection_name}")