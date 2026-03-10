import hashlib
import json
import uuid

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sqlalchemy import create_engine, text
from qdrant_client.http import models

def json_run():
    with open('RDYS_PUBLIC_TBS.json', 'r',encoding='utf-8') as f:
        sql = json.loads(f.read())
        txt = ""
        table_info = {}
        name_dict = {}
        for x,y in sql['tables'].items():
            txt+=(y['comment']+x+'\n')
            middle = []
            name_dict.update({x:y['comment']})
            for k,v in y['fields'].items():
                middle.append(k)
                txt+=f" {k}:{v['comment']}"
            table_info.update({x:middle})
            txt+='\n'
        # print(txt)
        print(name_dict)
        # for k,v in table_info.items():
        #     if "XM_DW" in v :
        #         print(f"{k}-{name_dict[k]}")

        # print(table_info)
        # ttt = list(table_info.values())
        # result = list(set(ttt[0]).intersection(*map(set, ttt[1:])))
        # print(result)

def schema_run(table_name):
    with open('RDYS_PUBLIC_TBS_new.json', 'r', encoding='utf-8') as f:
        sql = json.loads(f.read())
        data = sql['tables'][table_name]
        print(f"表名：{data['comment']}")
        for k,v in data['fields'].items():
            print(f"字段名：{k}-{v['comment']}")


def get_embedding1(text_list):
    url = "http://192.168.100.160:8686/embed"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=text_list)
    return response.json()

# 表分类3：国有资产监督，经济监督，预算执行
# 每个分类中按照列进行单位映射，经济监督默认有单位字段不用考虑
# --构建表字典，标明表的关键字段，项目字段都是什么(是科目km还是xm)，合计在什么时候触发，默认单位？

name_dict = {'RDYS_LD_GZJD_FJRQY_FQYB': '国有资产监督—非金融企业-日常报告-日常监督-03省国资委监管企业（月度汇总）',
             'RDYS_LD_JJJD_JDYX_CZ': '经济监督-监督运行-财政',
             'RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC': '预算审查监督-预算执行表-全省一般公共预算收入完成情况表',
             'RDYS_LD_YSZX_BJSHBXJJYUZCWC': '预算执行-本级社会保险基金预算支出完成情况表',
             'RDYS_LD_YSZX_YBGGYS_SZPMQK': '预算执行-一般公共预算收支排名情况',
             'RDYS_LD_YSZX_GYZBJYYSSZ': '预算执行-国有资本经营预算收支情况简表',
             'RDYS_LD_YSZX_YBGGYS_DQZCWCQK': '预算执行-全省一般公共预算地区支出完成情况表',
             'RDYS_LD_YSZX_SHBXJJYSSZ': '预算执行-社会保险基金预算收支情况简表',
             'RDYS_LD_GZJD_FJRQY_FDQB': '国有资产监督—非金融-日常报告-日常监督-04各市指标表',
             'RDYS_LD_JJJD_JDYX_NY': '经济监督-季度运行-农业指标',
             'RDYS_LD_YSZX_SHBXJJYSDQSRWC': '预算执行-社会保险基金预算地区收入完成情况表',
             'RDYS_LD_GZJD_FJRQY_FHYB': '国有资产监督—非金融企业-日常报告-日常监督-05国有企业分行业表',
             'RDYS_LD_JJJD_JDYX_GYXY': '经济监督-季度运行-工业效益',
             'RDYS_LD_YSZX_BJGYZBJYYSZCWC': '预算执行-本级国有资本经营预算支出完成情况表',
             'RDYS_LD_YSZX_SHBXJJYSZCWC': '预算执行-社会保险基金预算支出完成情况表',
             'RDYS_LD_YSZX_ZFXJJ_QSSRWCQK': '预算执行-全省政府性基金预算收入完成情况表',
             'RDYS_LD_YSZX_SHBXJJYSSRWC': '预算执行-社会保险基金预算收入完成情况表',
             'RDYS_LD_YSZX_SHBXJJYSDQZCWC': '预算执行-社会保险基金预算地区支出完成情况表',
             'RDYS_LD_JJJD_JDYX_GNMY': '经济监督-季度运行-国内贸易',
             'RDYS_LD_JJJD_JDYX_WLNH': '经济监督-监督运行-物流能耗',
             'RDYS_LD_GZJD_FJRQY_QYCB': '国有资产监督—非金融企业-日常报告-日常监督-01&02企业财务快报',
             'RDYS_LD_JJJD_JDYX_GYSC': '经济监督-季度运行-工业生产',
             'RDYS_LD_JJJD_JDYX_HS': '经济监督-监督运行-国民经济核算',
             'RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC': '预算审查监督-预算执行表-全省一般公共预算支出完成情况表',
             'RDYS_LD_JJJD_JDYX_FWY': '经济监督-季度运行-规模以上服务业',
             'RDYS_LD_YSZX_YBGGYS_SBJZCWCQK': '预算执行-省本级一般公共预算支出完成情况表',
             'RDYS_LD_YSZX_ZFXJJ_SZQKJB': '预算执行-政府性基金预算收支情况简表',
             'RDYS_LD_JJJD_JDYX_LYJY': '经济监督-监督运行-旅游就业',
             'RDYS_LD_YSZX_BJGYZBJYYSSRWC': '预算执行-本级国有资本经营预算收入完成情况表',
             'RDYS_LD_JJJD_JDYX_JCK': '经济监督-监督运行-进出口',
             'RDYS_LD_YSZX_ZFXJJ_SBJSRWCQK': '预算执行-省本级政府性基金预算收入完成情况表',
             'RDYS_LD_JJJD_JDYX_GDZCTZ': '经济监督-季度运行-固定资产投资',
             'RDYS_LD_YSZX_GYZBJYYYZCWC': '预算执行-国有资本经营预算支出完成情况表',
             'RDYS_LD_YSZX_GYZBJYYSDQZCWC': '预算执行-国有资本经营预算地区支出完成情况表',
             'RDYS_LD_JJJD_JDYX_JR': '经济监督-监督运行-金融保险',
             'RDYS_LD_JJJD_JDYX_RMSH': '经济监督-监督运行-人民生活',
             'RDYS_LD_YSZX_BJSHBXJJYUSRWC': '预算执行-本级社会保险基金预算收入完成情况表',
             'RDYS_LD_YSZX_GYZBJYYYSRWC': '预算执行-国有资本经营预算收入完成情况表',
             'RDYS_LD_YSSC_BNYS_QSYBGGDQSRWC': '预算审查监督-预算执行表-全省一般公共预算地区收入完成情况表',
             'RDYS_LD_YSZX_GYZBJYYSDQSRWC': '预算执行-国有资本经营预算地区收入完成情况表',
             'RDYS_LD_YSZX_ZFXJJ_SBJZCWCQK': '预算执行-本级政府性基金预算支出完成情况表',
             'RDYS_LD_JJJD_JDYX_JG': '经济监督-监督运行-价格', 'RDYS_LD_JJJD_JDYX_XXZB': '经济监督-季度运行-先行指标'}

table_info = {
    "RDYS_LD_GZJD_FJRQY_FQYB":{
        "key_words":[], # 关键字段用于用户问题命中此表但是没提及查什么的补充
        "project_key":[], # 项目字段用于用户问题优化时使用项目名称，项目code,科目名称，科目code的映射
        "project_name":[], # 对key的释义用于问题补充时使用
        "is_hj":0, # 科目或项目是否包含合计字段
        "is_sheng":0, # 是否为省本级，默认0，代表全省
        "unit":{"万元":[],"%":[]}, # 对于所有字段单位的描述，用于问题生成时的具体描述
        "cate":"预算执行", # 表分类 已知经济监督表中包含单位字段，其他表好像不包含单位字段
        "zh_table_name":"中文表名"
    },
}


def table_info_create():
    """生成表描述"""
    with open('RDYS_PUBLIC_TBS.json', 'r', encoding='utf-8') as f:
        sql = json.loads(f.read())
        table_info = {}
        tables = ["RDYS_LD_YSZX_YBGGYS_SBJZCWCQK",
                  "RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC",
                  "RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC",
                  "RDYS_LD_YSZX_ZFXJJ_QSSRWCQK",
                  ]
        db_user_name = 'hbch'
        db_pwd = 'hbch2711'
        db_host = '192.168.100.160'
        port = 3306
        db_name = 'RDYS_PUBLIC_TBS'
        db_engine = create_engine(f"mysql+pymysql://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")
        # 使用 SQLAlchemy 执行 SQL 查询
        char_list = ['数', "%", "金", "额", "比"]
        with db_engine.connect() as connection:
            # for x, y in sql['tables'].items():
            for x in tables:
                y = sql['tables'][x]
                table_name = y['comment']
                use_key = []
                project_key = []
                project_name = []
                wan = {}
                percent = {}
                # 国有资产监督，经济监督
                if table_name.startswith('国有资产监督'):
                    pass
                elif table_name.startswith('经济监督'):
                    pass
                else:
                    cate = "预算执行"
                    # 预算执行中单位有两种万元和百分比
                    for k, v in y['fields'].items():
                        # k字段名 v字段信息
                        if "编码" in v['comment'] and v['comment'] not in ["区划编码","部门编码"]:
                            project_key.append(v['comment'])
                            use_key.append(k)
                        if "名称" in v['comment'] and v['comment'] not in ["区划名称","部门名称"]:
                            project_name.append(v['comment'])
                        if "DECIMAL" in  v['type'] and any(char in v['comment'] for char in char_list):
                            # 此逻辑筛选单位字段
                            ""
                            if any(v['comment'].endswith(u) for u in ["数","额"]):
                                wan.update({k:v['comment']})
                            else:
                                percent.update({k:v['comment']})

                    if not project_key and "XM_CODE" in y['fields']:
                        project_key.append(y['fields']["XM_CODE"]['comment'])
                        use_key.append("XM_CODE")
                    if not project_name and "XM_NAME" in y['fields']:
                        project_name.append(y['fields']["XM_NAME"]['comment'])

                    is_hj = 0
                    is_sheng = 1 if any(char in table_name for char in ['省级',"本级"]) else 0
                    for n,w in enumerate(use_key):
                        sql1 = f"""select DISTINCT {w} from  {x}"""
                        result = connection.execute(text(sql1))
                        for d in result.all():
                            if d[0]:
                                val = d[0].strip()
                                if " " in val:
                                    val = val.replace(" ", "")
                                if val == "合计":  # 总数 总额 合计 | 修改数据
                                    is_hj = 1
                                    break
                        if is_hj:
                            break
                    table_info.update({table_name: {"key_words": [],
                                           "project_key": project_key,
                                           "is_sheng": is_sheng,
                                           "project_name": project_name,
                                           "is_hj": is_hj,
                                           "unit": {"万元": wan, "百分比": percent},
                                           "cate": cate,
                                           "table": x}})



        print(table_info)


def sql_run(table):
    db_user_name = 'hbch'
    db_pwd = 'hbch2711'
    db_host = '192.168.100.160'
    port = 3306
    db_name = 'RDYS_PUBLIC_TBS'
    db_engine = create_engine(f"mysql+pymysql://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")
    # 使用 SQLAlchemy 执行 SQL 查询

    with db_engine.connect() as connection:
        # 使用 text() 包装 SQL 语句以支持原生 SQL
        sql1 = f"""
        select DISTINCT XM_NAME, XM_CODE from  {table}
        """
        middle = []
        result = connection.execute(text(sql1))
        for w in result.all():

            key = w[0]
            val = w[0].strip()
            if " " in val:
                val = val.replace(" ","")
            if val =="合计": # 总数 总额 合计 | 修改数据
                continue
            if "、" in val:
                middle.append((val.split("、")[1].strip(),key,table,name_dict[table],w[1]))
            elif "." in val:
                middle.append((val.split(".")[1].strip(), key, table, name_dict[table],w[1]))
            elif "#" in val:
                middle.append((val.strip("#"), key, table, name_dict[table],w[1]))
            elif '\n' in val:
                middle.append((val.replace("\n","").strip(), key, table, name_dict[table],w[1]))
            else:
                middle.append((val.strip(), key, table, name_dict[table],w[1]))
    return middle



def md5_word(word: str) -> str:
    return hashlib.md5(word.encode('utf-8')).hexdigest()


def id_search(client,point_id,words,payload,collection_name):

    # existing_points = client.retrieve(
    #     collection_name=collection_name,
    #     ids=[point_id],
    #     with_payload=True,
    #     with_vectors=False  # 不需要向量，节省带宽
    # )
    existing_points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="cate",
                    match=MatchValue(value=payload['cate'])
                ),
                FieldCondition(
                    key="words",
                    match=MatchValue(value=words)
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False
    )

    if not existing_points:
        point_to_upsert = PointStruct(
            id=point_id,
            vector= get_embedding1([words])['embeddings'][0],
            payload=payload
        )

    else:
        old_payload = existing_points[0].payload
        old_payload['table_select'].append(payload['table_select'][0])
        if 'select_code' in old_payload and old_payload['select_code']:
            old_payload['select_code'].append(payload['select_code'][0])
        old_payload['table'].append(payload['table'][0])

        old_payload['zh_table'].append(payload['zh_table'][0])
        point_to_upsert = PointStruct(
            id=existing_points[0].id,
            vector= get_embedding1([words])['embeddings'][0],
            payload=old_payload
        )

    client.upsert(
        collection_name=collection_name,
        points=[point_to_upsert]
    )

def search_next():
    """分页获取全部数据"""
    query_vector = get_embedding1(['2025年1月，河北省教育支出和其他支出是多少？'])['embeddings'][0]
    client = QdrantClient(host="192.168.100.160", port=6333)
    search_filter = Filter(
        must=[FieldCondition(
            key="cate",
            match=MatchValue(value="表描述"))])
    # results = client.search_matrix_pairs(
    results = client.query_points(
        collection_name="db_words_feng_260304",
        query=query_vector,
        query_filter=search_filter,
        limit=3,
        with_payload=True,
        with_vectors=False,
        score_threshold=None  # 如果需要最低相似度阈值，可在此设置 (如 0.7)
    )
    for x in results.points:
        print(x.payload)
    # print(results, "表描述相似度检索....")


    # next_offset = None
    # limit = 10
    # while True:
    #     # 第一次调用时 next_offset 为 None，之后使用上一次返回的 next_offset
    #     result, next_offset = client.scroll(
    #         collection_name="db_words_feng_260304",
    #         scroll_filter=Filter(
    #             must=[
    #                 FieldCondition(
    #                     key="cate",
    #                     match=MatchValue(value="科目/项目")
    #                 )
    #             ]
    #         ),
    #         limit=limit,  # 每页 100 条，可根据需要调整
    #         with_payload=True,
    #         with_vectors=False,
    #         offset=next_offset  # 传入上一次的 next_offset
    #     )
    #     print("当前页数据:", len(result))
    #     print("下一页偏移量:", next_offset)
    #     # 如果没有更多数据，退出循环
    #     if len(result) < limit:
    #         break

        # 收集当前页数据

def main():
    """往qdrant中插入数据"""
    tables = ["RDYS_LD_YSZX_YBGGYS_SBJZCWCQK",
              "RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC",
              "RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC",
              "RDYS_LD_YSZX_ZFXJJ_QSSRWCQK",
              ]
    client = QdrantClient(host="192.168.100.160", port=6333)
    record = []
    for x in tables:
        middle = sql_run(x)
        print(f"table:{x}=={len(middle)}")

        for y in middle:

            payload = {
                "words": y[0], # 唯一对应  城乡社区支出
                "table_select":[y[1]], # 表中的检索项 十、城乡社区支出
                "select_code":[y[4]], # 表中的检索项 十、城乡社区支出
                "table":[y[2]], # 所属表英文名
                "zh_table":[y[3]], # 所属表中文名
                "level":1,
                "cate":"科目/项目",
                "status":1
            }
            record.append(y[0])
            uid = uuid.uuid4().__str__().replace("-", "")
            id_search(client,uid,y[0],payload,"db_words_feng_260304")

    print(len(set(record)),"插入总长度....")


def run2():
    # 提取表中的筛选项将问题中的筛选项目

    # table_name = "RDYS_LD_YSZX_YBGGYS_SBJZCWCQK"
    with open('RDYS_PUBLIC_TBS.json', 'r',encoding='utf-8') as f:
        sql = json.loads(f.read())
        # middle = []
        tables = [
            "RDYS_LD_YSZX_YBGGYS_SBJZCWCQK",
                  "RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC",
                  "RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC",
                  "RDYS_LD_YSZX_ZFXJJ_QSSRWCQK",
                  ]
        char_list = ['数',"%","金","额","比"]
        count = 0
        client = QdrantClient(host="192.168.100.160", port=6333)
        for table_name in tables:
            for k,v in sql['tables'][table_name]['fields'].items():
                if "DECIMAL" in  v['type'] and  any(char in v['comment'] for char in char_list) :
                    payload = {
                        "words": v['comment'],  # 唯一对应
                        "table_select": [v['comment']],  # 表中的检索项
                        "select_code": [],  # 表中的检索项
                        "table": [table_name],  # 所属表英文名
                        "zh_table": [name_dict[table_name]],  # 所属表中文名
                        "level": 1,
                        "cate": "检索项",
                        "status": 1
                    }
                    count+=1
                    uid = uuid.uuid4().__str__().replace("-", "")
                    id_search(client,uid, v['comment'], payload, "db_words_feng_260304")


from qdrant_client.models import Filter, FieldCondition, MatchValue

def search_all():
    client = QdrantClient(host="192.168.100.160", port=6333)
    # 使用 scroll 方法遍历所有符合条件的记录，不返回向量
    result, next_offset = client.scroll(
        collection_name="db_words_feng_260304",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="cate",
                    match=MatchValue(value="科目/项目")
                )
            ]
        ),
        limit=1000,  # 根据需要调整数量，设为 None 可获取全部
        with_payload=True,  # 返回 payload
        with_vectors=False  # 不返回向量数据
    )

    # 打印结果
    end = []
    with open('project_words.txt','w',encoding='utf-8') as f:
        for point in result:
            f.write(point.payload['words']+'\n')

    # for point in result:
        # data = point.payload
        # if len(data['table_select'])!=len(data['select_code']):
        #     print(f"Payload: {point.payload}")
        # 判断一下三个等长列表长度是否不相等

        # end.append(point.payload['words'])
    # with open('table_xm1.json', 'w',encoding='utf-8') as f:
    #     f.write(json.dumps(end, ensure_ascii=False, indent=2))


def search_all2():
    client = QdrantClient(host="192.168.100.160", port=6333)
    # 使用 scroll 方法遍历所有符合条件的记录，不返回向量
    result, next_offset = client.scroll(
        collection_name="db_words_feng_260304",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="cate",
                    match=MatchValue(value="检索项")
                )
            ]
        ),
        limit=1000,  # 根据需要调整数量，设为 None 可获取全部
        with_payload=True,  # 返回 payload
        with_vectors=False  # 不返回向量数据
    )

    # 打印结果
    with open('col_words.txt','w',encoding='utf-8') as f:
        for point in result:
            f.write(point.payload['words']+'\n')
    # end = []
    # for point in result:
    #
    #     print(f"ID: {point.id}")
    #     print(f"Payload: {point.payload}")
    #     end.append(point.payload)
    # with open('table_col.json', 'w',encoding='utf-8') as f:
    #     f.write(json.dumps(end, ensure_ascii=False, indent=2))


def run3():
    tb1 = """
    表名：预算审查监督-预算执行表-全省一般公共预算收入完成情况表
    适用场景：查询全省各收入预算科目在指定年月的全省收入执行情况。
    支持科目：城镇土地使用税,国有资本经营收入,印花税,国内增值税,非税收入,行政事业性收费收入,土地增值税,罚没收入,个人所得税,企业所得税,其他收入,资源税,城市维护建设税,房产税,契税等其他税收,税收收入,国有资源（资产）有偿使用收入,专项收入,车船税,耕地占用税
    支持指标：
        1.当月维度：预算数、本月实际金额、上年同月数、同比增减额及百分比。
        2.累计维度（本年截至当月）：累计实际金额、上年同期累计数、累计同比增减额及百分比、累计预算完成率。
    **注意**：此表仅包含“全省一般公共预算收入”数据
    """
    tb2 = """
    表名：预算审查监督-预算执行表-全省一般公共预算支出完成情况表
    适用场景：查询全省各支出预算科目在指定年月的全省支出执行情况。
    支持科目：援助其他地区支出,科学技术支出,教育支出,其他支出,债务付息支出,自然资源海洋气象等支出,文化旅游体育与传媒支出,节能环保支出,国防支出,农林水支出,卫生健康支出,一般公共服务支出,债务发行费用支出,商业服务业等支出,住房保障支出,灾害防治及应急管理支出,社会保障和就业支出,粮油物资储备支出,资源勘探工业信息等支出,公共安全支出,城乡社区支出,交通运输支出,金融支出,资源勘探信息等支出
    支持指标：
        1.当月维度：预算数、本月实际金额、上年同月数、同比增减额及百分比。
        2.累计维度（本年截至当月）：累计实际金额、上年同期累计数、累计同比增减额及百分比、累计预算完成率。
     **注意**：此表仅包含“全省一般公共预算支出”数据
    """
    tb3 = """
    表名：预算执行-省本级一般公共预算支出完成情况表
    适用场景：查询省本级各预算科目在指定年月的省本级支出执行情况。
    支持科目：援助其他地区支出,科学技术支出,教育支出,其他支出,债务付息支出,自然资源海洋气象等支出,文化旅游体育与传媒支出,节能环保支出,国防支出,农林水支出,卫生健康支出,一般公共服务支出,债务发行费用支出,商业服务业等支出,住房保障支出,灾害防治及应急管理支出,社会保障和就业支出,粮油物资储备支出,资源勘探工业信息等支出,公共安全支出,城乡社区支出,交通运输支出,金融支出,资源勘探信息等支出
    支持指标：
        1.当月维度：预算数、本月实际金额、上年同月数、同比增减额及百分比。
        2.累计维度（本年截至当月）：累计实际金额、上年同期累计数、累计同比增减额及百分比、累计预算完成率。
     **注意**：此表仅包含“省本级一般公共预算支出”数据
    """
    tb4 = """
    表名：预算执行-全省政府性基金预算收入完成情况表
    适用场景：查询全省政府性基金各科目在指定年月的收入执行情况。
    支持科目：污水处理费,彩票公益金收入,农业土地开发资金收入,其他政府性基金收入,彩票发行机构和彩票销售机构的业务费用,国有土地使用权出让收入,车辆通行费收入,专项债务对应项目专项收入,国有土地收益基金收入,城市基础设施配套费收入,小型水库移民扶助基金收入,国家电影事业发展专项资金收入
    支持指标：
        1.当月维度：预算数、本月实际金额、上年同月数、同比增减额及百分比。
        2.累计维度（本年截至当月）：累计金额、上年同期累计数、累计同比增减额及百分比、累计预算完成率。
     **注意**：此表仅包含“全省政府性基金收入”数据
    """
    # 将表的描述内容也存入向量数据库
    client = QdrantClient(host="192.168.100.160", port=6333)
    data = [tb1, tb2, tb3, tb4]
    tables = ["RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC",
              "RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC",
              "RDYS_LD_YSZX_YBGGYS_SBJZCWCQK",
              "RDYS_LD_YSZX_ZFXJJ_QSSRWCQK",
              ]
    for n,x in enumerate(data):
        payload = {
            "cate": "表描述",
            "words": x,  # 唯一对应
            "table_select":[],
            "select_code":[],
            "table":[tables[n]],
            "zh_table":[name_dict[tables[n]]],
            "level": 1,
            "status": 1
        }

        point_id = uuid.uuid4().__str__().replace('-', '')
        point_to_upsert = PointStruct(
            id=point_id,
            vector=get_embedding1([x])['embeddings'][0],
            payload=payload
        )
        client.upsert(
            collection_name='db_words_feng_260304',
            points=[point_to_upsert]
        )






def create1(collection_name):
    # 创建一个名为 "articles" 的集合
    # vector_size=384 是所用嵌入模型的维度
    client = QdrantClient(host="192.168.100.160", port=6333)
    res = client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    print(res)
    client.close()


def del_many():
    client = QdrantClient(host="192.168.100.160", port=6333)
    # 定义过滤条件：删除所有 cate='news' 且 words='old_topic' 的数据
    delete_filter = Filter(
        must=[
            FieldCondition(
                key="cate",
                match=MatchValue(value="科目/项目")
            )

        ]
    )

    # 执行条件删除
    response = client.delete(
        collection_name="db_words_feng_260304",
        points_selector=delete_filter,  # 这里传入 Filter 对象而不是 ID 列表
        wait=True
    )

    print(f"删除状态: {response.status}")


def level2_insert(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # client = QdrantClient(host="192.168.100.160", port=6333)
        print(len(data),"总长度")
        # for x in data:
        #     print(x)
        #     point_id = uuid.uuid4().__str__().replace("-", "")
            # db_words_feng_260304
            # id_search(client, point_id, words, payload, collection_name)





# 预算执行-省本级一般公共预算支出完成情况表 RDYS_LD_YSZX_YBGGYS_SBJZCWCQK
# 情况记录：合   计 存在
# {'RDYS_LD_YSZX_YBGGYS_SBJZCWCQK': {'十三、资源勘探信息等支出': '资源勘探信息等支出',
# '十、城乡社区支出': '城乡社区支出', '十四、商业服务业等支出': '商业服务业等支出',
# '四、教育支出': '教育支出', '十七、自然资源海洋气象等支出': '自然资源海洋气象等支出',
# '八、卫生健康支出': '卫生健康支出', '二十三、其他支出': '其他支出',
# '十八、住房保障支出': '住房保障支出', '二十二、债务发行费用支出': '债务发行费用支出',
# '十六、援助其他地区支出': '援助其他地区支出', '二、国防支出': '国防支出',
# '十五、金融支出': '金融支出', '五、科学技术支出': '科学技术支出',
# '三、公共安全支出': '公共安全支出', '二十一、债务付息支出': '债务付息支出',
#  '十二、交通运输支出': '交通运输支出', '十一、农林水支出': '农林水支出',
#  '一、一般公共服务支出': '一般公共服务支出', '二十、灾害防治及应急管理支出': '灾害防治及应急管理支出',
#  '十九、粮油物资储备支出': '粮油物资储备支出', '七、社会保障和就业支出': '社会保障和就业支出',
#  '九、节能环保支出': '节能环保支出', '六、文化旅游体育与传媒支出': '文化旅游体育与传媒支出',
#  '十三、资源勘探工业信息等支出': '资源勘探工业信息等支出'}}
#----------------------------------------------------------------
# 预算审查监督-预算执行表-全省一般公共预算收入完成情况表 RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC
# '合    计',  '合    计' 存在是否删除？
# {'RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC': {'房产税 ': '房产税',
# '\u3000\u3000      其他收入': '其他收入',
# '\u3000\u3000      国有资源（资产）有偿使用收入': '国有资源（资产）有偿使用收入',
# '企业所得税': '企业所得税', '印花税': '印花税', '二、非税收入': '非税收入',
# '个人所得税': '个人所得税', '城镇土地使用税': '城镇土地使用税',
# '\u3000\u3000      行政事业性收费收入': '行政事业性收费收入',
# '罚没收入': '罚没收入', '土地增值税': '土地增值税',
# '耕地占用税': '耕地占用税', '城市维护建设税': '城市维护建设税',
# '一、税收收入': '税收收入', '契税等其他税收': '契税等其他税收',
# '资源税': '资源税', '国内增值税': '国内增值税',
# '\u3000      \u3000专项收入': '专项收入', '国有资本经营收入': '国有资本经营收入',
# '车船税': '车船税', '房产税': '房产税'}}
#----------------------------------------------------------------
# 预算审查监督-预算执行表-全省一般公共预算支出完成情况表 RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC
# 合   计 存在是否删除
# {'RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC': {'二十一、债务付息支出': '债务付息支出',
# '七、社会保障和就业支出': '社会保障和就业支出', '十、城乡社区支出': '城乡社区支出',
# '十八、住房保障支出': '住房保障支出', '二、国防支出': '国防支出', '二十二、债务发行费用支出': '债务发行费用支出',
# '五、科学技术支出': '科学技术支出', '十六、援助其他地区支出': '援助其他地区支出',
# '十九、粮油物资储备支出': '粮油物资储备支出', '合   计': '合   计',
# '二十、灾害防治及应急管理支出': '灾害防治及应急管理支出', '六、文化旅游体育与传媒支出': '文化旅游体育与传媒支出',
# '三、公共安全支出': '公共安全支出', '九、节能环保支出': '节能环保支出', '四、教育支出': '教育支出',
# '十七、自然资源海洋气象等支出': '自然资源海洋气象等支出', '十五、金融支出': '金融支出',
# '十四、商业服务业等支出': '商业服务业等支出', '十三、资源勘探信息等支出': '资源勘探信息等支出',
# '八、卫生健康支出': '卫生健康支出', '十一、农林水支出': '农林水支出', '二十三、其他支出': '其他支出',
# '一、一般公共服务支出': '一般公共服务支出',
# '十二、交通运输支出': '交通运输支出', '十三、资源勘探工业信息等支出': '资源勘探工业信息等支出'}}
# 预算执行-全省政府性基金预算收入完成情况表 RDYS_LD_YSZX_ZFXJJ_QSSRWCQK
# 合    计 存在
# {'RDYS_LD_YSZX_ZFXJJ_QSSRWCQK': {'6.城市基础设施配套费收入': '城市基础设施配套费收入',
# '5.彩票公益金收入': '彩票公益金收入', '11.其他政府性基金收入': '其他政府性基金收入',
# '9.污水处理费': '污水处理费', '12.专项债务对应项目专项收入': '专项债务对应项目专项收入',
# '10.彩票发行机构和彩票销售机构的业务费用': '彩票发行机构和彩票销售机构的业务费用',
# '3.农业土地开发资金收入': '农业土地开发资金收入',
# '1.国家电影事业发展专项资金收入': '国家电影事业发展专项资金收入',
# '7.小型水库移民扶助基金收入': '小型水库移民扶助基金收入', '8.车辆通行费收入': '车辆通行费收入',
# '4.国有土地使用权出让收入': '国有土地使用权出让收入', '2.国有土地收益基金收入': '国有土地收益基金收入',
# }}

if __name__ == '__main__':
    # search_next()
    # search_all2()
    # **run3()
    # schema_run('RDYS_LD_YSZX_ZFXJJ_QSSRWCQK')
    # level2_insert('final_table_xm.json')
    # del_many()
    # **run2()
    # create1('db_words_feng_260304')
    # **main()
    # search_all()
    # level2_insert('final_table_xm.json')
    table_info_create()
# ['RG_CODE', 'RG_NAME', 'XH', 'DATE_YEAR', 'DEPT_NAME', 'DATA_ID', 'DEPT_CODE'] 所有表都包含的字段