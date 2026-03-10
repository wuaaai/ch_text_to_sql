import time

import requests
from fastapi import FastAPI,Request
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import json
from starlette.middleware.cors import CORSMiddleware
from modelscope import AutoModelForCausalLM, AutoTokenizer
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import traceback
from sqlalchemy import create_engine
import ahocorasick
# pip install pyahocorasick

import collections
from typing import List, Dict, Optional, Tuple, Any

from starlette.responses import StreamingResponse, JSONResponse

api_config = {
    "api_key": "sk-d507bd835e174d99b57757f3010dfd02",
    "base_url": "https://api.deepseek.com",  #
    "model": "deepseek-chat",
    "max_tokens": 8192,
    "context_length": 130000,
    "plat": "deepseek"
}


class Text2SQLTableRanker:
    def __init__(self):
        # --- 配置区域：动态权重策略 ---
        self.weights_hit = {
            '项目匹配': 0.60,
            '表描述': 0.25,
            '列匹配': 0.15
        }
        self.weights_miss = {
            '项目匹配': 0.00,
            '表描述': 0.50,
            '列匹配': 0.50
        }
        # 列匹配增益：如果列匹配分极高，给予额外奖励
        self.column_boost_threshold = 0.8
        self.column_boost_factor = 0.1
        # 最低置信度阈值
        self.min_confidence = 0.36

    def parse_and_aggregate(self, raw_data: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        将扁平列表重组为：{ table_name: { source: max_score } }
        自动处理同一表同一来源的多条记录（取最大值）
        """
        # 初始化结构: { table: { source: [] } }
        temp_store = collections.defaultdict(lambda: collections.defaultdict(list))

        for item in raw_data:
            t_name = item['table']
            source = item['source']
            score = float(item['score'])
            temp_store[t_name][source].append(score)

        # 聚合: 取每个来源的最高分
        aggregated = {}
        for t_name, sources in temp_store.items():
            aggregated[t_name] = {}
            for source, scores in sources.items():
                # 策略：只要有一次高分匹配，就认为该维度匹配成功 (取 max)
                aggregated[t_name][source] = max(scores)

        return aggregated

    def calculate_score(self, scores: Dict[str, float]) -> Tuple[float, str]:
        """
        计算单张表的融合得分
        返回: (final_score, strategy_used)
        """
        s_kw = scores.get('项目匹配', 0.0)
        s_desc = scores.get('表描述', 0.0)
        s_col = scores.get('列匹配', 0.0)

        # 1. 决定使用哪套权重
        # 如果项目匹配分数 > 0 (或者你可以设一个阈值如 0.5)，视为命中
        if s_kw > 0.0:
            weights = self.weights_hit
            strategy = "KEYWORD_HIT"
        else:
            weights = self.weights_miss
            strategy = "SEMANTIC_ONLY"

        # 2. 线性加权
        base_score = (
                weights['项目匹配'] * s_kw +
                weights['表描述'] * s_desc +
                weights['列匹配'] * s_col
        )

        # 3. 列匹配增益 (Column Boost)
        # 如果列匹配度很高，说明这张表结构非常契合问题，额外加分
        boost = 0.0
        if s_col >= self.column_boost_threshold:
            boost = self.column_boost_factor * (s_col - self.column_boost_threshold)

        final_score = base_score + boost
        return final_score, strategy

    def rank(self, raw_data: List[Dict]) -> Optional[Dict]:
        """
        主入口：输入原始列表，输出最佳表信息
        """
        # 1. 数据重组
        candidates = self.parse_and_aggregate(raw_data)

        if not candidates:
            return None

        results = []

        # 2. 遍历计算每张表的得分
        for t_name, scores in candidates.items():
            # 确保缺失的源记为 0
            safe_scores = {
                '项目匹配': scores.get('项目匹配', 0.0),
                '表描述': scores.get('表描述', 0.0),
                '列匹配': scores.get('列匹配', 0.0)
            }

            final_score, strategy = self.calculate_score(safe_scores)

            results.append({
                'table': t_name,
                'final_score': final_score,
                'strategy': strategy,
                'details': safe_scores
            })

        # 3. 排序
        results.sort(key=lambda x: x['final_score'], reverse=True)

        best_match = results[0]

        # 4. 阈值过滤
        if best_match['final_score'] < self.min_confidence:
            return {
                'status': 'LOW_CONFIDENCE',
                'message': f"最高得分 {best_match['final_score']:.4f} 低于阈值 {self.min_confidence}",
                'top_table': best_match['table']
            }

        return {
            'status': 'SUCCESS',
            'selected_table': best_match['table'],
            'final_score': best_match['final_score'],
            'strategy': best_match['strategy'],
            'score_breakdown': best_match['details'],
            'all_candidates': results  # 可选：返回所有候选供调试
        }


class KeywordMatcher:
    def __init__(self, keywords):
        """
        初始化匹配器
        :param keywords: 词列表
        """
        self.A = ahocorasick.Automaton()
        for idx, word in enumerate(keywords):
            self.A.add_word(word, (idx, word))
        self.A.make_automaton()
        print(f"已加载 {len(keywords)} 个关键词，构建完成。")

    def find_any(self, text):
        """
        判断文本中是否包含任意一个关键词
        :return: (bool, found_word)
        """
        # end_index, (original_index, word) = item
        for item in self.A.iter(text):
            found_word = item[1][1]
            return True, found_word  # 找到一个就立即返回

        return False, None

    def find_all(self, text):
        """
        找出文本中所有的关键词
        :return: list of found words
        """
        results = []
        for item in self.A.iter(text):
            results.append(item[1][1])
        return results


app = FastAPI(debug=True,
              title="知识库+api",
              description="目前支持的表有：软件著作权表-software_copyright、农合机构核心经营指标表-nonghe_institution_indicators、企业财务数据表-bus_financial_data、企业主要人员表-key_people、工商信息+基础信息表-business_information、企业对外投资表-out_invest、作品著作权表-creation_copyright、企业变更记录表-change_log、专利信息表-patent_information、企业间接对外投资表-out_invest_indirect、股东信息表-shareholder_information、资质证书表-certification", )

# db_engine = create_engine(f"postgresql+psycopg2://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")

db_user_name = 'hbch'
db_pwd = 'hbch2711'
db_host = '192.168.100.160'
port = 3306
db_name = 'RDYS_PUBLIC_TBS'
db_engine = create_engine(f"mysql+pymysql://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")


@app.on_event("startup")
async def load_models():
    with open('RDYS_PUBLIC_TBS_new.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        table_dict2 = {}

        for x, y in data['tables'].items():
            table_dict2.update({y['comment']: {
                "db_id": data['db_id'],
                "schema": data['schema'],
                "tables": {x: data['tables'][x]}

            }})
    app.state.public_data = table_dict2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"加载 embedding 模型...{device}")
    app.state.embedding_model = SentenceTransformer(
        '/mnt/feng/models/bge-m3',
        device=device
    )
    app.state.embedding_model.eval()

    # 加载 rerank 模型
    app.state.rerank_model = CrossEncoder(
        '/mnt/feng/models/bge-rerank',
        device=device
    )
    local_path = '/mnt/feng/models/xiyan3b'
    app.state.sql_model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    app.state.chat = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"]
    )

    app.state.sql_tokenizer = AutoTokenizer.from_pretrained(local_path)
    app.state.use_table_info = {
        '预算执行-省本级一般公共预算支出完成情况表': {'key_words': ["BYS_JE"], 'project_key': ['科目编码'],
                                                      'is_sheng': 1, 'project_name': ['项目名称'], 'is_hj': 0, 'unit': {
                '万元': {'YSS': '预算数', 'BYS_JE': '本月数-金额', 'BYS_SNTYS': '本月数-上年同月数',
                         'BYS_TBE': '本月数-同比+、-额', 'BYLJS_JE': '至本月累计数-金额',
                         'BYLJS_SNTQS': '至本月累计数-上年同期数', 'BYLJS_TBE': '至本月累计数-同比+、-额',
                         'BNSY_LJS': '本年上月累计数', 'SNSY_LJS': '上年上月累计数'},
                '百分比': {'BYS_TBP': '本月数-同比+、-%', 'BYLJS_YSS': '至本月累计数-为预算数%',
                           'BYLJS_TBP': '至本月累计数-同比+、-%'}}, 'cate': '预算执行',
                                                      'table': 'RDYS_LD_YSZX_YBGGYS_SBJZCWCQK'},
        '预算审查监督-预算执行表-全省一般公共预算收入完成情况表': {'key_words': ["BYS_JE"], 'project_key': ['科目编码'],
                                                                   'is_sheng': 0, 'project_name': ['科目名称'],
                                                                   'is_hj': 0, 'unit': {
                '万元': {'YSS': '预算数', 'BYS_JE': '本月数—金额', 'BYS_SNTYS': '本月数—上年同月数',
                         'BYS_TBE': '本月数—同比+-额', 'ZBYLJS_JE': '至本月累计数-金额',
                         'ZBYLJS_SNTQS': '至本月累计数-上年同期数', 'ZBYLJS_TBE': '至本月累计数-同比+-额',
                         'BNSY_LJS': '本年上月累计数', 'SNSY_LJS': '上年上月累计数'},
                '百分比': {'BYS_TBB': '本月数—同比+-%', 'ZBYLJS_WYSS': '至本月累计数-为预算数%',
                           'ZBYLJS_TBB': '至本月累计数-同比+-%'}}, 'cate': '预算执行',
                                                                   'table': 'RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC'},
        '预算审查监督-预算执行表-全省一般公共预算支出完成情况表': {'key_words': ["BYS_JE"], 'project_key': ['科目编码'],
                                                                   'is_sheng': 0, 'project_name': ['科目名称'],
                                                                   'is_hj': 0, 'unit': {
                '万元': {'YSS': '预算数', 'BYS_JE': '本月数—金额', 'BYS_SNTYS': '本月数—上年同月数',
                         'BYS_TBE': '本月数—同比+-额', 'ZBYLJS_JE': '至本月累计数-金额',
                         'ZBYLJS_SNTQS': '至本月累计数-上年同期数', 'ZBYLJS_TBE': '至本月累计数-同比+-额',
                         'BNSY_LJS': '本年上月累计数', 'SNSY_LJS': '上年上月累计数'},
                '百分比': {'BYS_TBB': '本月数—同比+-%', 'ZBYLJS_WYSS': '至本月累计数-为预算数%',
                           'ZBYLJS_TBB': '至本月累计数-同比+-%'}}, 'cate': '预算执行',
                                                                   'table': 'RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC'},
        '预算执行-全省政府性基金预算收入完成情况表': {'key_words': ["BYS_JE"], 'project_key': ['科目代码'], 'is_sheng': 0,
                                                      'project_name': ['项目'], 'is_hj': 0, 'unit': {
                '万元': {'YSW': '预算数', 'BYS_JE': '本月数-金额', 'BYS_SNTYS': '本月数-上年同月数',
                         'BYS_TBS': '本月数-同比+、-额', 'BYLJS_JE': '至本月累计数-金额',
                         'BYLJS_SNTQS': '至本月累计数-上年同期数', 'BYLJS_TBE': '至本月累计数-同比+、-额',
                         'BNSY_LJS': '本年上月累计数', 'SNSY_LJS': '上年上月累计数'},
                '百分比': {'BYS_TBP': '本月数-同比+、-%', 'BYLJS_YSS': '至本月累计数-为预算数%',
                           'BYLJS_TBP': '至本月累计数-同比+、-%'}}, 'cate': '预算执行',
                                                      'table': 'RDYS_LD_YSZX_ZFXJJ_QSSRWCQK'}}

    with open('project_words.txt', 'r', encoding='utf-8') as f2:
        word_list = [x.strip() for x in f2.readlines()]
    with open('col_words.txt', 'r', encoding='utf-8') as f3:
        col_word_list = [x.strip() for x in f3.readlines()]
    app.state.collection_name = "db_words_feng_260304"
    app.state.words_match = KeywordMatcher(word_list)
    app.state.col_match = KeywordMatcher(col_word_list)
    app.state.client = QdrantClient(host="192.168.100.160", port=6333)
    app.state.nl2sqlite_template_cn = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
                            【用户问题】
                            {question}

                            【数据库schema】
                            {db_schema}

                            【参考信息】
                            {evidence}

                            【用户问题】
                            {question}

                            ```sql"""
    app.state.prompt1 = """# Role
                                       你是一名智能数据分析师。你的任务是将 SQL 查询结果转化为用户易懂的自然语言报告，并根据数据特征和用户需求，判断是否生成 Mermaid 图表代码（Dify 原生支持渲染）。

                                       # Input Data
                                       - **用户原始问题**: {question}
                                       - **改写后问题**: {new_question}
                                       - **表结构 Schema**: {demo}
                                       - **SQL 执行结果**: {rows}
                                       - **执行的 SQL 语句**: {sql_query}

                                       # Constraints & Rules

                                       ## 1. 自然语言描述 (必须)
                                       - **结论先行**: 第一句话直接回答核心问题。
                                       - **数据洞察**: 总结趋势、极值、占比。不要罗列所有数字，要提炼观点。
                                       - **空数据处理**: 如果结果为空，礼貌告知未找到数据。
                                       - **语气**: 专业、客观，禁止使用“查询结果显示”等术语。

                                       ## 2. 可视化决策 (Mermaid 优先)
                                       - **触发条件**: 用户明确请求画图，或者数据具有明显的对比/趋势/占比特征。
                                       - **图表选择逻辑**:
                                         - **饼图 (`pie`)**: 仅当数据表示“部分占整体”，且类别数量 <= 8 时。
                                         - **柱状图 (`xychart-beta`)**: 适用于分类对比（Dify 新版支持 xychart，若不支持则降级为文字描述）。*注意：Mermaid 对复杂柱状图支持有限，如果数据复杂，优先选饼图或折线。*
                                         - **折线图 (`xychart-beta`)**: 适用于时间序列趋势。
                                         - **不画图**: 如果数据只有 1 个点，或类别太多，或用户未要求，则不生成图表代码。
                                       - **输出格式**: 
                                         - 如果需要画图，**必须**输出标准的 Mermaid 代码块，语言标记为 `mermaid`。
                                         - 确保 Mermaid 语法正确，标签中不要包含可能导致解析错误的特殊字符（如冒号、引号需转义或简化）。

                                       ## 3. 输出结构
                                       请严格按照以下 Markdown 格式输出：

                                       ### 📊 数据分析
                                       [在此处撰写自然语言描述]

                                       ### 📈 可视化图表
                                       [如果需要画图，在此处输出 mermaid 代码块；如果不需要，输出“暂无适合的数据可视化图表。”]

                                       ```mermaid
                                       [在此处填入 mermaid 代码，例如：
                                       pie title 预算分布
                                           "教育支出" : 500
                                           "其他支出" : 300
                                       ]"""


@app.post("/embed", summary='embedding模型接口-bge-m3 1024维度')
async def embed(texts: list[str]):
    model = app.state.embedding_model
    with torch.no_grad():
        embeddings = model.encode(texts, convert_to_tensor=True)
    return {"embeddings": embeddings.cpu().numpy().tolist()}


@app.post("/rerank", summary='rerank模型接口')
async def rerank(query: str, documents: list[str]):
    model = app.state.rerank_model
    scores = model.predict([(query, doc) for doc in documents])
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_documents = [documents[i] for i in ranked_indices]
    ranked_scores = [float(scores[i]) for i in ranked_indices]
    return {
        "ranked_documents": ranked_documents,
        "scores": ranked_scores
    }


@app.post("/table_info", summary="人大表结构信息查询：不传table_name时默认返回所有表的粗略信息,传中文名称时返回具体信息")
async def table_info(table_name: str = None):
    if table_name not in app.state.public_data:
        res = []
        for x, y in app.state.public_data.items():
            tables = y['tables']
            t_name, t_info = "", {}
            for k, v in tables.items():
                t_name = k
                t_info = v
            res.append({
                "表名：": x,
                "英文表名：": t_name,
                "列数": len(t_info['fields']),
            })
        return res
    else:
        data = app.state.public_data[table_name]['tables']

        t_name, t_info = "", {}
        for k, v in data.items():
            t_name = k
            t_info = v
        res = {
            "表名": table_name,
            "英文表名：": t_name,
            "表信息：": t_info,
        }

        return res


@app.post("/sql2")
async def sql2(question: str, table_name: str = None):
    if table_name not in app.state.public_data:
        tables_list = [k for k in app.state.public_data.keys()]
        scores = app.state.rerank_model.predict([(question, doc) for doc in tables_list])
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_documents = [tables_list[i] for i in ranked_indices]
        ranked_scores = [float(scores[i]) for i in ranked_indices]
        desc_info = f"表名：{ranked_documents[0]}-相关度：{ranked_scores[0]}"
        print(desc_info)
        demo = app.state.public_data[ranked_documents[0]]

    else:
        desc_info = f"指定表名：{table_name}"
        demo = app.state.public_data[table_name]
    # demo只会选中一张表，因为后续要根据这张表去进行sql优化

    nl2sqlite_template_cn = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
        【用户问题】
        {question}

        【数据库schema】
        {db_schema}

        【参考信息】
        {evidence}

        【用户问题】
        {question}

        ```sql"""
    evidence = """
    question：2025年12月项目名称为个人所得税的预算数是多少？
    answer: SELECT `YSS`  FROM `RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC` WHERE `YEAR_MONTH` = '202512' AND `XM_NAME` = '个人所得税';
    """
    ## dialects -> ['SQLite', 'PostgreSQL', 'MySQL']
    prompt = nl2sqlite_template_cn.format(
        dialect="MySQL", db_schema=demo,
        question=question, evidence=evidence)
    message = [{'role': 'user', 'content': prompt}]

    sss = app.state.sql_tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = app.state.sql_tokenizer([sss], return_tensors="pt").to(app.state.sql_model.device)

    generated_ids = app.state.sql_model.generate(
        **model_inputs,
        pad_token_id=app.state.sql_tokenizer.pad_token_id,
        eos_token_id=app.state.sql_tokenizer.eos_token_id,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.8,
        do_sample=True,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    sql_query = app.state.sql_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 优化sql生成过程不带符号的问题
    try:
        table_data_col = list(demo['tables'].values())[0]['fields'].keys()
    except:
        table_data_col = None
    if table_data_col:
        print(table_data_col)
        print(sql_query)
        for u in table_data_col:
            if u + ' ' in sql_query and f"`{u}`" not in sql_query:
                sql_query = sql_query.replace(u + ' ', f"`{u}` ")

    try:
        # 使用 SQLAlchemy 执行 SQL 查询
        with db_engine.connect() as connection:
            # 使用 text() 包装 SQL 语句以支持原生 SQL
            result = connection.execute(text(sql_query))

            # 如果是查询语句（SELECT），获取结果
            if sql_query.strip().upper().startswith('SELECT'):
                # 获取列名
                columns = result.keys()

                # 获取所有行数据
                rows = []
                for row in result:
                    # 将 Row 对象转换为字典
                    row_dict = {}
                    for i, column in enumerate(columns):
                        row_dict[column] = row[i]
                    rows.append(row_dict)

                return {
                    "success": True,
                    "desc_info": desc_info,
                    "data": rows,
                    "columns": list(columns),
                    "row_count": len(rows),
                    "sql_query": sql_query,
                    "message": "SQL 查询执行成功"
                }


    except SQLAlchemyError as e:
        # 捕获 SQLAlchemy 相关错误
        error_msg = str(e.__cause__) if e.__cause__ else str(e)
        return {
            "success": False,
            "desc_info": desc_info,
            "error_type": "SQLAlchemyError",
            "error_message": error_msg,
            "sql_query": sql_query
        }
    except Exception as e:
        # 捕获其他异常
        return {
            "success": False,
            "desc_info": desc_info,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "sql_query": sql_query,
            "traceback": traceback.format_exc()
        }


def wait_info(search_point, check_sheng, use_table_info, source):
    wait_tables = []
    for x in search_point.points:
        for n, y in enumerate(x.payload['zh_table']):
            if check_sheng:
                if use_table_info[y]['is_sheng']:
                    wait_tables.append({"table":y, "score": x.score, "source": source})
            else:
                if not use_table_info[y]['is_sheng']:
                    wait_tables.append({"table": y, "score": x.score, "source": source})
    return wait_tables


@app.post("/model_chat_test")
def model_chat_test(question: str, name: str, data1: dict, data2: dict):
    prompt = f"""
        # Role
    你是一名专业的 Text-to-SQL 语义解析助手。你的任务是根据用户原始自然语言问题，结合提供的数据库 schema 映射信息，将问题重写为逻辑清晰、术语精确的“优化后问题”。

    # Input Data
    1. **用户原始问题**: 2025年1月河北省教育和其他支出的预算是多少？
    2.**使用名称**:项目名称
    3. **名称映射表** (用户词汇 -> 数据库实际值):
    {{"教育": "教育支出", "其他支出": "其他支出"}}
    4. **列名映射表** (用户词汇 -> 数据库实际列名):
    {{"预算": "预算数"}}

    # Constraints & Rules
    1. **术语替换**: 必须严格使用【使用名称】、【名称映射表】和【列名映射表】中的“数据库实际值/列名”替换用户问题中的对应口语化词汇。
    2. **逻辑保留**: 保持原问题的时间、地点、筛选条件（如“和”、“或”）及查询意图不变。
    3. **句式规范**: 优化后的问题应是一个完整的陈述句或疑问句，结构通常为：“[时间][地点]项目名称为[具体项目值]的[具体列名]是多少？”
    4. **无多余输出**: 最终输出**仅包含**优化后的问题文本，不要包含任何解释、前缀（如“优化结果：”）或标点符号以外的字符。

    # Few-Shot Example
    **输入**:
    - 用户原始问题: 2025年1月河北省教育和其他支出的预算是多少？

    - 项目名称映射表: {{"教育": "教育支出", "其他支出": "其他支出"}}
    - 列名映射表: {{"预算": "预算数"}}

    **输出**:
    2025年1月河北省项目名称为教育支出和其他支出的预算数是多少？

    # Execution
    请根据上述规则处理以下输入：

    **输入**:
    - 用户原始问题: {question}
    - 使用名称：{name}
    - 名称映射表: {data1}
    - 列名映射表: {data2}
    **输出**:
        """
    new_message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    create_params = {
        "model": "deepseek-chat",
        "messages": new_message,
        "temperature": 0.7,
        "max_tokens": 8192,
        "stream": False,
    }
    # params_copy["extra_body"] = {"thinking": {"type": "disabled"}}
    response = app.state.chat.chat.completions.create(**create_params)
    ret = response.choices[0].message.content
    return ret


def model_chat(chat, question, name, data1, data2):
    prompt = f"""
            # Role
        你是一名专业的 Text-to-SQL 语义解析助手。你的任务是根据用户原始自然语言问题，结合提供的数据库 schema 映射信息，将问题重写为逻辑清晰、术语精确的“优化后问题”。

        # Input Data
        1. **用户原始问题**: 2025年1月河北省教育和其他支出的预算是多少？
        2.**使用名称**:项目名称
        3. **名称映射表** (用户词汇 -> 数据库实际值):
        {{"教育": "教育支出", "其他支出": "其他支出"}}
        4. **列名映射表** (用户词汇 -> 数据库实际列名):
        {{"预算": "预算数"}}

        # Constraints & Rules
        1. **术语替换**: 必须严格使用【使用名称】、【名称映射表】和【列名映射表】中的“数据库实际值/列名”替换用户问题中的对应口语化词汇。
        2. **逻辑保留**: 保持原问题的时间、地点、筛选条件（如“和”、“或”）及查询意图不变。
        3. **句式规范**: 优化后的问题应是一个完整的陈述句或疑问句，结构通常为：“[时间][地点]项目名称为[具体项目值]的[具体列名]是多少？”
        4. **无多余输出**: 最终输出**仅包含**优化后的问题文本，不要包含任何解释、前缀（如“优化结果：”）或标点符号以外的字符。

        # Few-Shot Example
        **输入**:
        - 用户原始问题: 2025年1月河北省教育和其他支出的预算是多少？

        - 项目名称映射表: {{"教育": "教育支出", "其他支出": "其他支出"}}
        - 列名映射表: {{"预算": "预算数"}}

        **输出**:
        2025年1月河北省项目名称为教育支出和其他支出的预算数是多少？

        # Execution
        请根据上述规则处理以下输入：

        **输入**:
        - 用户原始问题: {question}
        - 使用名称：{name}
        - 名称映射表: {data1}
        - 列名映射表: {data2}
        **输出**:
            """
    new_message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    create_params = {
        "model": "deepseek-chat",
        "messages": new_message,
        "temperature": 0.7,
        "max_tokens": 8192,
        "stream": False,
    }
    # params_copy["extra_body"] = {"thinking": {"type": "disabled"}}
    response = chat.chat.completions.create(**create_params)
    ret = response.choices[0].message.content
    return ret


def ret_format(msg):
    return json.dumps({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": msg
                    },
                    "finish_reason": None
                }]
            }) + "\n\n"
@app.post("/v1/chat/completions")
async def sql3(request: Request):
    # 此接口不用指定表名-会在目前的四张表中自己根据问题选一张
    # demo只会选中一张表，因为后续要根据这张表去进行sql优化
    # 多个词命中情况--多表 预算数和一般公共预算收入
    # 多个词命中情况--一表多指标

    # 1. 解析原始请求体 (保持原样，兼容字节流等格式)
    body = await request.body()
    body_str = body.decode('utf-8')

    try:
        request_json: Dict[str, Any] = json.loads(body_str)

    except json.JSONDecodeError:

        if body_str.startswith('b\'') and body_str.endswith('\''):
            body_str = body_str[2:-1]
            request_json = json.loads(body_str)
        else:
            # 返回错误响应而不是抛出异常
            return {
                "error": {
                    "type": "invalid_request_error",
                    "message": "请求异常，收到的参数不是json"
                }
            }
    print("request_json:",request_json)
    # 2. 提取参数 (Dify 会自动带上 tools 参数如果配置了工具)
    messages = request_json.get("messages", [])
    question = messages[-1].get("content", "")
    stream = request_json.get("stream", False)
    print(question,"question")
    # TODO:问题可能会被拆分成两个表的查询语句，目前先不考虑连表和多表查询情况
    question_words = app.state.words_match.find_all(question)

    # all_info = {}
    # 如果问题中没有出现明确的省本级意图，(省级、本级)
    # 清理结果中的省本级的表。如果出现省本级字眼，其他跟省本级无关的表剔除
    # 1.先找到所有科目相关的关键词对问题进行分词，判断关键词是否出现，此动作可以直接定位几张表
    # 2.对问题直接进行表描述级的向量检索，此动作也可以找到相关的top3表
    # 3.如果关键词没找到，直接根据表检索结果确定
    # 4.对于4本账的情况分清楚省本级，省级的表定位
    async def generate_progress():
        name = ""
        wait_tables = []
        data1 = {}
        data2 = {}
        check_sheng = 1 if any(char in question for char in ['省级', "本级"]) else 0
        yield ret_format(f"解析到问题可能和{'省本级数据' if check_sheng else '全省数据'}相关\n\n")
        project_words = []
        if question_words:
            ques_words = list(set(question_words))
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="cate",
                        match=MatchValue(value="科目/项目")
                    ),
                    FieldCondition(
                        key="words",
                        match=MatchAny(any=ques_words)
                    )
                ]
            )
            existing_points = app.state.client.query_points(
                collection_name=app.state.collection_name,
                # query=dummy_vector,
                query_filter=query_filter,
                limit=len(ques_words),
                with_payload=True,
                with_vectors=False,
                score_threshold=None  # 不过滤分数
            )
            # 构建data1映射字典
            for ww in existing_points.points:
                for n, u in enumerate(ww.payload["zh_table"]):
                    project_words.append(ww.payload["words"])
                    if u not in data1:
                        data1[u] = [{ww.payload["words"]: ww.payload["select_code"][n]}]
                    else:
                        data1[u].append({ww.payload["words"]: ww.payload["select_code"][n]})

            wait_tables.extend(wait_info(existing_points, check_sheng, app.state.use_table_info, source="项目匹配"))
        yield ret_format(f"已找到{len(project_words)}个关键词，分别为{','.join(project_words)}\n\n")

        with torch.no_grad():
            embedding = app.state.embedding_model.encode([question], convert_to_tensor=True)
            query_vector = embedding.cpu().numpy().tolist()[0]
        search_filter = Filter(
            must=[FieldCondition(
                key="cate",
                match=MatchValue(value="表描述"))])
        results = app.state.client.query_points(
            collection_name=app.state.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=3,
            with_payload=True,
            with_vectors=False,
            score_threshold=None  # 如果需要最低相似度阈值，可在此设置 (如 0.7)
        )
        wait_tables.extend(wait_info(results, check_sheng, app.state.use_table_info, source="表描述"))
        yield ret_format(f"表相似度检索完毕。\n\n")
        # 相关列匹配,现在用的强匹配
        col_words = app.state.col_match.find_all(question)
        col_words2 = []
        if col_words:
            col_words = list(set(col_words))
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="cate",
                        match=MatchValue(value="检索项")
                    ),
                    FieldCondition(
                        key="words",
                        match=MatchAny(any=col_words)
                    )
                ]
            )
            col_points = app.state.client.query_points(
                collection_name=app.state.collection_name,
                query_filter=query_filter,
                limit=len(col_words),
                with_payload=True,
                with_vectors=False,
                score_threshold=None  # 不过滤分数
            )
            for yy in col_points.points:
                for n, u in enumerate(yy.payload["zh_table"]):
                    col_words2.append(yy.payload["words"])
                    if u not in data2:
                        data2[u] = [{yy.payload["words"]: yy.payload["table_select"][n]}]
                    else:
                        data2[u].append({yy.payload["words"]: yy.payload["table_select"][n]})
            wait_tables.extend(wait_info(col_points, check_sheng, app.state.use_table_info, source="列匹配"))
        yield ret_format(f"已找到{len(col_words2)}个列关键词，分别为{','.join(col_words2)}\n\n")
        ranker = Text2SQLTableRanker()
        print("wait_tables:", wait_tables)
        table_name = ""
        result = ranker.rank(wait_tables)
        if result['status'] == "SUCCESS":
            yield ret_format(f"经过权重算法分析选中最相关表为：{result['selected_table']}。\n\n")
            table_name = result['selected_table']
            name = app.state.use_table_info[table_name]['project_key'][0]
            # TODO:基于大模型回复不稳定的问题提出优化方案
            new_question = model_chat(app.state.chat, question, name, data1.get(table_name, {}),
                                      data2.get(table_name, {}))
            print(new_question, "--改良后的问题")
            yield ret_format(f"问题已优化为：{new_question}\n\n")
            # 如果选中了表，那么根据选择过程中的数据去优化问题
            # 这里要使用一次模型
            # 5.根据选中的表进行sql生成,要求每个步骤把定位的内容实时输出到控制台

            demo = app.state.public_data[table_name]


            evidence = """
                    question：2025年12月项目名称为个人所得税的预算数是多少？
                    answer: SELECT `YSS`  FROM `RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC` WHERE `YEAR_MONTH` = '202512' AND `XM_NAME` = '个人所得税';
                    """
            ## dialects -> ['SQLite', 'PostgreSQL', 'MySQL']
            prompt = app.state.nl2sqlite_template_cn.format(
                dialect="MySQL", db_schema=demo,
                question=new_question, evidence=evidence)
            message = [{'role': 'user', 'content': prompt}]

            sss = app.state.sql_tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = app.state.sql_tokenizer([sss], return_tensors="pt").to(app.state.sql_model.device)

            generated_ids = app.state.sql_model.generate(
                **model_inputs,
                pad_token_id=app.state.sql_tokenizer.pad_token_id,
                eos_token_id=app.state.sql_tokenizer.eos_token_id,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.8,
                do_sample=True,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            sql_query = app.state.sql_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # 优化sql生成过程不带符号的问题
            # 默认使用序号排序
            try:
                table_data_col = list(demo['tables'].values())[0]['fields'].keys()
            except:
                table_data_col = None
            if table_data_col:
                for u in table_data_col:
                    if u + ' ' in sql_query and f"`{u}`" not in sql_query:
                        sql_query = sql_query.replace(u + ' ', f"`{u}` ")
            if "ORDER BY" not in sql_query:
                sql_query = sql_query.replace(';', " ") + " ORDER BY XH;"
            yield ret_format(f"已生成 SQL 语句为：{sql_query}\n\n")
            try:
                # 使用 SQLAlchemy 执行 SQL 查询
                with db_engine.connect() as connection:
                    # 使用 text() 包装 SQL 语句以支持原生 SQL
                    result = connection.execute(text(sql_query))

                    # 如果是查询语句（SELECT），获取结果
                    if sql_query.strip().upper().startswith('SELECT'):
                        # 获取列名
                        columns = result.keys()
                        # 获取所有行数据
                        rows = []
                        for row in result:
                            # 将 Row 对象转换为字典
                            row_dict = {}
                            for i, column in enumerate(columns):
                                row_dict[column] = row[i]
                            rows.append(row_dict)
                        yield ret_format(f"执行 SQL 查询成功\n\n")
                        try:
                            new_prompt = app.state.prompt1.format(
                                question=question, new_question=new_question, demo=demo,
                            rows=rows,sql_query=sql_query)
                            new_message = [
                                {"role": "system", "content": new_prompt},
                                {"role": "user", "content": question},
                            ]
                            create_params = {
                                "model": "deepseek-chat",
                                "messages": new_message,
                                "temperature": 0.7,
                                "max_tokens": 8192,
                                "stream": True,
                            }
                            # params_copy["extra_body"] = {"thinking": {"type": "disabled"}}
                            e1 = app.state.chat.chat.completions.create(**create_params)
                            for chunk in e1:
                                # 直接将 API 的 chunk 转发给客户端
                                yield json.dumps({
                                    "id": f"chatcmpl-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": "deepseek-chat",
                                    "choices": chunk.model_dump()["choices"]
                                }) + "\n\n"
                            yield json.dumps({
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "deepseek-chat",
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }) + "\n\n"

                        except SQLAlchemyError as e:
                            # 捕获 SQLAlchemy 相关错误
                            error_msg = str(e.__cause__) if e.__cause__ else str(e)
                            yield ret_format(f"执行SQL查询失败，错误信息为：{error_msg}\n\n")
                            yield json.dumps({
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "deepseek-chat",
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }) + "\n\n"

            except Exception as e:
                # 捕获其他异常
                yield ret_format(f"执行SQL捕获其他异常：{str(e)}\n\n")
                yield json.dumps({
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "deepseek-chat",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }) + "\n\n"


        else:
            yield ret_format(f"选表失败，启动兜底策略\n\n")
            yield json.dumps({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }) + "\n\n"
            # TODO:没有选中表时设计兜底策略
            # return {"status": "FAIL", "message": "没有找到相关表"}

    if stream:

        return StreamingResponse(
            generate_progress(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Nginx 禁用缓冲
            }
        )
    else:
        return JSONResponse(
            status_code=200,
            content={
                "error": {
                    "message": "响应成功！但是本模型只支持流式返回，请使用 stream=True 参数",
                    "type": "system_error"
                }
            }
        )

app.add_middleware(  # 解决跨域问题
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('bge_main:app', host=f'192.168.100.160', port=8787, workers=1)
