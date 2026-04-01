import datetime
import random
import time
import re

import requests
# from echarts import Echart, Legend, Bar, Line, Axis, Tooltip, Pie
from pyecharts.charts import Bar, Line, Pie
from pyecharts import options as opts
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.params import Body, Form
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
import json
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
import ahocorasick
import redis
from collections import defaultdict

import collections
from typing import List, Dict, Optional, Tuple, Any

from starlette.responses import StreamingResponse, JSONResponse

BASE_URL = "http://192.168.100.160:8991"
api_config = {
    "api_key": "sk-d507bd835e174d99b57757f3010dfd02",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "max_tokens": 8192,
    "context_length": 130000,
    "plat": "deepseek",
    "timeout": 60,

}
def get_embedding1(text_list):
    url = f"{BASE_URL}/embed"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=text_list)
    return response.json()


def get_text2sql(question,demo,evidence):
    try:
        data = requests.post(url=f"{BASE_URL}/text2sql",
                            json={"question": question, "demo": json.dumps(demo), "evidence": evidence},
                            timeout=30
                            )
        ret = data.json()
        print("text2sql->",ret)
    except BaseException as e:
        ret = {"status":"error", "message": str(e)}

    return ret


class Text2SQLTableRanker:
    def __init__(self):
        # 表描述最重要
        self.weight_desc = 0.6
        self.weight_project = 0.27
        self.weight_column = 0.13

        # 总分阈值
        self.min_confidence = 0.38

        # 表描述相关阈值
        self.min_desc_score = 0.45
        self.low_desc_score = 0.30

        # 表描述高分时的协同奖励
        self.desc_high_threshold = 0.75
        self.project_synergy_bonus = 0.04
        self.column_synergy_bonus = 0.02

        # 候选差距阈值
        self.min_score_gap = 0.03
        self.min_desc_gap = 0.05

    def parse_and_aggregate(self, raw_data: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
        temp_store = collections.defaultdict(lambda: collections.defaultdict(list))

        for item in raw_data:
            t_name = item['table']
            source = item['source']
            score = float(item['score'])
            temp_store[t_name][source].append(score)    #temp_store[表名][来源] = [分数1, 分数2, 分数3...]

        aggregated = {}
        for t_name, sources in temp_store.items():
            aggregated[t_name] = {}
            for source, scores in sources.items():
                aggregated[t_name][source] = {
                    'max_score': max(scores),
                    'count': len(scores)
                }

        return aggregated

    def count_to_strength(self, count: int) -> float:
        """
        更平缓的次数增益，避免项目匹配2次就轻易压过表描述
        """
        if count <= 0:
            return 0.0
        elif count == 1:
            return 0.65
        elif count == 2:
            return 0.78
        elif count == 3:
            return 0.88
        else:
            return 0.95

    def calculate_score(self, stats: Dict[str, Dict[str, float]]) -> float:
        desc_score = stats.get('表描述', {}).get('max_score', 0.0)

        project_count = stats.get('项目匹配', {}).get('count', 0)
        column_count = stats.get('列匹配', {}).get('count', 0)

        project_strength = self.count_to_strength(project_count)
        column_strength = self.count_to_strength(column_count)

        final_score = (
                self.weight_desc * desc_score +
                self.weight_project * project_strength +
                self.weight_column * column_strength
        )

        if desc_score >= self.desc_high_threshold and project_count > 0:
            final_score += self.project_synergy_bonus

        if desc_score >= self.desc_high_threshold and column_count > 0:
            final_score += self.column_synergy_bonus

        return final_score

    def rank(self, raw_data: List[Dict]) -> Optional[Dict]:
        candidates = self.parse_and_aggregate(raw_data)
        if not candidates:
            return None

        results = []

        for t_name, stats in candidates.items():
            final_score = self.calculate_score(stats)

            desc_score = stats.get('表描述', {}).get('max_score', 0.0)
            project_count = stats.get('项目匹配', {}).get('count', 0)
            column_count = stats.get('列匹配', {}).get('count', 0)

            score_breakdown = {
                '表描述分': desc_score,
                '项目匹配次数': project_count,
                '列匹配次数': column_count,
                '项目匹配强度': self.count_to_strength(project_count),
                '列匹配强度': self.count_to_strength(column_count),
            }

            results.append({
                'table': t_name,
                'final_score': round(final_score, 6),
                'details': score_breakdown
            })

        results.sort(key=lambda x: x['final_score'], reverse=True)
        best_match = results[0]

        # 总分阈值
        if best_match['final_score'] < self.min_confidence:
            return {
                'status': 'LOW_CONFIDENCE',
                'message': f"最高得分 {best_match['final_score']:.4f} 低于总分阈值 {self.min_confidence}",
                'top_table': best_match['table'],
                'all_candidates': results
            }
        # 规则3：top1和top2过近，说明区分度不足
        # TODO：兜底策略，未实现，当前几个得分最高的表得分相近时的情况，后续要用到（候选差距阈值self.min_score_gap = 0.03和self.min_desc_gap = 0.05）
        if len(results) > 1:
            if best_match['final_score'] < self.min_confidence:
                return {
                    'status': 'LOW_CONFIDENCE',
                    'message': f"最高得分 {best_match['final_score']:.4f} 低于阈值 {self.min_confidence}",
                    'top_table': best_match['table'],
                    'all_candidates': results
                }

        return {
            'status': 'SUCCESS',
            'selected_table': best_match['table'],
            'final_score': best_match['final_score'],
            'score_breakdown': best_match['details'],
            'all_candidates': results
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
db_name = 'RDYS_PUBLIC_TBS_WU'
db_engine = create_engine(f"mysql+pymysql://{db_user_name}:{db_pwd}@{db_host}:{port}/{db_name}")


@app.on_event("startup")
async def load_models():
    app.state.redis = redis.Redis(
        host="127.0.0.1",
        port=6379,
        db=11,
        encoding="utf-8",
        decode_responses=True

    )  # password=REDIS_PASS

    with open('text2sql_project/RDYS_PUBLIC_TBS_WU.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        table_dict2 = {}

        for x, y in data['tables'].items():
            table_dict2.update({y['comment']: {
                "db_id": data['db_id'],
                "schema": data['schema'],
                "tables": {x: data['tables'][x]}

            }})
    app.state.public_data = table_dict2
    app.state.chat = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"],
        max_retries=2,
        timeout=60
    )

    # TODO: 表描述 table_info.json未出现此文件
    with open('text2sql_project/table_info.json', 'r', encoding='utf-8') as f4:
        app.state.use_table_info = json.load(f4)

    with open('text2sql_project/project_words.txt', 'r', encoding='utf-8') as f2:
        word_list = [x.strip() for x in f2.readlines()]
    with open('text2sql_project/col_words.txt', 'r', encoding='utf-8') as f3:
        col_word_list = [x.strip() for x in f3.readlines()]
    app.state.collection_name = "db_words_wu"  # all_data_test_0317 db_words_feng_260304
    app.state.words_match = KeywordMatcher(word_list)
    app.state.col_match = KeywordMatcher(col_word_list)
    app.state.zone_dict = {'河北省': '130000000', '河北省本级': '130000000', '邢台市': '130500000',
                           '唐山市': '130200000', '邢台': '130500000', '南和区': '130506000', '曹妃甸区': '130209000',
                           '滦州市': '130284000', '平乡县': '130532000', '柏乡县': '130524000', '迁西县': '130227000',
                           '广宗县': '130531000', '巨鹿县': '130529000', '临西县': '130535000', '沙河市': '130582000',
                           '遵化市': '130281000', '清河县': '130534000', '路北区': '130203000', '丰南区': '130207000',
                           '路南区': '130202000', '丰润区': '130208000', '玉田县': '130229000', '南宫市': '130581000',
                           '滦南县': '130224000', '临城县': '130522000'}
    app.state.zone_match = KeywordMatcher(list(app.state.zone_dict.keys()))

    app.state.client = QdrantClient(host="192.168.100.160", port=6333,timeout = 60)
    app.state.nl2sqlite_template_cn = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
                            【约束规则】
                             1.查询结果中筛选列不可缺失。即生成的sql的检索结果必须带上检索的日期维度或者地区维度。
                             2.生成的查询语句中不允许出现别名情况。例如：SELECT `DEPT_NAME` AS 部门名称。严禁出现别名定义
                             3.相同的时间字段在SELECT时不可出现两次。比如：YEAR_MONTH和DATE_YEAR不要同时存在
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
你是一名资深智能数据分析师。你的核心任务是将 SQL 查询结果转化为专业、易懂的自然语言分析报告。你需要具备极强的数据敏感度，能够识别单位陷阱、清洗脏数据，并根据数据特征智能选择呈现方式。

# Input Data
- **用户原始问题**: {question}
- **改写后问题**: {new_question}
- **表结构 Schema**: {demo}
- **SQL 执行结果**: {rows}
- **执行的 SQL 语句**: {sql_query}

# Critical Pre-processing Rules (数据预处理与校准 - 最高优先级)
在生成任何分析之前，必须在内心执行以下数据清洗步骤：
1. **单位统一与换算**:
   - 检查数值字段是否包含中文单位（如 "万元", "亿元", "千人" 等）。
   - **严禁**直接保留带单位的字符串进行数学比较。必须将其转换为纯数字（建议统一转换为标准单位，如“元”或保持“万元”但数值正确）。
   - *错误示例*: 输入 "1000万元"，分析时写成 "1000元" 或 "0.1万元"。
   - *正确操作*: 输入 "1161668.000000万元" -> 理解为 116.17 亿元 (若需展示为万元，则为 1,161,668.00 万元)。**绝对禁止**将百万级的数值误读为百级。
2. **维度真实性校验**:
   - **严禁捏造数据中不存在的维度**（如月份、地区、类别）。
   - 如果 `{rows}` 中仅包含数值列表而无时间/分类字段，**不得**强行假设其为“1月、2月...”或“A类、B类...”。
   - 若无明确时间字段，仅描述“第1条记录”至“第N条记录”的趋势，或指出“数据缺乏时间维度，无法判断具体月份趋势”。
3. **异常值识别**:
   - 若发现数值量级差异巨大（如相差1000倍以上），需在分析中标记为潜在异常或确认单位是否一致。

# Constraints & Rules

## 1. 自然语言深度分析 (必须)
- **结论先行**: 第一段必须用一句话直接回答用户的核心问题，给出明确的结论（包含正确的量级和单位）。
- **数据洞察**: 
  - **趋势分析**: 仅在数据包含明确时间维度时分析趋势。若无时间维度，分析数值分布（如：大部分数据集中在哪个区间）。
  - **极值与异常**: 准确指出最大值、最小值。**注意单位换算后的真实大小**。
  - **拒绝罗列**: 严禁简单重复 SQL 结果。必须提炼观点。
- **空数据处理**: 如果 `{rows}` 为空，礼貌且专业地告知未找到相关数据。
- **语气风格**: 专业、客观、简洁。禁止使用“根据查询结果”、“数据显示”等机械性套话。

## 2. 智能表格生成策略
根据数据特征，自主判断是否生成 Markdown 表格：

- **触发条件**:
  1. **数据量适中**: 行数在 2 行到 15 行之间。
  2. **维度清晰**: 数据中包含明确的分类或时间标签（若原始数据无标签，不要强行添加“月份”列，可用“序号”或“记录ID”代替，或在表头注明“未知时间维度”）。
  3. **关键指标对比**: 用户需要对比具体数值。

- **不生成表格的情况**:
  1. **单行/单值**: 直接用文字描述。
  2. **数据量过大**: 超过 15 行，仅展示 Top 3/Bottom 3 或统计摘要。
  3. **维度缺失导致误导**: 如果强行加月份会误导用户，则不生成带月份的表格，改为纯文本分析或仅列出数值。

- **表格格式要求**:
  - 表头名称应简洁明了，**必须包含正确的单位**（如 `金额 (万元)`）。
  - 数值列格式化：去除多余的零，保留2位小数，增加千分位分隔符以提高可读性。

## 3. 输出结构
请严格按照以下 Markdown 格式输出：

### 📊 核心结论与分析
[在此处撰写自然语言深度分析报告。确保数值量级准确，单位换算正确。若无时间维度，不要编造月份趋势。]

### 📋 数据明细
[如果需要展示表格，在此处输出 Markdown 表格。
 **重要**: 若原始数据无时间字段，表格第一列请使用“序号”或原始数据中的唯一标识，**严禁**虚构“2025年XX月”。
 若不需要表格，输出提示语。]

[在此处填入 Markdown 表格代码]

---

# Few-Shot Examples (参考范例 - 修正版)

## 范例 1: 单位陷阱处理 (关键)
**输入数据**: `[{{'val': '1500000.00万元'}}]` (意为150万亿元，或数据源本身有误，需按字面理解量级)
**错误思维**: 直接当成150万元。
**正确输出**:
### 📊 核心结论与分析
当前记录显示的数值高达 **150万亿** (1,500,000.00 万元)，该量级极其巨大，可能代表累计总额或存在单位录入异常，需进一步核实数据源准确性。

### 📋 数据明细
| 序号 | 金额 (万元) | 备注 |
| :--- | :--- | :--- |
| 1 | 1,500,000.00 | 量级异常巨大 |

## 范例 2: 无时间维度的列表数据
**输入数据**: `[{{'amt': 100}}, {{'amt': 200}}, {{'amt': 150}}]` (无日期字段)
**错误思维**: 强行说是1月、2月、3月。
**正确输出**:
### 📊 核心结论与分析
数据共包含3条记录，数值在100至200之间波动。其中第2条记录达到峰值200，较最低值（第1条记录，100）增长了100%。由于数据未包含具体时间戳，无法判断月度趋势。

### 📋 数据明细
| 序号 | 金额 (元) | 相对变化 |
| :--- | :--- | :--- |
| 1 | 100.00 | - |
| 2 | 200.00 | 🔺 +100% |
| 3 | 150.00 | 🔻 -25% |
"""
    app.state.prompt2 = """你是一个专业的数据可视化助手。请根据用户提供的数据结构、表中文名和用户问题，完成以下任务：

        【输入说明】
        - data: 一个包含字典的列表，每个字典代表一行数据。
        - table_name: 数据来源表的中文名称。
        - query: 用户提出的具体问题。

        【任务要求】
        1. **判断是否可以绘制图表**：  
           - 必须同时具备分类维度（如时间、地区等）和数值指标（金额、数量、百分比等）。  
           - 数值字段需可解析为数字；分类字段需有至少两个不同值。  
           - 若不满足，返回 {{"can_plot": 0}}。

        2. **若可绘图，请执行以下子任务**：
           a. **确定最合适的图表类型**（三选一）：
              - 柱状图：适合比较离散类别的数值（如各月支出）。
              - 折线图：适合展示时间序列趋势（如连续月份变化）。
              - 饼图：仅当分类数量 ≤ 12 且强调占比时使用（通常不适用于时间序列）。
              ***注意***：绘图代码支持折线+柱状图的复合图表。如果data中的字典列表中数值指标超过2个，自行判断合适的绘图方式。

           b. **提取x轴和y轴要素**：
              - x轴首选年月，地区等分类维度的字段，如果时间类型字段不存在则地区字段次之。
              - x轴应为分类字段（如“业务年度”,“地区名称”等），时间类字段需标准化为可读格式（如 '2025年2月'）。
              - y轴应为数值字段（如“本月数—金额”,“同比”等字段），需将字符串转为浮点数保留两位小数。
              - y轴数据中出现万元、万吨等单位时默认该数据就是此单位。当出现%结尾时该数据是百分比数据。百分比数据一般用折线图
              - 对于饼图，x 轴即为类别，y 轴为对应数值。

           c. **生成图表元信息**：
              - 主题：结合 table_name 和 query，简洁概括（如“2025年2月至10月教育支出趋势”）。
              - x_axis_label：x 轴含义（如“月份”）。
              - y_axis_label：y 轴含义（如“支出金额（万元）”）。
              ***注意***： y_axis_label是一个列表。例如["支出金额（万元）","本年上月累计数（万元）"]
           d. **输出标准化数据格式**（无论选哪种图，都按以下字段输出）：
              - categories: x 轴标签列表（按时间或自然顺序排序）。
              - values: 与x轴对应的y值列表（与 categories 一一对应）。
              ***注意***：values是一个列表。
              1.当输出饼图时格式为：[{{"pie":[100.0,200.0,300.0]}}]
              2.当输出柱状图、折线图、柱状+折线时格式为：[{{"bar":[100.0,200.0,300.0]}},{{"line":[80.0,85.0,88.0]}}]
              3.y_axis_label的标题信息的位置应和values一一对应

        【输出格式】
        严格以 JSON 格式输出，包含以下字段：
        {demo_data1}

        【示例输入】
        data = [{{'业务年度': '202508', '本月数—金额': '1161668.000000万元'}}, {{'业务年度': '202507', '本月数—金额': '1263361.000000万元'}}, {{'业务年度': '202509', '本月数—金额': '1870492.000000万元'}}]
        table_name = "预算审查监督-预算执行表-全省一般公共预算支出完成情况表"
        query = "2025年2月到202510月期间教育支出的本月金额分别是多少？"
        **补充说明**（供你内部参考）
        时间格式标准化：202502 → '2025年2月'，注意补零（2月而非02月）。
        数值处理：去除“万元”后缀，转为 float，保留两位小数（但输出可为整数或 float）。
        排序逻辑：按业务年度字符串升序（即时间顺序）排序后再提取 categories 和 values。
        图表选择逻辑：
        本例是 连续月份的时间序列 → 优先折线图（展示趋势），其次柱状图也可接受。
        饼图不合适（时间序列 + 强调绝对值而非占比）。
        【示例输出（预期结果）】
        {demo_data2}
        *注*：1.values和y_axis_label位置一一对应。当用户问题中明确提出绘制某种图时优先考虑使用该类型。但是数据明显不符合条件则可放弃，改用最适合的方式。
             2.一般年度、地区数值对比类最适合柱状图，当用户明确提到使用饼图时结合数据要素自行判断，百分比类型的适合用折线图
        【输出前检查】
        *注意*：输出严格限制为json格式。  
        **用户输入**:
        data={data}
        table_name={table_name} 
        query={question}
        **输出**:
        """
    app.state.demo_data1 = {
        "can_plot": 1,  # 在1/0中选择输出
        "title": "图表主题",
        "x_axis_label": "x轴名称",
        "y_axis_label": ["y轴名称"],
        "categories": ["2022年", "..."],
        "values": [{"bar": [100.0, 200.0, 300.0]}, ],

    }
    app.state.demo_data2 = {
        "can_plot": 1,
        "title": "2025年2月至10月教育支出金额趋势",
        "x_axis_label": "月份",
        "y_axis_label": ["支出金额（万元）"],
        "categories": ["2025年2月", "2025年3月", "2025年4月"],
        "bar_values": [{"bar": [1052312.00, 2028028.00, 1463275.00]}],
    }
    app.state.prompt3 = """你是一个 **NL2SQL 前处理助手**。你的任务是将用户原始问题整理为标准化查询模板，供后续 SQL 生成模型使用。**不生成 SQL，不输出 JSON，不输出解释，不输出推理过程，不输出额外说明**。

## 输入
- 用户原始问题
- 最相关表名
- 表 schema
- 命中的项目编码
- 查询相关列候选
- 区划编码
- 当前日期
- chart_hint（可选，取值可能为：line、bar、pie、auto）

---

## 任务
请根据输入补全以下内容：
- 时间范围
- 区划编码
- 项目编码
- 核心意图
- 查询相关列
- 指标列表
- 分组维度
- 排序字段
- 排序方向
- 条数限制

---

## 总体要求
1. 你的输出是一个**标准化模板问题**，用于帮助后续模型稳定理解查询意图。
2. **禁止输出任何未替换的槽位词**，例如“【时间范围】”“【指标列表】”等。
3. **禁止臆造 schema 或候选列中不存在的原生字段名**。
4. 但“**指标列表**”不等同于原生字段名：
   - **明细查询、趋势分析**：优先使用原生字段或原生维度列；
   - **非明细、非趋势分析**时，`指标列表` 应按**最终结果实际返回的指标形态**输出，可以是语义化聚合指标名，**不要求必须是 schema 原生列名**；
   - 例如可以输出：`业务费用求和`、`彩票发行机构业务费用求和`、`记录数`、`用户数去重计数`、`平均金额`；
   - 但这些语义化指标必须能够由“查询相关列 + 用户问题语义”推出，不能凭空捏造业务概念。
5. `查询相关列` 与 `指标列表` 是两个不同层次：
   - **查询相关列**：底层取数依赖的原生列或候选列；
   - **指标列表**：最终结果中应返回的列或指标，可为原生列，也可为聚合后的语义指标名。

---

## 规则

### 1）项目编码
- 项目编码直接使用输入提供的值，不提取、不改写。
- 最终输出中：
  - 若项目编码非空，写“项目编码为【项目编码】”
  - 若项目编码为空，则不要输出项目条件。

---

### 2）查询相关列
- 若提供了“查询相关列候选”，优先直接使用候选列。
- 若未提供，则结合用户问题和 schema 选择最相关的**数值列**。
- 若用户问题与 schema 中数值列相关性较低，则优先寻找与以下含义相关的列：
  - 金额
  - 费用
  - 收入
  - 支出
  - 数量
  - 数值
  - 当月金额
  - 本月金额
  - 累计金额
  - 合计金额
- `查询相关列` 必须来自 schema 或候选列，不能臆造。

---

### 3）时间范围（重点）
- 数据是**年月粒度**，时间范围必须规范输出为：`YYYY年MM月-YYYY年MM月`
- **严禁输出“时间范围未知”“按默认时间范围处理”之类表述**
- 若用户没有明确提到时间范围，**默认使用当前月**
- 若用户写的是模糊时间表达，必须换算成明确时间范围，不能原样保留

#### 时间识别规则
- “2025年” / “25年” → `2025年01月-2025年12月`
- “2025年3月” / “25年3月” → `2025年03月-2025年03月`
- “2025年1-3月” / “2025年1到3月” → `2025年01月-2025年03月`
- “2025年全年” → `2025年01月-2025年12月`
- “2025年各月” / “2025年各月份” / “25年每月” → `2025年01月-2025年12月`
- “本月” → 当前月
- “上个月” → 上一月
- “今年” → 当前年份01月-12月
- “去年” → 上一年01月-12月
- “近3年” → 当前年份往前推2年后的1月，到当前年份12月
- 两位年份统一按 2000 年后处理，如 25年=2025年

---

### 4）核心意图（只能选 1 个）
核心意图只能从以下 10 类中选择：
- 明细查询
- 统计
- 计数
- 去重计数
- 平均值
- 最大值
- 最小值
- 趋势分析
- 对比分析
- 分布统计

#### 意图判定规则
- 明细/列表/记录/原始数据/逐条/详情/情况/分别是多少 → 明细查询
- 条数/多少条/数量/次数 → 计数
- 去重/独立用户/唯一实体数 → 去重计数
- 平均/均值/人均 → 平均值
- 最大/最高/峰值 → 最大值
- 最小/最低 → 最小值
- 趋势/走势/变化趋势/各月变化/逐月变化/按月变化 → 趋势分析
- 对比/比较/差异/分别对比/不同区域差异/不同对象差异 → 对比分析
- 分布/占比/构成/分桶/分类分布 → 分布统计

- 若是问题未确定明确类型,默认为明细查询


#### 年月粒度数据下的特殊约束
- 你的数据是**年月粒度汇总数据**
- 当用户问题是“2025年A和B分别是多少”优先识别为【明细】
---

### 5）指标列表（重点优化）
`指标列表` 表示**最终查询结果中实际要返回的字段或指标**，必须优先满足结果展示和图表绘制需要。

#### 基本原则
- `指标列表` 不要求必须来自 schema 原生字段名。
- 除 **明细查询** 和 **趋势分析** 外，其它场景下 `指标列表` 应优先输出**结果态指标名**，而不是机械复用原生列名。
- 所谓“结果态指标名”，是指经过聚合、统计、计数、比较后，最终返回结果中更合理的名字。

#### 指标列表生成规则
1. **明细查询**
   - 返回原生字段，不做聚合
   - `指标列表` 直接写原始记录字段或用户关注字段

2. **趋势分析**
   - 返回“时间维度 + 原生指标列”
   - 趋势分析优先保留原生指标表达，不强制改成“求和/平均值”等后缀，除非用户语义明确要求
   - 例如：`年月, 业务费用`


3. **计数**
   - 输出 `记录数`、`数量`、`次数` 或更贴近语义的计数指标名
   - 例如：`记录数`

4. **去重计数**
   - 输出 `X去重计数`、`独立用户数`、`唯一项目数`
   - 例如：`用户ID去重计数`

5. **平均值**
   - 输出 `X平均值`
   - 例如：`业务费用平均值`

6. **最大值**
   - 输出 `X最大值`

7. **最小值**
   - 输出 `X最小值`

8. **对比分析**
   - 若是按维度对比：输出 `对比维度 + 指标`
   - 若是多个指标分别比较：可输出 `A求和, B求和` 或 `A, B`，以最终结果最清晰为准
   - 若问题重点是“分别是多少”，即使涉及多个对象，也优先将结果指标逐个展开，不得合并成单一泛化指标

9. **分布统计**
   - 输出 `分类维度 + 指标`
   - 若是占比类，可输出 `分类维度, 占比`
   - 若是数量分布，可输出 `分类维度, 记录数`

10. **统计**
   - 若问题只表达“统计一下”但未明确聚合方式，`指标列表` 应尽量贴近最终返回结果的自然语义，不要机械回填原生列名
   - 例如：`业务费用统计值`、`销售金额统计值`
   - 但如果可从语义中明显判断是求和/计数等，应直接归入对应意图，不使用“统计值”兜底

#### 特殊硬约束
- 当原始列只是底层承载列，而用户问题实际指向多个语义指标时，`指标列表` 应按**语义结果**输出，而不是照搬底层列名
- `指标列表` 必须体现“最终结果要返回什么”，而不是仅体现“底层从哪列取数”

---

### 6）分组维度
- 按区域统计时，应返回区域字段
- 按项目统计时，应返回项目字段
- 按时间统计时，应返回时间粒度字段
- 若用户提到“各月 / 每月 / 分月 / 走势 / 趋势”，优先补全为**年月字段**
- 若只是问“2025年xxx是多少”，不要自动补时间分组，应按全年汇总处理
- 若适合图表展示但缺少必要维度列，应结合 schema 和问题补全最合理的维度列

---

### 7）chart_hint
- `line`：优先保证返回“时间维度 + 指标”
- `bar`：优先保证返回“类别维度或时间维度 + 指标”
- `pie`：优先保证返回“分类维度 + 指标”
- `auto` 或未提供：按问题语义判断最合理返回列
- `chart_hint` 只用于辅助补全返回列，**不改变核心意图**

---

### 8）输出格式（严格遵守）
在【实际时间范围】内，查询项目编码为【项目编码】的【查询相关列】数据。  
查询类型为：【核心意图】。  
结果中返回以下字段或指标：【指标列表】。  
【如存在分组维度，则补充：按【分组维度】统计。】  
【如存在排序，则补充：结果按【排序字段】的【排序方向】排序。】  
【如存在条数限制，则补充：返回【条数限制】条结果。】  

---

## 输入内容
【用户原始问题】  
{user_question}

【最相关表名】  
{table_name}

【表 schema】  
{table_schema}

【命中的项目编码】  
{matched_project_name}

【查询相关列候选】  
{relevant_columns_candidates}

【区划编码】  
{region_value}

【当前日期】  
{now_date}

【chart_hint】  
{chart_hint}

请只输出最终填充后的模板问题，不要输出解释。
    """
    app.state.prompt4 = """
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


@app.on_event("shutdown")
async def shutdown():
    app.state.redis.close()


def wait_info(search_point, check_sheng, use_table_info, source):
    wait_tables = []
    for x in search_point.points:
        for n, y in enumerate(x.payload['zh_table']):
            if use_table_info.get(y):
                if check_sheng:
                    if use_table_info[y]['is_sheng']:
                        wait_tables.append({"table": y, "score": x.score, "source": source})
                else:
                    if not use_table_info[y]['is_sheng']:
                        wait_tables.append({"table": y, "score": x.score, "source": source})
    return wait_tables




def model_chat(chat, question, prompt):
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
        "timeout": api_config["timeout"]
    }
    # params_copy["extra_body"] = {"thinking": {"type": "disabled"}}
    response = chat.chat.completions.create(**create_params)
    ret = response.choices[0].message.content
    return ret


def create_image(categories, values, title, y_titles):
    chart_colors = [
        '#5470C6',  # 深蓝色（参考示例）
        '#91CC75',  # 鲜绿色
        '#EE6666',  # 亮红色
        '#FAC858',  # 金黄色
        '#73C0DE',  # 天蓝色
        '#3BA272',  # 翠绿色
        '#FC8452',  # 橙红色
        '#9A60B4',  # 紫色
        '#E5A3C9',  # 粉红色
        '#FADB7A',  # 淡黄色
        '#A0A7D4',  # 淡紫色
        '#7FCFC0',  # 薄荷绿
        '#F9B8B8',  # 浅粉色
        '#B1D580',  # 嫩绿色
        '#E6965C',  # 橙色
        '#C77EB5',  # 紫罗兰
        '#6DC8BF',  # 蓝绿色
        '#FFB7A5',  # 珊瑚色
        '#D9A7E0',  # 薰衣草紫
        '#A8D5E5'  # 浅天蓝
    ]
    # chart = Echart(title)
    # 初始化图表
    chart = (
        Bar(init_opts=opts.InitOpts(width="100%", height="600px"))
        if any("bar" in v for v in values)
        else Line(init_opts=opts.InitOpts(width="100%", height="600px"))
        if any("line" in v for v in values)
        else Pie(init_opts=opts.InitOpts(width="100%", height="600px"))
    )

    # 设置标题
    chart.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            bottom=0,
            left="center"
        )
    )

    use_tip = None
    is_xy = True

    def get_unit(text):
        match = re.search(r'（(.*?)）', text)
        return match.group(1) if match else ""

    # 循环添加系列
    for n, x in enumerate(values):
        if not chart_colors:
            chart_colors = [
                '#5470C6', '#91CC75', '#EE6666', '#FAC858', '#73C0DE'
            ]
        color = random.choice(chart_colors)
        chart_colors.remove(color)

        if "bar" in x:
            chart.add_yaxis(
                series_name=y_titles[n],
                y_axis=x["bar"],
                itemstyle_opts=opts.ItemStyleOpts(color=color),
            )

        elif "line" in x:
            chart.add_yaxis(
                series_name=y_titles[n],
                y_axis=x["line"],
                linestyle_opts=opts.LineStyleOpts(width=2, color=color),
                itemstyle_opts=opts.ItemStyleOpts(color=color),
                symbol="circle",
                symbol_size=8,
                is_smooth=True,
            )

        elif "pie" in x:
            is_xy = False
            pie_data = [
                opts.PieItem(name=categories[i], value=val)
                for i, val in enumerate(x["pie"])
            ]
            chart = Pie(init_opts=opts.InitOpts(width="100%", height="600px"))
            chart.add(
                series_name=y_titles[n],
                data_pie=pie_data,
                radius="55%",
                center=["50%", "60%"],
                label_opts=opts.LabelOpts(
                    formatter="{b}:{d}%",
                    position="outside",
                ),
            )
            chart.set_global_opts(
                legend_opts=opts.LegendOpts(
                    item_gap=12,
                    textstyle_opts=opts.TextStyleOpts(font_size=12),
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="item",
                    formatter="{a}<br/>{b}: {c}"
                ),
                title_opts=opts.TitleOpts(title=title, bottom=0, left="center"),
            )

    # ==================== XY 轴图表（柱状/折线）====================
    if is_xy:
        chart.add_xaxis(categories)
        unit_list = []
        tip_formatter = ""

        if len(y_titles) <= 1:
            unit = get_unit(y_titles[0])
            tip_formatter = f"x轴:{{b}}<br/>{{a}}: {{c}}{unit}"
            unit_list.append(unit)
        else:
            tip_formatter = "x轴:{b}<br/>"
            for i, y in enumerate(y_titles):
                u = get_unit(y)
                unit_list.append(u)
                tip_formatter += f"{{a{i}}}: {{c{i}}}{u}"
                if i != len(y_titles) - 1:
                    tip_formatter += "<br/>"

        chart.set_global_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                formatter=tip_formatter,
                background_color="rgba(50,50,50,0.9)",
                textstyle_opts=opts.TextStyleOpts(color="#fff"),
                border_color="#333",
                border_width=1,
            ),
            legend_opts=opts.LegendOpts(
                left="center",
                top=20,
                textstyle_opts=opts.TextStyleOpts(font_size=12),
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                splitline_opts=opts.SplitLineOpts(
                    is_show=True, linestyle_opts=opts.LineStyleOpts(type_="dashed", color="#eee")
                ),
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                name="数值",
                name_location="end",
                name_gap=15,
                splitline_opts=opts.SplitLineOpts(
                    is_show=True, linestyle_opts=opts.LineStyleOpts(type_="dashed", color="#eee")
                ),
            ),
        )

        # 双 Y 轴（万元 + %）
        unique_units = set(u for u in unit_list if u)
        if len(unique_units) > 1 and "%" in unit_list:
            money_values = []
            percent_values = []
            for i, y in enumerate(unit_list):
                vals = list(values[i].values())[0]
                if y == "万元":
                    money_values.extend(vals)
                elif y == "%":
                    percent_values.extend(vals)

            if money_values:
                max_m = max(money_values)
                min_m = min(money_values)
                chart.options["yAxis"][0]["min"] = 0
                chart.options["yAxis"][0]["max"] = max_m * 1.1

            if percent_values:
                max_p = max(percent_values)
                min_p = min(percent_values)
                chart.options["yAxis"].append({
                    "type": "value",
                    "position": "right",
                    "name": "百分比 (%)",
                    "nameLocation": "end",
                    "min": min_p * 0.9,
                    "max": max_p * 1.1,
                    "axisLabel": {"formatter": "{value} %"},
                    "splitLine": {"show": False},
                })

                for i, u in enumerate(unit_list):
                    if u == "%":
                        if "series" in chart.options:
                            chart.options["series"][i]["yAxisIndex"] = 1

    # 最终返回配置（和你原来用法完全一样！）
    config = chart.dump_options()
    if isinstance(config, str):
        config = json.loads(config)
    return config


def split_select_fields(select_part: str):
    """
    按顶层逗号切分 SELECT 字段，兼容函数/子表达式中的逗号
    例如：
    SUM(A, B), C AS 别名, YEAR_MONTH
    -> ['SUM(A, B)', 'C AS 别名', 'YEAR_MONTH']
    """
    fields = []
    buf = []
    depth = 0

    for ch in select_part:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            fields.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)

    if buf:
        fields.append(''.join(buf).strip())

    return fields


def remove_alias(field: str):
    """
    删除 SELECT 字段中的 AS 别名：
    `BYS_JE` AS 本月数_金额  -> `BYS_JE`
    SUM(A) AS 合计          -> SUM(A)

    只删除顶层 AS，避免影响表达式内部嵌套
    """
    buf = []
    depth = 0
    i = 0
    upper_field = field.upper()

    while i < len(field):
        ch = field[i]

        if ch == '(':
            depth += 1
            buf.append(ch)
            i += 1
        elif ch == ')':
            depth -= 1
            buf.append(ch)
            i += 1
        elif depth == 0 and upper_field[i:i + 4] == ' AS ':
            # 顶层遇到 AS，直接截断，丢弃后面的别名
            return ''.join(buf).strip()
        else:
            buf.append(ch)
            i += 1

    return ''.join(buf).strip()


def ensure_year_month_in_select(sql: str, sss) -> str:
    """
    规则：
    1. 如果 WHERE 子句中包含 YEAR_MONTH，则 SELECT 中必须包含 YEAR_MONTH
    2. 删除 SELECT 字段中的 AS 别名
    3. 保持原 SQL 其他部分不变
    4. 不做大小写转换，适配全大写 SQL
    """
    sql = sql.strip().rstrip(';')

    # 提取 SELECT / FROM / WHERE / 后续部分
    pattern = re.compile(
        r'^\s*SELECT\s+(?P<select>.*?)\s+FROM\s+(?P<from>.*?)(\s+WHERE\s+(?P<where>.*?))?(\s+ORDER\s+BY\s+(?P<orderby>.*))?$',
        re.S
    )
    m = pattern.match(sql)
    if not m:
        raise ValueError("SQL 格式不符合预期，无法解析。")

    select_part = m.group('select').strip()
    from_part = m.group('from').strip()
    where_part = (m.group('where') or '').strip()
    orderby_part = (m.group('orderby') or '').strip()

    # 1. 拆分 SELECT 字段
    fields = split_select_fields(select_part)

    # 2. 删除别名
    fields_no_alias = [remove_alias(f) for f in fields]

    # 3. 如果 WHERE 中有 YEAR_MONTH，则确保 SELECT 中包含 YEAR_MONTH
    if sss:
        has_year_month_in_where = sss in where_part
        has_year_month_in_select = any(f.strip() == sss for f in fields_no_alias)
    else:
        has_year_month_in_where = False
        has_year_month_in_select = False

    if has_year_month_in_where and not has_year_month_in_select:
        fields_no_alias.insert(0, f'`{sss}`')

    # 去重，保持顺序
    dedup_fields = []
    seen = set()
    is_limit_exit = 0
    for f in fields_no_alias:
        if f not in seen:
            if f in ["YEAR_MONTH", "DATE_YEAR"] and is_limit_exit:
                continue
            dedup_fields.append(f)
            seen.add(f)
            if f in ["YEAR_MONTH", "DATE_YEAR"]:
                is_limit_exit = 1
    # 4. 重组 SQL
    new_sql = f"SELECT {', '.join(dedup_fields)} FROM {from_part}"
    if where_part:
        new_sql += f" WHERE {where_part}"
    if orderby_part:
        new_sql += f" ORDER BY {orderby_part}"

    return new_sql + ';'


def ret_format(msg,stop=None):
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
            "finish_reason": stop
        }]
    }) + "\n\n"


def extract_non_numeric(text):
    """
    提取字符串中的非数字部分（移除数字和小数点）
    """
    # 匹配所有非数字和非小数点的字符
    result = re.sub(r'[\d.-]', '', text)
    return result


def extract_numeric_value(text):
    """
    提取字符串中的数字并转换为浮点数
    """
    match = re.search(r'-?[\d.]+', text)
    if match:
        return round(float(match.group()), 2)
    return 0.0


def reshape_for_chart(
        data,
        date_col='业务年月',
        category_cols=('区域名称', '科目名称'),
        value_cols=('本月数-金额',),
        value_name_map=None,
        sep='-'
):
    if value_name_map is None:
        value_name_map = {}

    grouped = defaultdict(dict)

    for row in data:
        date_value = row[date_col]
        grouped[date_value][date_col] = date_value

        # 拼接类别部分
        category_parts = [str(row[col]) for col in category_cols if col in row]

        # 展开多个指标列
        for val_col in value_cols:
            val_display_name = value_name_map.get(val_col, val_col)
            new_col = sep.join(category_parts + [val_display_name])
            grouped[date_value][new_col] = row.get(val_col)

    # 按业务年月排序输出
    return [grouped[k] for k in sorted(grouped.keys())]


def select_x(all_keys):
    # x轴的筛选原则1.优先年月 2.地区次之 3.不带单位
    for i in all_keys:
        if any(char in i for char in ['年度', "年月"]):  # 挑选一个x轴
            ret = i
            break
        if any(char in i for char in ['名称', "项目"]):
            ret = i
            break
    else:
        ret = all_keys[0]
    return ret


def creat_image_data(data, x_axis_label, question, app):
    # [{'业务年月': '202501', '本月数-金额': '0.0000万元'}, {'业务年月': '202502', '本月数-金额': '34.0000万元'}]
    # {
    #   "can_plot": 1,
    #   "title": "2025年1月至5月省级国防支出本月金额",
    #   "x_axis_label": "月份",
    #   "y_axis_label": ["本月金额（万元）"],
    #   "categories": ["2025年1月", "2025年2月", "2025年3月", "2025年4月", "2025年5月"],
    #   "values": [{"bar": [0.0, 34.0, 3481.0, 183.0, 53.0]}]
    # }
    app.state.redis.set(question.strip() + "_task_cache", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ex=3600)
    # 这个设计在于只有成功执行才会返回图像json,不适合画图和报错都不存缓存
    try:
        json_data = {
            "can_plot": 1,
            "title": "",
            "x_axis_label": x_axis_label,
            "y_axis_label": [],
            "categories": [],
            "values": [],
        }
        # 不适合画图的情况只有一列，行数为0
        if len(data) == 1 or len(data[0]) <= 1:
            json_data["can_plot"] = 0
        else:
            all_keys = list(set(data[0].keys()))
            # json_data['x_axis_label'] = select_x(all_keys)
            y_axis_label = [i for i in all_keys if i != json_data["x_axis_label"]]
            # 如果用户问题中提及饼图，优先判断是否符合饼图构建要素
            is_pie = 0
            if "饼图" in question and len(all_keys) == 2:
                is_pie = 1

            merry = []
            # 当没提及时默认走柱状图和折线图
            # 什么时候走折线和柱状图的混合
            for w in data:
                json_data['categories'].append(w[json_data["x_axis_label"]])
                if is_pie:
                    if json_data['values']:
                        json_data['values'][0]['pie'].append(extract_numeric_value(w[y_axis_label[0]]))
                    else:
                        json_data['values'] = [{"pie": [extract_numeric_value(w[y_axis_label[0]])]}]

                else:
                    # 这里暂时规则为金额数据走柱状图，百分比走折线图
                    if json_data['values']:
                        for n, i in enumerate(y_axis_label):
                            json_data['values'][n][merry[n]].append(extract_numeric_value(w[i]))
                    else:
                        for n, i in enumerate(y_axis_label):
                            if "%" in w[i]:
                                name = "line"
                            else:
                                name = "bar"
                            merry.append(name)
                            json_data['values'].append({name: [extract_numeric_value(w[i])]})

            json_data['y_axis_label'] = [i + f"（{extract_non_numeric(data[0][i])}）" for i in all_keys if
                                         i != json_data["x_axis_label"]]

        # prompt = app.state.prompt2.format(data=data, table_name=table_name,
        #                                   question=question, demo_data1=app.state.demo_data1,
        #                                   demo_data2=app.state.demo_data2)
        # create_params2 = {
        #     "model": "deepseek-chat",
        #     "messages": [
        #         {"role": "system", "content": prompt},
        #         {"role": "user", "content": question},
        #     ],
        #     "temperature": 0.7,
        #     "max_tokens": 8192,
        #     "stream": False,
        #     "timeout": api_config["timeout"]
        # }
        # response2 = app.state.chat.chat.completions.create(**create_params2)
        # ret2 = response2.choices[0].message.content
        # print("后台任务：",ret2)
        # # 解析ret2
        # if "```json" in ret2:
        #     json_data = json.loads(ret2.split("```json")[1].split("```")[0])
        # else:
        #     json_data = json.loads(ret2)
        if json_data["can_plot"]:
            categories = json_data['categories']
            values = json_data['values']
            title = json_data['title']
            y_titles = json_data['y_axis_label']
            chart_config = create_image(categories, values, title, y_titles)
            app.state.redis.set(question.strip() + "<-split->cache", json.dumps(chart_config, ensure_ascii=False),
                                ex=3600)

    except Exception as e:
        print("执行后台任务报错：", e)
        pass
    finally:
        app.state.redis.delete(question.strip() + "_task_cache")

def check_select_col(col_dict,select_list):
    check = []
    for x in select_list:
        if "(" in x:
            name = x.split("(")[1].strip(")")
            check.append(name in col_dict)
        elif "`" in x:
            name = x.strip("`")
            check.append(name in col_dict)
        else:
            check.append(x in col_dict)
    return all(check)


def get_agg_name(x):
    all_dd = {"COUNT": "计数", "SUM": "求和", "AVG": "平均值", "MIN": "最小值", "MAX": "最大值"}
    ret = ""
    for w,y in all_dd.items():
        if w in x:
            ret = y
            break
    else:
        ret = ""
    return  ret

@app.post("/v1/chat/completions")
async def sql3(request: Request, background_tasks: BackgroundTasks):
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
    print("request_json:", request_json)
    # 2. 提取参数 (Dify 会自动带上 tools 参数如果配置了工具)
    messages = request_json.get("messages", [])
    question = messages[-1].get("content", "")
    stream = request_json.get("stream", False)
    print(question, "question")
    # TODO支持结合记忆情况对问题和数据进行适当优化。
    # TODO:问题可能会被拆分成两个表的查询语句，目前先不考虑连表和多表查询情况
    question_words = app.state.words_match.find_all(question)

    # TODO:1.多表数据对比类问题 2，数值比对问题比如小于1亿的问题
    # all_info = {}
    # 如果问题中没有出现明确的省本级意图，(省级、本级)
    # 清理结果中的省本级的表。如果出现省本级字眼，其他跟省本级无关的表剔除
    # 1.先找到所有科目相关的关键词对问题进行分词，判断关键词是否出现，此动作可以直接定位几张表
    # 2.对问题直接进行表描述级的向量检索，此动作也可以找到相关的top3表
    # 3.如果关键词没找到，直接根据表检索结果确定
    # 4.对于4本账的情况分清楚省本级，省级的表定位
    async def generate_progress():
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
                    if ww.payload["words"] not in project_words:
                        project_words.append(ww.payload["words"])
                    if app.state.use_table_info[u]['project_key']:
                        if u not in data1:
                            data1[u] = [{ww.payload["words"]: ww.payload["select_code"][n]}]
                        else:
                            data1[u].append({ww.payload["words"]: ww.payload["select_code"][n]})
                    else:
                        if u not in data1:
                            data1[u] = [{ww.payload["words"]: ww.payload["table_select"][n]}]
                        else:
                            data1[u].append({ww.payload["words"]: ww.payload["table_select"][n]})

            wait_tables.extend(wait_info(existing_points, check_sheng, app.state.use_table_info, source="项目匹配"))
        yield ret_format(f"已找到{len(project_words)}个关键词，分别为{','.join(project_words)}\n\n")
        try:
            query_vector = get_embedding1([question])['embeddings'][0]
        except:
            yield ret_format("模型服务异常，请联系管理员维护！",stop="stop")
            return
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
        zone_words = app.state.zone_match.find_all(question)

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
                    if yy.payload["words"] not in col_words2:
                        col_words2.append(yy.payload["words"])
                    if u not in data2:
                        data2[u] = [{yy.payload["words"]: yy.payload["table_select"][n]}]
                    else:
                        data2[u].append({yy.payload["words"]: yy.payload["table_select"][n]})
            wait_tables.extend(wait_info(col_points, check_sheng, app.state.use_table_info, source="列匹配"))
        yield ret_format(f"已找到{len(col_words2)}个列关键词，分别为{','.join(col_words2)}\n\n")
        ranker = Text2SQLTableRanker()
        print("wait_tables:", wait_tables)
        result = ranker.rank(wait_tables)
        if result.get("status") == "SUCCESS":
            yield ret_format(f"经过权重算法分析选中最相关表为：{result['selected_table']}。\n\n")
            table_name = result['selected_table']
            demo = app.state.public_data[table_name]
            # pk =  app.state.use_table_info[table_name]['project_key']
            # if pk:
            #     name = pk[0]
            # else:
            #     name = app.state.use_table_info[table_name]['project_name'][0]

            unit_dict = {value: key for key, values in app.state.use_table_info[table_name]['unit'].items() for value in
                         values}
            # TODO:基于大模型回复不稳定的问题提出优化方案
            if "柱状图" in question:
                chart_hint = "bar"
            elif "饼图" in question:
                chart_hint = "pie"
            elif "折线图" in question:
                chart_hint = "line"
            else:
                chart_hint = "auto"
            cols_names = data2.get(table_name, [])
            old_col_names = []
            for x in cols_names:
                for k, v in x.items():
                    print(k, v, "列缓存.....")
                    if v:
                        old_col_names.append(f"{k}={v}")
            # 将用户问题中的命中列关键词替换为原始列名
            relevant_columns_candidates = "、".join(old_col_names)  # 命中项目编码
            print("命中的替换列字段:", relevant_columns_candidates)

            if zone_words:
                region_value = "、".join(set([app.state.zone_dict.get(i, "130000000") for i in zone_words]))
            else:
                # region_value = "130000000"  # 区域code
                region_value = ""  # 区域code

            if question_words:
                print(data1.get(table_name, []), "项目字典....")
                # TODO:近一年丰南区，本级国有资本经营预算中的国有企业退休人员社会化管理补助支出的累计同比额情况
                # 预算执行-本级国有资本经营预算支出完成情况表
                # [{'国有企业退休人员社会化管理补助支出': '2230105'}, {'社会化管理补助': None}, {'国有资本经营预算': '223'}]
                # 当相关项目词有包含情况出现时该如何选择合适的项目。

                matched_project_code = "、".join(
                    [value for item in data1.get(table_name, []) for value in item.values() if
                     value is not None])  # 命中项目编码
            else:
                # 当问题相关度很高但是并没有命中项目时根据此表返回一个最相关的项目
                query_filter22 = Filter(
                    must=[
                        FieldCondition(
                            key="cate",
                            match=MatchValue(value="科目/项目")
                        ),
                        FieldCondition(
                            key="zh_table",
                            match=MatchValue(value=table_name)
                        )])
                results22 = app.state.client.query_points(
                    collection_name=app.state.collection_name,
                    query=query_vector,
                    query_filter=query_filter22,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=None  # 如果需要最低相似度阈值，可在此设置 (如 0.7)
                )
                for zz in results22.points:
                    if table_name in zz.payload["zh_table"]:
                        index = zz.payload["zh_table"].index(table_name)
                        matched_project_code = zz.payload["select_code"][index]
                        break
                else:
                    matched_project_code = "201"
                    print("没找到最相关的编码...触发兜底")
            prompt4 = app.state.prompt3.format(
                user_question=question, table_name=table_name,
                table_schema=demo, matched_project_name=matched_project_code,
                relevant_columns_candidates=relevant_columns_candidates,
                region_value=region_value, chart_hint=chart_hint,
                now_date=datetime.datetime.now().strftime("%Y-%m-%d")
            )

            # prompt4 = app.state.prompt4.format(name=name,
            #                        data1=data1.get(table_name, {}), data2=data2.get(table_name, {}),
            #                     )
            new_question = model_chat(app.state.chat, question, prompt4)
            print(new_question, "--改良后的问题")
            yield ret_format(f"问题已优化为：{new_question}\n\n")
            # 如果选中了表，那么根据选择过程中的数据去优化问题
            # 这里要使用一次模型
            # 5.根据选中的表进行sql生成,要求每个步骤把定位的内容实时输出到控制台

            en_table_name = app.state.use_table_info[table_name]['table']
            col_name_dict = {k: v['comment'] for k, v in demo['tables'][en_table_name]['fields'].items()}
            # TODO:优化evidence 为列映射和模式说明
            evidence = """
               question：2025年12月项目名称为个人所得税的预算数是多少？
               answer: SELECT `YEAR_MONTH`,`YSS`  FROM `RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC` WHERE `YEAR_MONTH` = '202512' AND `XM_NAME` = '个人所得税';
               question：2025年2月到2025年10月期间科目编码为205的本月金额分别是多少？
               answer:SELECT `YEAR_MONTH`,`BYS_JE` FROM RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC WHERE `YEAR_MONTH` BETWEEN '202502' AND '202510' AND `XM_CODE` = '205'  ORDER BY XH;
               """
            sql_query_json = get_text2sql(new_question, demo, evidence)
            if sql_query_json['status']=='success':
                sql_query = sql_query_json['sql_query']
            else:
                yield ret_format(f"模型服务异常，请联系管理员！",stop="stop")
                return

            # 优化sql生成过程不带符号的问题
            # 默认使用序号排序
            try:
                table_data_col = list(demo['tables'].values())[0]['fields'].keys()
            except:
                table_data_col = None
                print("没有找到table_data_col。。。")
            if table_data_col:
                for u in table_data_col:
                    if u + ' ' in sql_query and f"`{u}` " not in sql_query:
                        sql_query = sql_query.replace(u + ' ', f"`{u}` ")
                    if u + ',' in sql_query and f"`{u}`," not in sql_query:
                        sql_query = sql_query.replace(u + ',', f"`{u}`,")
                    if u + ';' in sql_query and f"`{u}`;" not in sql_query:
                        sql_query = sql_query.replace(u + ';', f"`{u}`;")

            # 这里只是为了补上必要的时间字段
            match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE)
            if match:
                fields_str = match.group(1)
                # 判断筛选字段是否都是y轴字段,如果都是y轴数值字段，不具备画图要素默认填充一个时间字段
                fields = split_select_fields(fields_str)
                fields_no_alias = [remove_alias(f) for f in fields]
                print("fields_no_alias:", fields_no_alias)
                if check_select_col(col_name_dict, fields_no_alias):
                    if 'YEAR_MONTH' in sql_query and "YEAR_MONTH" not in fields_str:
                        sql_query = ensure_year_month_in_select(sql_query, "YEAR_MONTH")

            print("sql_query:", sql_query)
            print("执行 SQL 查询...")
            yield ret_format(f"已生成 SQL 语句为：{sql_query}\n\n")
            
            try:
                print("正在执行 SQL 查询...")
                # 使用 SQLAlchemy 执行 SQL 查询
                with (db_engine.connect() as connection):
                    # 使用 text() 包装 SQL 语句以支持原生 SQL
                    result = connection.execute(text(sql_query))
                    print(result, "sql执行结果...")
                    # 如果是查询语句（SELECT），获取结果
                    if sql_query.strip().upper().startswith('SELECT'):
                        # 获取列名
                        columns = result.keys()
                        # 获取所有行数据
                        rows = []
                        select_cn_keys = {}

                        for row in result:
                            # 将 Row 对象转换为字典
                            row_dict = {}
                            for i, column in enumerate(columns):
                                key = col_name_dict[column] if column in col_name_dict else column  # key是列中文名

                                if key not in select_cn_keys:
                                    # SUM(BY_JE)
                                    if key == column:
                                        if "(" in key:
                                            mi = key.split("(")[1].strip(")")
                                            key = col_name_dict.get(mi, f"数值列{i}") + get_agg_name(column)
                                            if mi in unit_dict and "COUNT" not in column:
                                                unit_dict[column] = unit_dict[mi]

                                    select_cn_keys.update({key: column})
                                if column in unit_dict:
                                    row_dict[key] = f"{row[i]}{unit_dict[column]}" if row[
                                        i] else f"0.0{unit_dict[column]}"
                                else:
                                    # 现在所有的去除所有的别名情况，但是单位还是要加
                                    row_dict[key] = f"{row[i]}" if row[i] else '0.0'
                            rows.append(row_dict)
                        yield ret_format(f"执行 SQL 查询成功。数据长度为：{len(rows)}\n\n")
                        if not rows:
                            yield ret_format(f"根据您提问的问题，没有找到任何数据。请您优化问题后重新提问！\n\n")
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
                            return
                        else:
                            # 对rows进行重构，用于适应多类目之间的对比分析，去除完全重复的列目
                            # 重构数据为适合的数据格式= 上海市-个人所得税-本月数金额（万元）| 200.0万元
                            date_col = select_x(list(select_cn_keys.keys()))
                            category_cols = []
                            value_cols = []
                            del select_cn_keys[date_col]
                            # 如果因为别名问题导致列没有匹配到具体单位该如何处理
                            for w1, w2 in select_cn_keys.items():
                                if w2 in ["RG_NAME", "RG_CODE", "XM_CODE", "XM_NAME", "YEAR_MONTH",
                                          "区划名称", "区划编码", "项目编码", "项目名称", "业务年月"
                                          ]:
                                    category_cols.append(w1)
                                elif w2 in unit_dict:
                                    value_cols.append(w1)
                                else:
                                    value_cols.append(w1)
                            rows = reshape_for_chart(rows, date_col=date_col, category_cols=category_cols,
                                                     value_cols=value_cols)
                            # 根据数据格式和用户问题确认图表要素是否齐全。1.数据维度大于2，数据行数大于1，则生成图表。
                            if not app.state.redis.get(question.strip() + "<-split->cache"):
                                # 执行一个异步任务，将值缓存
                                # await request.app.state.redis.set('haha', 1)
                                print("执行后台任务...")
                                background_tasks.add_task(creat_image_data, rows, date_col, question, app)
                            print("rows:", rows)
                            try:
                                new_prompt = app.state.prompt1.format(
                                    question=question, new_question=new_question, demo=demo,
                                    rows=rows, sql_query=sql_query)
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
                                        "choices": chunk.model_dump()["choices"],
                                    }).replace("“", '"').replace("”", '"') + "\n\n"

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


@app.post("/get_image_info", summary="获取图片信息")
async def get_image_info(request: Request,
                         data: str = Form(description='返回的查询数据', default=""),
                         table_name: str = Form(description='中文表名', default=''),
                         question: str = Form(description='用户问题'),
                         ):
    cache_data = app.state.redis.get(question.strip() + "<-split->cache")
    print("cache_data:", cache_data)
    if cache_data:
        return {"status": "success", "data": cache_data}
    else:
        # 如果不存在缓存这里有两个方案，根据data和table_name 继续获取图片信息
        task_status = app.state.redis.get(question.strip() + "_task_cache")
        if task_status:
            print(f"{task_status}任务执行中...")
            # 执行中的任务怎么杀死后续优化..目前执行中就等着任务绝对不会超过2分钟
            return {"status": "error", "data": task_status}
        else:
            if data and table_name:

                try:
                    prompt = app.state.prompt2.format(data=data, table_name=table_name,
                                                      question=question, demo_data1=app.state.demo_data1,
                                                      demo_data2=app.state.demo_data2)
                    create_params2 = {
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": question},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 8192,
                        "stream": False,
                        "timeout": api_config["timeout"]
                    }
                    response2 = app.state.chat.chat.completions.create(**create_params2)
                    ret2 = response2.choices[0].message.content
                    # 解析ret2
                    if "```json" in ret2:
                        json_data = json.loads(ret2.split("```json")[1].split("```")[0])
                    else:
                        json_data = json.loads(ret2)
                    if json_data["can_plot"]:
                        categories = json_data['categories']
                        values = json_data['values']
                        title = json_data['title']
                        y_titles = json_data['y_axis_label']
                        chart_config = create_image(categories, values, title, y_titles)
                        app.state.redis.set(question.strip() + "<-split->cache",
                                            json.dumps(chart_config, ensure_ascii=False), ex=3600)
                        print("chart_config---->", chart_config)
                        return {"status": "success", "data": chart_config}
                    else:
                        return {"status": "error", "data": "数据格式不支持绘图！"}
                except Exception as e:

                    return {"status": "error", "data": str(e)}
            else:
                return {"status": "error", "data": "数据格式不支持绘图！"}


app.add_middleware(  # 解决跨域问题
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('bge_main_new_liu:app', host=f'192.168.100.160', port=8794, workers=1)
