import json
import time
import datetime
import re
from fastapi import APIRouter, Request, BackgroundTasks, Depends
from starlette.responses import StreamingResponse, JSONResponse
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from dependencies import get_res, AppResources
from core.ranker import Text2SQLTableRanker
from core.utils import (
    get_embedding1, wait_info, get_text2sql, model_chat, 
    ensure_year_month_in_select, split_select_fields, remove_alias, 
    ret_format, select_x, reshape_for_chart, check_select_col, get_agg_name
)
from core.visual import creat_image_data
from prompts import PROMPT3, PROMPT1

router = APIRouter()

@router.post("/chat/completions")
async def sql3(request: Request, background_tasks: BackgroundTasks, res: AppResources = Depends(get_res)):
    # 【原注释保留】此接口不用指定表名-会在目前的四张表中自己根据问题选一张
    # 【原注释保留】demo只会选中一张表，因为后续要根据这张表去进行sql优化
    # 【原注释保留】多个词命中情况--多表 预算数和一般公共预算收入
    # 【原注释保留】多个词命中情况--一表多指标
    
    # 1. 解析原始请求体 (保持原样，兼容字节流等格式)
    body = await request.body()
    body_str = body.decode('utf-8')

    try:
        request_json = json.loads(body_str)
    except json.JSONDecodeError:
        # 【解读前同事逻辑】这里是为了兼容某些前端或者客户端传过来的奇怪数据格式，
        # 有时候传过来的不是纯 json，而是带了 b'' 标识的字符串，所以他强行截断处理了一下。
        if body_str.startswith('b\'') and body_str.endswith('\''):
            body_str = body_str[2:-1]
            request_json = json.loads(body_str)
        else:
            return {"error": {"type": "invalid_request_error", "message": "请求异常，收到的参数不是json"}}
            
    # 2. 提取参数 (Dify 会自动带上 tools 参数如果配置了工具)
    messages = request_json.get("messages", [])
    question = messages[-1].get("content", "") if messages else ""
    stream = request_json.get("stream", False)
    
    # 使用 Aho-Corasick 自动机从问题中提取所有的“相关项目/科目”实体词
    question_words = res.words_match.find_all(question)
    print(f"========== [DEBUG] 步骤 0: 实体抽取 ==========")
    print(f"[DEBUG] 命中的项目/科目词: {question_words}")

    # 【原注释保留】TODO支持结合记忆情况对问题和数据进行适当优化。
    # 【原注释保留】TODO:问题可能会被拆分成两个表的查询语句，目前先不考虑连表和多表查询情况

    # 定义一个异步生成器，用于流式返回数据（SSE 格式）给前端，实现打字机效果
    async def generate_progress():
        # wait_tables 是用来存放候选表的池子，后续送给 Text2SQLTableRanker 打分
        wait_tables = []
        
        # 【解读前同事逻辑】：
        # data1 用来存 "科目/项目" 的映射关系 (用户说的词 -> 数据库的 code 或 标准名)
        # data2 用来存 "列名(检索项)" 的映射关系
        data1 = {}
        data2 = {}
        
        # 【原注释保留】如果问题中没有出现明确的省本级意图，(省级、本级)
        # 【原注释保留】清理结果中的省本级的表。如果出现省本级字眼，其他跟省本级无关的表剔除
        check_sheng = 1 if any(char in question for char in ['省级', "本级"]) else 0
        yield ret_format(f"解析到问题可能和{'省本级数据' if check_sheng else '全省数据'}相关\n\n")
        
        project_words = []
        
        # ==============================================================================
        # 第 1 步：根据实体词在 Qdrant 向量库中检索【科目/项目】
        # ==============================================================================
        if question_words:
            ques_words = list(set(question_words))
            query_filter = Filter(
                must=[
                    FieldCondition(key="cate", match=MatchValue(value="科目/项目")),
                    FieldCondition(key="words", match=MatchAny(any=ques_words))
                ]
            )
            existing_points = res.client.query_points(
                collection_name=res.collection_name,
                query_filter=query_filter,
                limit=len(ques_words),
                with_payload=True,
                with_vectors=False,
                score_threshold=None # 【原注释保留】不过滤分数
            )
            print(f"========== [DEBUG] 步骤 1: 项目/科目 Qdrant 检索 ==========")
            # 【原注释保留】构建data1映射字典
            for ww in existing_points.points:
                print(f"[DEBUG] 命中词: {ww.payload.get('words')}, 所属表: {ww.payload.get('zh_table')}, 得分/概率: {ww.score}")
                for n, u in enumerate(ww.payload["zh_table"]):
                    if ww.payload["words"] not in project_words:
                        project_words.append(ww.payload["words"])
                    
                    # 按照表信息配置，决定取 `select_code` 还是 `table_select`
                    if res.use_table_info[u]['project_key']:
                        if u not in data1:
                            data1[u] = [{ww.payload["words"]: ww.payload["select_code"][n]}]
                        else:
                            data1[u].append({ww.payload["words"]: ww.payload["select_code"][n]})
                    else:
                        if u not in data1:
                            data1[u] = [{ww.payload["words"]: ww.payload["table_select"][n]}]
                        else:
                            data1[u].append({ww.payload["words"]: ww.payload["table_select"][n]})

            # 将命中的表放入候选池
            wait_tables.extend(wait_info(existing_points, check_sheng, res.use_table_info, source="项目匹配"))
        yield ret_format(f"已找到{len(project_words)}个关键词，分别为{','.join(project_words)}\n\n")
        
        # ==============================================================================
        # 第 2 步：将整个问题向量化，检索最相似的【表描述】
        # ==============================================================================
        try:
            query_vector = get_embedding1([question])['embeddings'][0]
        except:
            yield ret_format("模型服务异常，请联系管理员维护！",stop="stop")
            return
            
        search_filter = Filter(must=[FieldCondition(key="cate", match=MatchValue(value="表描述"))])
        results = res.client.query_points(
            collection_name=res.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=3,
            with_payload=True,
            with_vectors=False,
            score_threshold=None
        )
        print(f"========== [DEBUG] 步骤 2: 表描述 Qdrant 检索 ==========")
        for pt in results.points:
             print(f"[DEBUG] 匹配到表描述: {pt.payload.get('zh_table')}, 得分/概率: {pt.score}")
        wait_tables.extend(wait_info(results, check_sheng, res.use_table_info, source="表描述"))
        yield ret_format(f"表相似度检索完毕。\n\n")

        # ==============================================================================
        # 第 3 步：提取相关的【列/检索项】和【地区】实体，并去向量库查表
        # ==============================================================================
        # 【原注释保留】相关列匹配,现在用的强匹配
        col_words = res.col_match.find_all(question)
        zone_words = res.zone_match.find_all(question)
        print(f"[DEBUG] 命中的列名词: {col_words}")
        print(f"[DEBUG] 命中的地区词: {zone_words}")
        col_words2 = []
        if col_words:
            col_words = list(set(col_words))
            query_filter = Filter(
                must=[
                    FieldCondition(key="cate", match=MatchValue(value="检索项")),
                    FieldCondition(key="words", match=MatchAny(any=col_words))
                ]
            )
            col_points = res.client.query_points(
                collection_name=res.collection_name,
                query_filter=query_filter,
                limit=len(col_words),
                with_payload=True,
                with_vectors=False,
                score_threshold=None
            )
            print(f"========== [DEBUG] 步骤 3: 检索项/列名 Qdrant 检索 ==========")
            for yy in col_points.points:
                 print(f"[DEBUG] 命中列名: {yy.payload.get('words')}, 对应表: {yy.payload.get('zh_table')}, 得分/概率: {yy.score}")
            # 构建 data2 映射字典 (用户口语的列名 -> 数据库真实的列名)
            for yy in col_points.points:
                for n, u in enumerate(yy.payload["zh_table"]):
                    if yy.payload["words"] not in col_words2:
                        col_words2.append(yy.payload["words"])
                    if u not in data2:
                        data2[u] = [{yy.payload["words"]: yy.payload["table_select"][n]}]
                    else:
                        data2[u].append({yy.payload["words"]: yy.payload["table_select"][n]})
            wait_tables.extend(wait_info(col_points, check_sheng, res.use_table_info, source="列匹配"))
        yield ret_format(f"已找到{len(col_words2)}个列关键词，分别为{','.join(col_words2)}\n\n")
        
        # ==============================================================================
        # 第 4 步：汇总候选表，计算权重得出 Top1 最优表
        # ==============================================================================
        print(f"========== [DEBUG] 步骤 4: 选表权重打分 ==========")
        print(f"[DEBUG] 最终进入 Ranker 的候选表池 (wait_tables): {wait_tables}")
        ranker = Text2SQLTableRanker()
        result = ranker.rank(wait_tables)
        print(f"[DEBUG] Ranker 打分结果详情:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result and result.get("status") == "SUCCESS":
            yield ret_format(f"经过权重算法分析选中最相关表为：{result['selected_table']}。\n\n")
            table_name = result['selected_table']
            demo = res.public_data[table_name] # 拿到该表的 schema 信息

            # 获取该表字段的单位映射 (例如：金额 -> 万元)
            unit_dict = {value: key for key, values in res.use_table_info[table_name]['unit'].items() for value in values}
            
            # 【原注释保留】基于大模型回复不稳定的问题提出优化方案
            # 解析画图倾向，传给模型供后续组装
            if "柱状图" in question:
                chart_hint = "bar"
            elif "饼图" in question:
                chart_hint = "pie"
            elif "折线图" in question:
                chart_hint = "line"
            else:
                chart_hint = "auto"
                
            # 【原注释保留】将用户问题中的命中列关键词替换为原始列名
            cols_names = data2.get(table_name, [])
            old_col_names = []
            for x in cols_names:
                for k, v in x.items():
                    if v:
                        old_col_names.append(f"{k}={v}")
            relevant_columns_candidates = "、".join(old_col_names) 

            # 获取查询的地区条件
            if zone_words:
                region_value = "、".join(set([res.zone_dict.get(i, "130000000") for i in zone_words]))
            else:
                region_value = "" 

            # 【原注释保留】TODO:近一年丰南区，本级国有资本经营预算中的国有企业退休人员社会化管理补助支出的累计同比额情况
            # 【原注释保留】当相关项目词有包含情况出现时该如何选择合适的项目。
            if question_words:
                matched_project_code = "、".join([value for item in data1.get(table_name, []) for value in item.values() if value is not None]) 
            else:
                # 【原注释保留】当问题相关度很高但是并没有命中项目时根据此表返回一个最相关的项目
                query_filter22 = Filter(
                    must=[
                        FieldCondition(key="cate", match=MatchValue(value="科目/项目")),
                        FieldCondition(key="zh_table", match=MatchValue(value=table_name))
                    ])
                results22 = res.client.query_points(
                    collection_name=res.collection_name,
                    query=query_vector,
                    query_filter=query_filter22,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=None
                )
                for zz in results22.points:
                    if table_name in zz.payload["zh_table"]:
                        index = zz.payload["zh_table"].index(table_name)
                        matched_project_code = zz.payload["select_code"][index]
                        break
                else:
                    matched_project_code = "201" # 兜底编码
                    
            # ==============================================================================
            # 第 5 步：重写/优化用户问题，标准化为模板句
            # ==============================================================================
            # 【原注释保留】如果选中了表，那么根据选择过程中的数据去优化问题
            prompt4 = PROMPT3.format(
                user_question=question, table_name=table_name,
                table_schema=demo, matched_project_name=matched_project_code,
                relevant_columns_candidates=relevant_columns_candidates,
                region_value=region_value, chart_hint=chart_hint,
                now_date=datetime.datetime.now().strftime("%Y-%m-%d")
            )

            new_question = model_chat(res.chat, question, prompt4)
            print(f"========== [DEBUG] 步骤 5: 问题改写 ==========")
            print(f"[DEBUG] 原始用户问题: {question}")
            print(f"[DEBUG] 传给大模型的 Prompt 模板:\n{prompt4}")
            print(f"[DEBUG] 大模型改写并优化后的问题: {new_question}")
            yield ret_format(f"问题已优化为：{new_question}\n\n")

            # ==============================================================================
            # 第 6 步：外部 API 调用生成 Text2SQL
            # ==============================================================================
            en_table_name = res.use_table_info[table_name]['table']
            col_name_dict = {k: v['comment'] for k, v in demo['tables'][en_table_name]['fields'].items()}
            
            # 提供 few-shot 案例给 Text2SQL 引擎
            evidence = """
            【析言 Text2SQL 生成规则（高优先级）】
            你是财政预算执行场景 SQL 生成器。请严格遵守以下规则：

            一、SQL 结构硬约束
            1. 只允许输出一条 MySQL SELECT 语句；禁止 DML/DDL（INSERT/UPDATE/DELETE/ALTER/CREATE 等）。
            2. 严禁列别名：禁止 `AS xxx`，也禁止隐式别名（如 `SUM(a) total`）。
            3. 严禁 `SELECT *`，必须显式列出字段。
            4. 表名与字段名统一使用反引号包裹。
            5. 输出内容只能是 SQL 本身，不要解释、不要 markdown、不要注释。

            二、时间口径规则（YYYYMM）
            1. 时间字段优先使用 `business_year_and_month`。
            2. 单月：`= 'YYYYMM'`。
            3. 区间：`BETWEEN 'YYYYMM' AND 'YYYYMM'`。
            4. 当用户问“分别/趋势/每月/各月”，SELECT 必须包含时间字段，并按时间升序排序。

            三、排序与条数规则
            1. 用户问“最多/最大/最高”时：按目标数值列 DESC 排序并 `LIMIT 1`（或用户指定 N）。
            2. 用户问“最少/最小/最低”时：按目标数值列 ASC 排序并 `LIMIT 1`（或用户指定 N）。
            3. 非极值问句默认不加 LIMIT（除非用户明确要求）。

            四、筛选规则
            1. 项目筛选优先使用编码列（如 `project_code`），其次名称列。
            2. 省本级问题优先选择省本级表；全省问题优先选择全省表。
            3. 用户未给区域时，不强行追加区域过滤条件。

            五、口径匹配规则（重点）
            1. 本月口径词（本月/当月/当前月份/本月执行数/本月金额）
            -> 优先映射：`当前月份发生的实际金额`
            2. 累计口径词（累计/年初至今/截至本月/本年累计）
            -> 优先映射：`从本年1月截至本月的累计发生总额`
            3. 同比百分比口径
            -> 优先映射：`本年累计金额相较于上年同期累计金额的增减百分比` 或
                            `本月金额相较于上年同月的增长或减少百分比`
            4. 同比差额口径
            -> 优先映射：`本年累计金额减去上年同期累计金额的增减额` 或
                            `本月金额减去上年本月金额的差额（正数为增加，负数为减少）`
            5. 关键冲突约束：
            - 问题是“本月”口径时，禁止误用“累计”列；
            - 问题是“累计”口径时，禁止误用“本月”列。

            六、字段语义映射（同义词 -> 标准字段）
            1) 本年累计金额相较于上年同期累计金额的增减百分比
            同义词：同比增长率、累计同比增幅、今年累计比去年同期的增长百分比、累计金额同比变化百分比、累计同比
            2) 去年1月至去年同月的累计发生总额
            同义词：去年同期累计、上年同期累计金额、去年截止同月累计、上年同期累计
            3) 截至去年上一个月份的累计金额
            同义词：去年截止上个月的累计、上年同期上月累计、去年截至上月的累计金额
            4) 本月金额相较于上年同月的增长或减少百分比
            同义词：本月同比百分比、单月同比增长率、本月同比增幅、本月同比变动比例
            5) 项目全年的总预算额度
            同义词：年度预算、全年预算、预算总额、总预算
            6) 本年累计金额占全年预算金额的百分比
            同义词：预算执行率、累计预算完成率、预算进度百分比、预算完成度
            7) 从本年1月截至本月的累计发生总额
            同义词：本年累计、年初至今累计、截止本月累计、年初至本月累计
            8) 本年累计金额减去上年同期累计金额的增减额
            同义词：累计同比增减额、累计同比差额、累计金额同比差额
            9) 当前月份发生的实际金额
            同义词：本月发生额、当月金额、本月实际数、本月发生数、本月金额、金额
            10) 本月金额减去上年本月金额的差额（正数为增加，负数为减少）
            同义词：本月同比增减额、单月同比差额、本月同比变动额、本月增减额
            11) 去年同月份发生的实际金额
            同义词：去年同期单月、去年同月金额、上年本月金额、去年同月发生额
            12) 截至本年上一个月份的累计金额
            同义词：本年截止上月的累计、年初至上月累计、截止上月累计金额、本年上月末累计
            
            七、示例（必须学习风格）
            示例1（本月口径，时间区间）
            question: 2025年1月至5月省本级科技支出的本月执行数是多少？
            answer: SELECT `business_year_and_month`, `current_month_actual_amount` FROM `RDYS_LD_YSZX_YBGGYS_SBJZCWCQK` WHERE `project_code` = '206' AND `business_year_and_month` BETWEEN '202501' AND '202505' ORDER BY `business_year_and_month`;

            示例2（累计口径，时间区间）
            question: 2025年1月至5月省本级科技支出的累计执行数是多少？
            answer: SELECT `business_year_and_month`, `year_to_date_accumulated_amount` FROM `RDYS_LD_YSZX_YBGGYS_SBJZCWCQK` WHERE `project_code` = '206' AND `business_year_and_month` BETWEEN '202501' AND '202505' ORDER BY `business_year_and_month`;

            示例3（极值 Top1）
            question: 2025年度公共安全支出中金额最多的是哪项？
            answer: SELECT `project_name`, `current_month_actual_amount` FROM `RDYS_LD_YSZX_YBGGYS_SBJZCWCQK` WHERE `project_code` = '204' AND `business_year_and_month` BETWEEN '202501' AND '202512' ORDER BY `current_month_actual_amount` DESC LIMIT 1;

            示例4（编码筛选 + 按月返回）
            question: 2025年2月到2025年10月期间科目编码为205的本月金额分别是多少？
            answer: SELECT `business_year_and_month`, `current_month_actual_amount` FROM `RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC` WHERE `business_year_and_month` BETWEEN '202502' AND '202510' AND `project_code` = '205';

            示例5 （去年 相对时间计算）
            question: 去年第一季度每个月社会保障和就业支出是多少
            answer: SELECT `business_year_and_month`, `current_month_actual_amount` FROM `RDYS_LD_YSZX_YBGGYS_SBJZCWCQK` WHERE `project_code` = '208' AND `business_year_and_month` BETWEEN '202501' AND '202503' ORDER BY `business_year_and_month`;
            示例6 （项目编码）
            question: 2025年全年卫生健康支出比教育支出多多少？
            answer: SELECT
            SUM(CASE WHEN `project_code` IN ('210','205') THEN `current_month_actual_amount` ELSE 0 END)
            -
            SUM(CASE WHEN `project_code` = '205' THEN `current_month_actual_amount` ELSE 0 END)
            FROM `RDYS_LD_YSZX_YBGGYS_SBJZCWCQK`
            WHERE `business_year_and_month` BETWEEN '202501' AND '202512';


            八、列命中自检（生成前必须执行）
            在生成 SQL 前，先做以下自检：
            1) 先识别问题口径标签：
            - 月度口径标签：本月/当月/当前月份/单月
            - 季度口径标签：第一季度/第二季度/第三季度/第四季度
            - 累计口径标签：累计/截至/年初至今/本年累计
            - 比例标签：同比/百分比/占比/增长率
            - 差额标签：增减额/差额
            2) 再检查拟选列是否与标签一致：
            - 月度口径 -> 仅可选“当前月份发生的实际金额”或其同义列
            - 累计口径 -> 仅可选“从本年1月截至本月的累计发生总额”或其同义列
            - 比例标签 -> 仅可选百分比类列
            - 差额标签 -> 仅可选差额类列
            3) 若冲突，必须重选列并重写 SQL，不得输出冲突 SQL。
            九、冲突示例（反例约束，必须避免）
            - 问题：2025年1月至5月省本级科技支出的本月执行数是多少？
            错误列：`year_to_date_accumulated_amount`（累计列）
            正确列：`current_month_actual_amount`（本月列）
            - 问题：2025年1月至5月省本级科技支出的累计执行数是多少？
            错误列：`current_month_actual_amount`（本月列）
            正确列：`year_to_date_accumulated_amount`（累计列）
            - 问题：去年第一季度每个月社会保障和就业支出是多少？
            错误时间：`202401-202403`
            正确时间：`202501-202503`
            
            十、季度的计算规则
            1. 季度划分：第一季度为1-3月，第二季度为4-6月，第三季度为7-9月，第四季度为10-12月，季末为10-12月
            2. 季度累计金额等于该季度内各月份发生额的加和汇总
            3. 季度同比计算公式：（本季度累计 - 上年同季度累计）/ 上年同季度累计
            4. 季度环比计算公式：（本季度累计 - 上季度累计）/ 上季度累计
            5. 用户提及本季度、当季、Q1-Q4、第一季度至第四季度、上个季度、去年同季度等表述时，均按季度规则处理
            6. 季度字段映射：季度累计发生总额对应的用户表述包括季度发生额、季度总额、本季度金额、当季累计、季度合计

            十一、相对时间计算规则
            1. 去年数据计算规则
            去年定义：当前年份减1所对应的完整自然年度 
            去年累计金额：指去年1月1日至去年12月31日期间的发生总额
            去年同月：指去年与当前月份相同的那一个月，例如当前为2026年3月，去年同月即为2025年3月
            去年同季度：指去年与当前季度相同的那个季度，例如当前为2026年第二季度（4-6月），去年同季度即为2025年第二季度（4-6月）
            去年累计同比：指本年累计金额与去年同周期累计金额的对比

            2. 前年数据计算规则
            前年定义：当前年份减2所对应的完整自然年度
            前年累计金额：指前年1月1日至前年12月31日期间的发生总额
            前年同月：指前年与当前月份相同的那一个月
            前年同季度：指前年与当前季度相同的那个季度

            3. 去年、前年相关用户表述映射
            用户提及以下表述时，均应识别为去年查询意图：
            “去年”、“上年”、“上一年”、“去年全年”、“上年全年”、“去年同期”、“去年同月”、“去年同季度”、“去年同比”
            用户提及以下表述时，均应识别为前年查询意图：
            “前年”、“前一年”、“前年全年”、“前年同期”、“前年同月”、“前年同季度”

            4. 跨年对比规则
            用户输入“比去年多了多少”时，计算逻辑为（本年累计金额 - 去年累计金额）
            用户输入“比去年增长了百分之多少”时，计算逻辑为（本年累计金额 - 去年累计金额）/ 去年累计金额
            用户输入“和前年相比”时，取前年同周期数据进行对比
            用户输入“近三年”时，取前年、去年、本年三个完整年度的数据

            5. 边界处理规则
            若当前年份为1月，查询“去年同月”时，去年同月为去年1月，数据存在，正常返回
            若当前年份为1月，查询“上季度”跨年时，上季度为去年第四季度，需跨年取数
            若查询前年数据时当前年份为2026年，前年即为2024年，按完整年度返回数据

            6.上半年计算规则
            上半年指一个自然年度的前六个月，即1月1日至6月30日，上半年累计金额等于1月至6月各月份发生额的加总
            用户提及上半年、上半年度、前半年、年初到6月底、1月到6月、一到六月、上半年合计、上半年累计、上半年总额等表述时，按上半年规则处理
            今年上半年指当前年份的1月至6月，去年上半年指上一个自然年度的1月至6月，前年上半年指前一个自然年度的1月至6月
            上半年同比计算公式为（今年上半年累计 - 去年上半年累计）/ 去年上半年累计
            上半年占全年比例计算公式为（上半年累计 / 全年累计）× 100%
            上半年预算完成率计算公式为（上半年累计 / 全年预算）× 100%
            若当前日期早于6月30日，查询今年上半年时返回1月1日至当前日期的累计，并提示上半年尚未结束
            上半年包含第一季度和第二季度，与下半年互补构成全年

            7.相对时间解析硬规则（最高优先级）
            1) 所有相对时间词必须以{current_year} 为唯一锚点解析，禁止使用训练记忆中的固定年份。
            2) 若 {current_year}=2026-xx-xx：
            - 今年 = {current_year}
            - 去年 = {current_year}-1
            - 前年 = {current_year}-2
            3) 季度换算（按自然季度）：
            - 第一季度 = 01-03月
            - 第二季度 = 04-06月
            - 第三季度 = 07-09月
            - 第四季度 = 10-12月
            4) “去年第一季度每个月”必须解析为：
            - 时间范围：2025年01月-2025年03月
            - SQL 条件：BETWEEN '202501' AND '202503'
            - 且需按月份升序返回。
            5) 若问题出现“每个月/各月/逐月/趋势”，SELECT 必须包含年月字段，并 ORDER BY 年月字段升序。
            6) 若相对时间解析结果与 now_date 推导不一致，必须重写 SQL。
            9、项目筛选规则（高优先级）
            1) 涉及“项目/科目/支出类别”（如教育、卫生健康、节能环保、科技等）时：
            - 必须优先使用 `project_code`做筛选、分组、比较；
            - `project_name` 仅用于识别用户意图，不作为主过滤条件。
            2) 只有在没有可用编码映射时，才允许退化为 `project_name` 查询。
            3) 若同一问题同时出现名称和编码，必须以编码为准。
            4) 两项对比（A 比 B 多多少）时：
            - 应在同一 SQL 中基于编码完成对比计算；
            - 禁止分别用 `project_name='A'` 和 `project_name='B'` 作为主条件。
            5) 输出前自检：
            - 若问题包含“教育/卫生/环保/科技”等项目词，而 SQL 未使用 `project_code`，则判为不合格并重写。
            6)表名中英文映射：
            - 预算审查监督-预算执行表-全省一般公共预算收入完成情况表:RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC
            - 预算审查监督-预算执行表-全省一般公共预算支出完成情况表:RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC
            - 预算执行-省本级一般公共预算支出完成情况表:RDYS_LD_YSZX_YBGGYS_SBJZCWCQK
            """
               
            sql_query_json = get_text2sql(new_question, demo, evidence)
            print(f"========== [DEBUG] 步骤 6: 外部大模型生成 Text2SQL ==========")
            print(f"[DEBUG] 模型接口返回的完整 JSON: {sql_query_json}")
            if sql_query_json['status']=='success':
                sql_query = sql_query_json['sql_query']
            else:
                yield ret_format(f"模型服务异常，请联系管理员！",stop="stop")
                return

            # ==============================================================================
            # 第 7 步：SQL 后处理（暴力字符串替换填坑）
            # ==============================================================================
            # 【原注释保留】优化sql生成过程不带符号的问题
            # 【解读前同事逻辑】：这里是因为 LLM 生成的 SQL 列名有时不带反引号(`)，
            # 遇到关键字会导致 SQL 报错，所以他暴力用字符串 replace 把所有的列名强行加上反引号。
            try:
                table_data_col = list(demo['tables'].values())[0]['fields'].keys()
            except:
                table_data_col = None
            if table_data_col:
                for u in table_data_col:
                    # if u + ' ' in sql_query and f"`{u}` " not in sql_query:
                    #     sql_query = sql_query.replace(u + ' ', f"`{u}` ")
                    # if u + ',' in sql_query and f"`{u}`," not in sql_query:
                    #     sql_query = sql_query.replace(u + ',', f"`{u}`,")
                    # if u + ';' in sql_query and f"`{u}`;" not in sql_query:
                    #     sql_query = sql_query.replace(u + ';', f"`{u}`;")
                    pattern = rf"(?<!`)\b{u}\b(?!`)"
                    sql_query = re.sub(pattern, f"`{u}`", sql_query)

            # 【原注释保留】判断筛选字段是否都是y轴字段,如果都是y轴数值字段，不具备画图要素默认填充一个时间字段
            match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE)
            if match:
                fields_str = match.group(1)
                fields = split_select_fields(fields_str)
                fields_no_alias = [remove_alias(f) for f in fields]
                if check_select_col(col_name_dict, fields_no_alias):
                    if 'business_year_and_month' in sql_query and "business_year_and_month" not in fields_str:
                        sql_query = ensure_year_month_in_select(sql_query, "business_year_and_month")

            yield ret_format(f"已生成 SQL 语句为：{sql_query}\n\n")
            
            # ==============================================================================
            # 第 8 步：执行 SQL 并处理结果集
            # ==============================================================================
            print(f"========== [DEBUG] 步骤 7 & 8: 最终入库执行的 SQL ==========")
            print(f"[DEBUG] 经过后处理修正的最终 SQL: \n{sql_query}")
            try:
                with (res.db_engine.connect() as connection):
                    result_proxy = connection.execute(text(sql_query))

                    if sql_query.strip().upper().startswith('SELECT'):
                        columns = result_proxy.keys()
                        rows = []
                        select_cn_keys = {} # 存储 英文列名->中文列名 的映射字典，用于构造最终显示

                        for row in result_proxy:
                            row_dict = {}
                            for i, column in enumerate(columns):
                                # 尝试把数据库列名转成中文解释
                                key = col_name_dict[column] if column in col_name_dict else column

                                # 处理聚合函数（比如把 SUM(amt) 变成 "金额求和"）
                                if key not in select_cn_keys:
                                    if key == column:
                                        if "(" in key:
                                            mi = key.split("(")[1].strip(")")
                                            key = col_name_dict.get(mi, f"数值列{i}") + get_agg_name(column)
                                            if mi in unit_dict and "COUNT" not in column:
                                                unit_dict[column] = unit_dict[mi]

                                    select_cn_keys.update({key: column})
                                
                                # ==========================================
                                # 👇 拼接单位 (万元、% 等) 及 调试日志 👇
                                # ==========================================
                                original_value = row[i] # 记录原始数据库取出的数值
                                
                                if column in unit_dict:
                                    final_value = f"{row[i]}{unit_dict[column]}" if row[i] else f"0.0{unit_dict[column]}"
                                    row_dict[key] = final_value
                                    # 打印带单位的拼接过程
                                    print(f"[DEBUG] 字段: {column} | 原始值: {original_value} | 匹配单位: {unit_dict[column]} | 最终拼接: {final_value}")
                                else:
                                    final_value = f"{row[i]}" if row[i] else '0.0'
                                    row_dict[key] = final_value
                                    # 打印没有单位的普通字段
                                    print(f"[DEBUG] 字段: {column} | 原始值: {original_value} | 无单位配置 | 最终值: {final_value}")
                            
                            rows.append(row_dict)
                            
                        yield ret_format(f"执行 SQL 查询成功。数据长度为：{len(rows)}\n\n")
                        
                        if not rows:
                            yield ret_format(f"根据您提问的问题，没有找到任何数据。请您优化问题后重新提问！\n\n")
                            yield json.dumps({
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "deepseek-chat",
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                            }) + "\n\n"
                            return
                        else:
                            # ==============================================================================
                            # 第 9 步：数据结构重构与图表生成
                            # ==============================================================================
                            # 【原注释保留】对rows进行重构，用于适应多类目之间的对比分析，去除完全重复的列目
                            # 【原注释保留】重构数据为适合的数据格式= 上海市-个人所得税-本月数金额（万元）| 200.0万元
                            date_col = select_x(list(select_cn_keys.keys()))
                            category_cols = []
                            value_cols = []
                            del select_cn_keys[date_col]
                            
                            # 【解读前同事逻辑】：区分维度列（地区、项目名称等）和 指标列（数值列）
                            # 方便送给 Echarts 进行 X 轴 Y 轴和 Legend 的绑定
                            for w1, w2 in select_cn_keys.items():
                                if w2 in ["RG_NAME", "RG_CODE", "XM_CODE", "XM_NAME", "YEAR_MONTH", "区划名称", "区划编码", "项目编码", "项目名称", "业务年月"]:
                                    category_cols.append(w1)
                                elif w2 in unit_dict:
                                    value_cols.append(w1)
                                else:
                                    value_cols.append(w1)
                                    
                            rows = reshape_for_chart(rows, date_col=date_col, category_cols=category_cols, value_cols=value_cols)
                            
                            # 【解读前同事逻辑】：这里将绘图任务扔进了后台（BackgroundTasks）。
                            # 后台任务会调用 LLM 判断怎么画图，并生成 echarts 的 json 配置存到 Redis 里。
                            # 另外一个接口 `/get_image_info` 会过来 Redis 取出这套 json 给前端渲染。
                            if not res.redis.get(question.strip() + "<-split->cache"):
                                background_tasks.add_task(creat_image_data, rows, date_col, question, res)

                            # ==============================================================================
                            # 第 10 步：生成最终的自然语言数据分析报告
                            # ==============================================================================
                            try:
                                new_prompt = PROMPT1.format(question=question, new_question=new_question, demo=demo, rows=rows, sql_query=sql_query)
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
                                # 接收大模型流式响应，组装成 SSE chunk 并直接 yield 给前端
                                e1 = res.chat.chat.completions.create(**create_params)
                                for chunk in e1:
                                    yield json.dumps({
                                        "id": f"chatcmpl-{int(time.time())}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": "deepseek-chat",
                                        "choices": chunk.model_dump()["choices"],
                                    }).replace("“", '"').replace("”", '"') + "\n\n"

                                # 发送停止信号
                                yield json.dumps({
                                    "id": f"chatcmpl-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": "deepseek-chat",
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                                }) + "\n\n"

                            except SQLAlchemyError as e:
                                error_msg = str(e.__cause__) if e.__cause__ else str(e)
                                yield ret_format(f"执行SQL查询失败，错误信息为：{error_msg}\n\n")
                                yield json.dumps({
                                    "id": f"chatcmpl-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": "deepseek-chat",
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                                }) + "\n\n"

            except Exception as e:
                yield ret_format(f"执行SQL捕获其他异常：{str(e)}\n\n")
                yield json.dumps({
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "deepseek-chat",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }) + "\n\n"
        else:
            # ==============================================================================
            # 表匹配失败的兜底分支
            # ==============================================================================
            yield ret_format(f"选表失败，启动兜底策略\n\n")
            yield json.dumps({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }) + "\n\n"

    # ==============================================================================
    # 接口路由响应层
    # ==============================================================================
    if stream:
        return StreamingResponse(
            generate_progress(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
    else:
        return JSONResponse(
            status_code=200,
            content={"error": {"message": "响应成功！但是本模型只支持流式返回，请使用 stream=True 参数", "type": "system_error"}}
        )