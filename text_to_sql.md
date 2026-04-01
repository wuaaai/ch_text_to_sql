### 阶段一 系统启动
1. **load_models()**  
输入: 无（挂载在 @app.on_event("startup")）。  
输出: 无直接输出，但会在全局对象 app.state 上绑定大量资源。  
连接 Redis 和 Qdrant数据库  
初始化 OpenAI 客户端  
加载四个 Prompt 模板（prompt1 到 prompt4）。  
读取本地的词典文件（project_words.txt, col_words.txt），并实例化 KeywordMatcher。   
- 为什么要在这里预加载词典和模型？  
  因为构建 Aho-Corasick 自动机（KeywordMatcher）和读取文件是耗时操作，如果放在接口里每次请求都读，响应会极其卡顿。放在启动时加载，空间换时间。

### 阶段二：问题解析与找表找列
2. 用户的问题（如：“2025年唐山市教育支出是多少？”）进来了，系统第一步是要弄清楚去哪张表查。  
**get_embedding1(text_list)**
- 输入: 文本列表，例如 ["2025年唐山市教育支出是多少？"]。  
- 输出: 文本的向量表示（一串浮点数数组）。  
- 原理: 调外部微服务（8991端口）将文本转化为向量，用于后续在 Qdrant 中进行语义相似度搜索。  
**wait_info(search_point, check_sheng, use_table_info, source)**  
- 输入
  - search_point: Qdrant 向量数据库搜回来的原始结果（可能包含多张候选表）。
  - check_sheng: 布尔值，问题是否包含“省级/本级”。
  - use_table_info: 预加载的表元数据配置。
  - source: 数据来源标记（比如是“表描述命中”还是“列匹配命中”）。
- 输出: 过滤清洗后的一组字典，如 [{"table": "表A", "score": 0.8, "source": "表描述"}]
- 原理: 这是一个硬规则过滤器。如果用户问了“省级”，系统就必须把 Qdrant 搜出来但不属于省级（is_sheng 为 False）的表剔除，防止模型查错。  
**Text2SQLTableRanker 类的 rank(raw_data) 方法**  
- 输入: wait_info收集到的所有候选表及得分数据。
- 输出: 选出的最相关的一张表，包含最终得分和明细，如 {"status": "SUCCESS", "selected_table": "表A", ...}。
- 原理: 这是一个多路召回融合打分算法。代码里定义了三个权重（表描述占 0.6，项目匹配占 0.27，列匹配占 0.13）。它把向量搜索的分数和精确词汇匹配的次数结合起来，算出一个总分，并选出 Top 1。
- 难点解析: 这里的难点在于平滑处理（count_to_strength 函数）。如果用户话里命中了 5 次项目名，如果不做平滑，分数会直接爆表。作者用了一个非线性函数（命中 1 次得 0.65，3 次得 0.88）来压制极端分数，这是一种很老练的搜索排序降权技巧。

### 阶段三：SQL Generation & Patching ####  
确定了表，接下来要借助大模型生成 SQL。  

3. **model_chat(chat, question, prompt)**  
- 输入: OpenAI 客户端、用户问题、系统 Prompt。  
- 输出: 大模型生成的字符串。  
- 原理: 简单的 OpenAI 接口封装。在此流程中，它结合 prompt3 被用来把用户口语化的问题，改写成非常严谨的标准查询模板（比如补全默认时间、明确聚合意图）  
**get_text2sql(question, demo, evidence)**
- 输入: 标准化后的问题、表结构 Demo、参考示例 Evidence。
- 输出: JSON 格式，包含生成的 SQL 语句。
- 原理: 调外部的微服务（8991端口）进行 Text-to-SQL 转换。  
**SQL 补丁三兄弟：split_select_fields, remove_alias, ensure_year_month_in_select**  
这三个函数是代码里最硬核、最容易出 Bug 的部分，用来手动修改大模型生成的、可能有瑕疵的 SQL。

- 输入: 大模型生成的原始 SQL，例如 SELECT SUM(A) AS 合计 FROM T WHERE YEAR_MONTH='202501'。
- 输出: 强制带上时间字段且去掉别名的 SQL，例如 SELECT \YEAR_MONTH, `SUM(A) FROM T WHERE YEAR_MONTH='202501';`。
- 原理与难点: 大模型生成的 SQL 经常不带时间维度，导致后面没法画趋势图（画图必须有 X 轴）。
  - ensure_year_month_in_select 发现 WHERE 里有时间条件，但 SELECT 里没有时，会强行把 YEAR_MONTH 塞进 SELECT 里。
  - 为什么不用简单的 split(",") 来切分 SELECT 字段？因为 SQL 里会有 SUM(A, B) 这样的函数，简单的逗号切分会把函数腰斩。
  - 难点解析：split_select_fields 和 remove_alias 使用了状态机（深度计数器 depth）的原理。遇到左括号 ( 深度加 1，遇到右括号 ) 深度减 1。只有在 depth == 0 时遇到的逗号，才是真正的字段分隔符；只有在 depth == 0 时遇到的 AS，才是顶层别名。这是非常经典的简易抽象语法树（AST）解析思想。
  
### 阶段四：查库、数据清洗与画图 ###
SQL 查出数据了（存在变量 rows 里），现在要把它变成前端能画图的样子。

4. **reshape_for_chart(data, date_col, ...)**
- 输入: 数据库查出来的原始行列表（一维数组）。  
- 输出: 透视（Pivot）后的列表，用于给 Echarts 喂数据。  
- 原理: 如果查出来的表是多维度的（比如同时有上海、北京两个地区的数据交织在一起），这个函数负责把它们按时间（date_col）分组，并把分类列和指标列拼接起来（比如变成“上海市-本月数”），完成数据拍平。  

**creat_image_data(data, x_axis_label, question, app)**
- 输入: 拍平后的数据、X轴标签、原问题、app 对象。
- 输出: 无（直接把结果存入 Redis）。
- 原理: 这是一个挂在后台跑的异步任务（Background Task）。它把干瘪的数字扔给大模型（基于 prompt2），让大模型判断这批数据适合画饼图、柱状图还是折线图，并提取出 Categories 和 Values。  

**create_image(categories, values, title, y_titles)**
- 输入: X轴数据、Y轴数据、标题。
- 输出: 一个极其复杂的 Echarts JSON 配置字典。
- 原理: 纯手工利用 Python 的字典组装 Echarts 的各种配置项。
- 难点解析: 难点在于双 Y 轴处理。如果数据里同时有“万元”（绝对值）和“%”（比例），它会动态计算数据的极值（最大/最小值），并左右各开一个 Y 轴（yAxisIndex: 1），防止绝对值太大把百分比的折线压成一条平线。

### 阶段五：流式响应调度器 (The Orchestrator) ###
5. **ret_format(msg, stop)**
- 输入: 一段文字信息。
- 输出: 包装成 OpenAI 格式的 Server-Sent Events (SSE) 数据块。
- 原理: 伪装成 OpenAI 的 API 返回格式，前端接收到 data: {"choices": [{"delta": {"content": "..."}}]} 后，就能像打字机一样把后台的一举一动实时显示在屏幕上。  
- 
**/v1/chat/completions (即函数 sql3)**
- 原理: 它是总指挥。用 yield 关键字不断地调用 ret_format，把上述流程串起来：  
yield "找到关键词..."  
yield "搜到相关表..."  
yield "已生成 SQL..."  
最后，拿着查出来的数据再去问一次大模型（prompt1），流式 yield 大模型写出的最终文字分析报告。同时触发异步任务去画图。


'RDYS_LD_YSSC_YSZX_QSYBGGYSSRWC': '预算审查监督-预算执行表-全省一般公共预算收入完成情况表',  
'RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC': '预算审查监督-预算执行表-全省一般公共预算支出完成情况表',
'RDYS_LD_YSZX_YBGGYS_SBJZCWCQK': '预算执行-省本级一般公共预算支出完成情况表',

保留字段：
DATA_ID 唯一标识  
TEAR_MONTH 业务年度  
XM_CODE 项目编码
XM_NAME 项目名称
YSS 预算数
BYS_JE 本月数金额
BYS_SNTYS 本月数-上年同月数
BYS_TBE 本月数-同比加减额
BYS_TBB 本月数-同比加减百分比
ZBYLJS_JE 至本月累计数-金额
ZBYLJS_WYSS 至本月累计数-为预算额百分之多少
ZBYLJS_SNTQS 至本月累计数-上年同期数
ZBYLJS_TBE 至本月累计数-同比加减额
ZBYLJS_TBB 至本月累计数-同比加减百分比
BNSY_LJS 本年上月累计数
SNSY_LJS 上年上月累计数


2025年1月至5月交通运输的本月执行数是多少
2025年1月至5月技术支出的本月执行数是多少
2025年1月至5月车辆通行费收入的本月执行数是多少

2025年1月至5月交通运输支出的本月执行数是多少
