# core/utils.py
import re
import json
import time
import requests
import datetime
from collections import defaultdict
from config import BASE_URL, API_CONFIG

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
        "timeout": API_CONFIG["timeout"]
    }
    response = chat.chat.completions.create(**create_params)
    ret = response.choices[0].message.content
    return ret

def split_select_fields(select_part: str):
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
            return ''.join(buf).strip()
        else:
            buf.append(ch)
            i += 1

    return ''.join(buf).strip()

import re

def ensure_year_month_in_select(sql: str, sss: str) -> str:
    """
    【原注释保留】规则：
    1. 如果 WHERE 子句中包含 YEAR_MONTH，则 SELECT 中必须包含 YEAR_MONTH
    2. 删除 SELECT 字段中的 AS 别名
    3. 保持原 SQL 其他部分不变
    4. 不做大小写转换，适配全大写 SQL（注：现已优化为兼容大小写，防止改小写后报错）
    """
    # 1. 清理末尾的分号和空白符
    sql = sql.strip().rstrip(';')
    
    # 2. 正则表达式解析 SQL
    # 【解读】：利用正则硬拆 SQL，提取出 SELECT、FROM、WHERE 和 ORDER BY 四个部分。
    # 潜在坑点：这个正则没考虑 GROUP BY 和 HAVING，如果大模型生成了带 GROUP BY 的语句可能会解析不准。
    # 但根据你们的业务（底层是查宽表明细），可能很少出现 GROUP BY，所以暂时够用。
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

    # 3. 拆分字段并删除别名
    # 【解读】：大模型极度喜欢自己编别名，比如 SELECT amt AS '当月金额'。
    # 如果留着别名，后面 Echarts 画图时和你们的 table_info.json 字典就对不上了。
    # 所以前同事用 split_select_fields 切分逗号，再用 remove_alias 把 AS 及后面的内容全部砍掉，只留纯字段。
    fields = split_select_fields(select_part)
    fields_no_alias = [remove_alias(f) for f in fields]

    # 4. 时间字段（X轴）强制补全逻辑
    # 【解读】：sss 通常传进来的是 "YEAR_MONTH"。
    # 解决的核心痛点是：用户问“2025年各月收入是多少”，大模型可能写出 SELECT 收入 FROM 表 WHERE 年月='2025'。
    # 注意到了吗？SELECT 里没有“年月”！这就导致前端画折线图时，根本没有数据做 X 轴。
    if sss:
        # 判断大模型是否在 WHERE 条件里用到了这个时间字段（忽略大小写比较）
        has_year_month_in_where = sss.upper() in where_part.upper()
        # 遍历已有的 SELECT 字段，看看有没有查时间（去反引号、转大写比较，完美兼容小写字段）
        has_year_month_in_select = any(f.strip('`').upper() == sss.upper() for f in fields_no_alias)
    else:
        has_year_month_in_where = False
        has_year_month_in_select = False

    # 【补救】：如果在 WHERE 里按时间过滤了，但 SELECT 里忘了写，我们强行把它插在第一列！
    if has_year_month_in_where and not has_year_month_in_select:
        fields_no_alias.insert(0, f'`{sss}`')

    # 5. 去重与排他控制
    dedup_fields = []
    seen = set()
    is_limit_exit = 0 # 标记变量：是否已经存在一个时间维度了
    
    for f in fields_no_alias:
        # 去掉字段两边的反引号并转为大写，用于安全比较
        safe_f = f.strip('`').upper() 
        
        if f not in seen:
            # 【解读】：解决提示词中的规则3（不能同时存在 YEAR_MONTH 和 DATE_YEAR）。
            # 因为图表只能有一个主 X 轴，如果大模型发神经把“年度”和“年月”一起 SELECT 出来了，
            # 后端画图代码会直接懵逼。所以这里判断：只要已经塞进了一个时间字段（is_limit_exit=1），
            # 后面再遇到时间字段，直接 continue 丢弃掉！
            if safe_f in ["business_year_and_month", "DATE_YEAR"] and is_limit_exit:
                continue
                
            dedup_fields.append(f)
            seen.add(f)
            
            # 记录：已经出现过时间维度了
            if safe_f in ["business_year_and_month", "DATE_YEAR"]:
                is_limit_exit = 1
                
    # 6. 重新拼装出干净、规范的 SQL 语句
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
    result = re.sub(r'[\d.-]', '', text)
    return result

def extract_numeric_value(text):
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
        category_parts = [str(row[col]) for col in category_cols if col in row]

        for val_col in value_cols:
            val_display_name = value_name_map.get(val_col, val_col)
            new_col = sep.join(category_parts + [val_display_name])
            grouped[date_value][new_col] = row.get(val_col)

    return [grouped[k] for k in sorted(grouped.keys())]

def select_x(all_keys):
    for i in all_keys:
        if any(char in i for char in ['年度', "年月"]):
            ret = i
            break
        if any(char in i for char in ['名称', "项目"]):
            ret = i
            break
    else:
        ret = all_keys[0]
    return ret

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