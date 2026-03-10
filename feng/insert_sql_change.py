import re
import os
import shutil
from datetime import datetime


def convert_identifier_quotes(line):
    """
    将 SQL 语句中作为标识符（表名、列名）的双引号替换为反引号。
    逻辑：匹配 "word" 模式，但排除掉那些明显是字符串值的情况。
    在 INSERT 语句中，通常括号内、逗号后、FROM/INTO 后的双引号都是标识符。
    """
    # 策略：替换所有独立的双引号包裹的单词为反引号包裹
    # 正则解释：匹配 " 开头，中间是非双引号字符， " 结尾
    # 注意：这可能会误伤包含在字符串值里的双引号，但在标准 INSERT 语句中，
    # 列名和表名通常紧跟在 ( ) , INTO 等关键字后，而值通常在 VALUES 后面。
    # 更安全的做法是只替换特定上下文的双引号，但为了通用性，我们采用启发式替换：
    # 如果双引号内的内容不包含空格且看起来像标识符，则替换。

    # 简单且高效的方案：直接替换所有 " 为 `
    # 前提：SQL 文件中字符串值不使用双引号包裹（标准 SQL 字符串用单引号 '）
    # 达梦/Oracle 风格通常字符串用单引号，标识符用双引号。
    # 如果字符串值里真的包含双引号（例如 "He said \"Hello\""），这种简单替换会出错。
    # 但观察你的样例，字符串都是单引号 '...'，所以直接全局替换是安全的。

    return line.replace('"', '`')


def convert_date_format(line):
    """
    识别并转换日期格式：'YYYY/M/D' 或 'YYYY/MM/DD' -> 'YYYY-MM-DD'
    只处理被单引号包裹的疑似日期格式。
    """
    # 正则匹配：单引号包裹的 数字/数字/数字
    # 组1: 年, 组2: 月, 组3: 日
    date_pattern = r"'(\d{4})/(\d{1,2})/(\d{1,2})'"

    def replacer(match):
        year, month, day = match.groups()
        # 补零
        month = month.zfill(2)
        day = day.zfill(2)
        return f"'{year}-{month}-{day}'"

    return re.sub(date_pattern, replacer, line)


def process_line(line):
    """处理单行逻辑"""
    # 1. 跳过空行或纯注释行（可选，这里保留原样）
    if not line.strip():
        return line

    # 2. 替换标识符引号
    # 注意：如果 VALUES 部分的字符串值里含有双引号，这一步可能会有风险。
    # 但基于达梦导出习惯，字符串值通常用单引号。
    # 为了更安全，我们可以只处理 INSERT 和 CREATE 相关的行，或者信任单引号包裹的是值。
    # 这里采用全局替换，因为你的样例中值都是单引号。
    new_line = convert_identifier_quotes(line)

    # 3. 转换日期格式
    new_line = convert_date_format(new_line)

    return new_line


def convert_sql_file(input_path, output_path=None, keep_backup=True):
    if not os.path.exists(input_path):
        print(f"错误：文件不存在 - {input_path}")
        return

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_mysql{ext}"


    print(f"开始转换：{input_path} -> {output_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
                open(output_path, 'w', encoding='utf-8') as f_out:

            count = 0
            for line in f_in:
                # 简单的优化：如果不是 INSERT 语句，且不需要改引号（比如注释），可以直接写入
                # 但为了统一风格（表名引号），我们处理每一行
                new_line = process_line(line)
                f_out.write(new_line)
                count += 1

        print(f"转换完成！共处理 {count} 行。")
        print(f"输出文件：{output_path}")

    except Exception as e:
        print(f"发生错误：{e}")
        # 如果出错，尝试恢复备份（略）


if __name__ == "__main__":
    # 配置区域
    # 请将此处修改为你的实际文件名
    input_file = "RDYS_LD_YSZX_YBGGYS_SBJSRWCQK数据.sql"

    # 如果没有文件，创建一个测试文件供演示
    if not os.path.exists(input_file):
        print("文件不存在！")
    else:
    # 执行转换
        convert_sql_file(input_file)
# 导入数据
# cat all_insert.sql | docker exec -i mysql80 mysql -uroot -paidb369 RDYS_PUBLIC_TBS
# 查看导入情况
# docker exec -it mysql80 mysql -uroot -paidb369 -e "USE RDYS_PUBLIC_TBS; SELECT COUNT(*) as total_rows FROM RDYS_LD_YSZX_YBGGYS_SBJSRWCQK;"