import pymysql

# ================= 配置区 =================
DB_CONFIG = {
    'host': '192.168.100.160',       # 数据库地址
    'port': 3306,              # 数据库端口
    'user': 'hbch',   # 数据库用户名
    'password': 'hbch2711',# 数据库密码
    'charset': 'utf8mb4',      # 使用 utf8mb4 支持完整中文
    'cursorclass': pymysql.cursors.DictCursor
}

OLD_DB = 'RDYS_PUBLIC_TBS'   # 你的旧数据库名
NEW_DB = 'RDYS_PUBLIC_TBS_WU'   # 你想创建的新数据库名

OLD_TABLE = 'RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC'   # 旧表名
NEW_TABLE = 'RDYS_LD_YSSC_YSZX_QSYBGGYSZCWC'   # 新表名

# ================= 迁移逻辑 =================
def migrate_database_and_data():
    # 字段映射关系字典升级：'旧字段': ('超长全拼英文字段', '你的详细注释')
    # 专为 Text-to-SQL 优化：无任何缩写，极度语义化
    field_mapping = {
        'DATA_ID': ('Primary Key', '数据的唯一标识，主键'),
        'YEAR_MONTH': ('business_year_and_month', '记录业务发生的年份和月份，格式为YYYYMM'),
        'XM_CODE': ('project_code', '预算项目编码'),
        'XM_NAME': ('project_name', '预算项目名称'),
        'YSS': ('full_year_total_budget_amount', '项目全年的总预算额度'),
        'BYS_JE': ('current_month_actual_amount', '当前月份发生的实际金额'),
        'BYS_SNTYS': ('last_year_same_month_actual_amount', '去年同月份发生的实际金额'),
        'BYS_TBE': ('year_over_year_current_month_difference_amount', '本月金额减去上年本月金额的差额（正数为增加，负数为减少）'),
        'BYS_TBB': ('year_over_year_current_month_difference_percentage', '本月金额相较于上年同月的增长或减少百分比'),
        'ZBYLJS_JE': ('current_year_accumulated_amount_to_current_month', '从本年1月截至本月的累计发生总额'),
        'ZBYLJS_WYSS': ('current_year_accumulated_amount_execution_percentage', '本年累计金额占全年预算金额的百分比'),
        'ZBYLJS_SNTQS': ('last_year_accumulated_amount_to_same_month', '去年1月至去年同月的累计发生总额'),
        'ZBYLJS_TBE': ('year_over_year_accumulated_difference_amount', '本年累计金额减去上年同期累计金额的增减额'),
        'ZBYLJS_TBB': ('year_over_year_accumulated_difference_percentage', '本年累计金额相较于上年同期累计金额的增减百分比'),
        'BNSY_LJS': ('current_year_accumulated_amount_to_last_month', '截至本年上一个月份的累计金额'),
        'SNSY_LJS': ('last_year_accumulated_amount_to_last_month', '截至去年上一个月份的累计金额')
    }

    # 构建 SELECT 子句，提取元组中的新英文字段名（索引 0）
    select_clauses = [f"`{old}` AS `{new_info[0]}`" for old, new_info in field_mapping.items()]
    select_sql = ",\n        ".join(select_clauses)

    # 1. 创建新数据库
    create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{NEW_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

    # 2. 跨库建表并插入数据
    create_and_insert_sql = f"""
    CREATE TABLE `{NEW_DB}`.`{NEW_TABLE}` AS
    SELECT
        {select_sql}
    FROM `{OLD_DB}`.`{OLD_TABLE}`;
    """

    # 3. 动态获取你设置的主键新名字，并生成设置主键的 SQL
    pk_new_name = field_mapping['DATA_ID'][0]
    alter_pk_sql = f"ALTER TABLE `{NEW_DB}`.`{NEW_TABLE}` ADD PRIMARY KEY (`{pk_new_name}`);"

    print("正在连接 MySQL 服务器...")
    connection = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            # 第一步：创建新数据库
            print(f"1. 正在创建新数据库 `{NEW_DB}`...")
            cursor.execute(create_db_sql)
            
            # 第二步：跨库建表并迁移数据
            print(f"2. 正在从 `{OLD_DB}` 将数据迁移到 `{NEW_DB}`.`{NEW_TABLE}`...")
            cursor.execute(create_and_insert_sql)
            
            # 第三步：设置主键
            print(f"3. 正在为新表设置主键 (`{pk_new_name}`)...")
            cursor.execute(alter_pk_sql)

            # 第四步：给新表的字段加注释
            print("4. 正在提取字段结构并写入注释...")
            # 查出新表继承过来的数据类型
            cursor.execute(f"SHOW COLUMNS FROM `{NEW_DB}`.`{NEW_TABLE}`")
            columns_info = cursor.fetchall()

            # 生成一个 {新字段名: 注释内容} 的字典，方便查找
            comment_lookup = {info[0]: info[1] for info in field_mapping.values()}

            for col in columns_info:
                col_name = col['Field']
                col_type = col['Type']
                # 保持原有的 Null 约束
                is_nullable = "NULL" if col['Null'] == 'YES' else "NOT NULL"
                
                if col_name in comment_lookup:
                    comment_text = comment_lookup[col_name]
                    # 防止注释里有单引号导致 SQL 语法错误，做一个替换保护
                    safe_comment = comment_text.replace("'", "''") 
                    
                    # 拼接并执行增加注释的 SQL
                    alter_comment_sql = f"ALTER TABLE `{NEW_DB}`.`{NEW_TABLE}` MODIFY COLUMN `{col_name}` {col_type} {is_nullable} COMMENT '{safe_comment}';"
                    cursor.execute(alter_comment_sql)
            
        # 提交更改
        connection.commit()
        print("✅ 新数据库创建、数据跨库迁移及字段注释写入成功完成！")

    except pymysql.Error as e:
        if connection:
            connection.rollback()
        print(f"❌ 数据库操作失败: {e}")
    finally:
        if connection:
            connection.close()
            print("数据库连接已关闭。")

if __name__ == "__main__":
    migrate_database_and_data()