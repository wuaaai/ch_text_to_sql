# core/visual.py
import re
import json
import random
import datetime
from echarts import Echart, Legend, Bar, Line, Axis, Tooltip, Pie
from core.utils import extract_non_numeric, extract_numeric_value

def create_image(categories, values, title, y_titles):
    chart_colors = [
        '#5470C6', '#91CC75', '#EE6666', '#FAC858', '#73C0DE', 
        '#3BA272', '#FC8452', '#9A60B4', '#E5A3C9', '#FADB7A', 
        '#A0A7D4', '#7FCFC0', '#F9B8B8', '#B1D580', '#E6965C', 
        '#C77EB5', '#6DC8BF', '#FFB7A5', '#D9A7E0', '#A8D5E5'
    ]
    chart = Echart(title)
    use_tip = None
    use_lg = None
    is_xy = True

    def get_unit(text):
        match = re.search(r'（(.*?)）', text)
        if match:
            return match.group(1)
        else:
            return ""

    for n, x in enumerate(values):
        if "bar" in x:
            use_lg = Legend(data=y_titles, left='center', top=20, textStyle={'fontSize': 12})
            color = random.choice(chart_colors)
            chart_colors.remove(color)
            chart.use(Bar(name=y_titles[n], data=x['bar'], itemStyle={'color': color}))
        elif "line" in x:
            use_lg = Legend(data=y_titles, left='center', top=20, textStyle={'fontSize': 12})
            color = random.choice(chart_colors)
            chart_colors.remove(color)
            chart.use(Line(name=y_titles[n], data=x['line'], smooth=True, lineStyle={'width': 2, 'color': color}, itemStyle={'color': color}, symbol='circle', symbolSize=8))
        elif "pie" in x:
            is_xy = False
            use_tip = Tooltip(trigger='item', formatter='{a}<br/>{b}: {c}')
            use_lg = Legend(data=categories, itemGap=12, textStyle={'fontSize': 12})
            chart.use(Pie(
                name=y_titles[n],
                data=[{"name": categories[n], "value": i} for n, i in enumerate(x['pie'])],
                radius='55%', center=['50%', '60%'],
                label={'show': True, 'position': 'outside', 'formatter': '{b}:{d}%', 'alignTo': 'labelLine', 'bleedMargin': 10},
                labelLine={'length': 10, 'length2': 10, 'smooth': True}
            ))

    chart.use(use_lg)
    unit_list = []
    if is_xy:
        if len(y_titles) <= 1:
            tip_formatter = 'x轴:{b}<br/>{a}: {c}' + get_unit(y_titles[0])
            unit_list.append(get_unit(y_titles[0]))
        else:
            tip_formatter = 'x轴:{b}<br/>'
            for index, u in enumerate(y_titles):
                unit_list.append(get_unit(u))
                tip_formatter += '{a' + str(index) + '}: {c' + str(index) + '}' + get_unit(u)
                if index != len(y_titles) - 1:
                    tip_formatter += '<br/>'

        use_tip = Tooltip(trigger='axis', formatter=tip_formatter, backgroundColor='rgba(50,50,50,0.9)', textStyle={'color': '#fff'}, borderColor='#333', borderWidth=1)
        chart.use(Axis(type='category', position='bottom', data=categories, min=0, nameLocation='middle', nameGap=25, nameTextStyle={'fontSize': 14, 'fontWeight': 'bold'}, axisTick={'alignWithLabel': True}, splitLine={'show': True, 'lineStyle': {'color': ['#eee'], 'type': 'dashed'}}))
        chart.use(Axis(type='value', position='left', name='数值', min=0, nameLocation='end', nameGap=15, splitLine={'show': True, 'lineStyle': {'color': ['#eee'], 'type': 'dashed'}}))

    chart.use(use_tip)
    config = chart.json
    if isinstance(config, str):
        config = json.loads(config) 

    config['title']['bottom'] = 0
    config['title']['left'] = 'center'
    if not is_xy:
        config.pop('xAxis', None)
        config.pop('yAxis', None)
    else:
        unique_units = set([i for i in unit_list if i])
        money_all_values = []
        for n, y in enumerate(unit_list):
            if y == '万元':
                money_all_values.extend(list(values[n].values())[0])

        if money_all_values:
            money_max_val = max(money_all_values)
            money_min_val = min(money_all_values)
            money_range_val = money_max_val - money_min_val
            if money_max_val < 0 and money_min_val < 0:
                money_min_axis = money_min_val - 0.1 * money_range_val
                money_max_axis = 0
            elif money_max_val > 0 and money_min_val > 0:
                money_min_axis = 0
                money_max_axis = money_max_val + 0.1 * money_range_val
            else:
                money_min_axis = money_min_val - 0.1 * money_range_val
                money_max_axis = money_max_val + 0.1 * money_range_val
            config['yAxis'][0]['min'] = money_min_axis
            config['yAxis'][0]['max'] = money_max_axis
            
        if len(unique_units) > 1 and any(u in ['%', '百分比'] for u in unit_list):
            percent_all_values = []
            for n, y in enumerate(unit_list):
                if y in ['%', '百分比']:
                    percent_all_values.extend(list(values[n].values())[0])
                    if 'series' in config and len(config['series']) > 1:
                        config['series'][n]['yAxisIndex'] = 1
            max_val = max(percent_all_values)
            min_val = min(percent_all_values)
            range_val = max_val - min_val
            min_axis = min_val - 0.1 * range_val
            max_axis = max_val + 0.1 * range_val
            config['yAxis'].append({
                'type': 'value', 'position': 'right', 'name': '百分比 (%)',
                'nameLocation': 'end', 'min': min_axis, 'max': max_axis,
                'axisLabel': {'formatter': '{value} %'}, 'splitLine': {'show': False}
            })

    return config

def creat_image_data(data, x_axis_label, question, resources):
    resources.redis.set(question.strip() + "_task_cache", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ex=3600)
    try:
        print("========== [DEBUG] 进入后台画图任务 ==========")
        print(f"[DEBUG] 准备画图的第一行数据: {data[0]}")
        
        json_data = {
            "can_plot": 1,
            "title": "",
            "x_axis_label": x_axis_label,
            "y_axis_label": [],
            "categories": [],
            "values": [],
        }
        if len(data) == 1 or len(data[0]) <= 1:
            json_data["can_plot"] = 0
        else:
            all_keys = list(set(data[0].keys()))
            y_axis_label = [i for i in all_keys if i != json_data["x_axis_label"]]
            is_pie = 0
            if "饼图" in question and len(all_keys) == 2:
                is_pie = 1

            merry = []
            for w in data:
                json_data['categories'].append(w[json_data["x_axis_label"]])
                if is_pie:
                    if json_data['values']:
                        json_data['values'][0]['pie'].append(extract_numeric_value(str(w[y_axis_label[0]])))
                    else:
                        json_data['values'] = [{"pie": [extract_numeric_value(str(w[y_axis_label[0]]))]}]
                else:
                    if json_data['values']:
                        for n, i in enumerate(y_axis_label):
                            json_data['values'][n][merry[n]].append(extract_numeric_value(str(w[i])))
                    else:
                        for n, i in enumerate(y_axis_label):
                            # ==========================================
                            # 👇 终极防呆判定逻辑 👇
                            # ==========================================
                            val_str = str(w[i]) # 强行转成字符串，防止 float 报错中断程序
                            
                            # 1. 绝对指令：优先听从用户在问题里的明确要求
                            if "折线" in question:
                                name = "line"
                            elif "柱状" in question:
                                name = "bar"
                            # 2. 智能推断：检查数值(val_str)和列名(i)
                            elif "%" in val_str or "百分比" in val_str or "%" in i or "百分比" in i or "率" in i:
                                name = "line"
                            # 3. 兜底默认值
                            else:
                                name = "bar"
                            
                            merry.append(name)
                            json_data['values'].append({name: [extract_numeric_value(val_str)]})

            # 提取单位时，也必须强转 str 防崩溃
            json_data['y_axis_label'] = [i + f"（{extract_non_numeric(str(data[0][i]))}）" for i in all_keys if i != json_data["x_axis_label"]]

        if json_data["can_plot"]:
            categories = json_data['categories']
            values = json_data['values']
            title = json_data['title']
            y_titles = json_data['y_axis_label']
            chart_config = create_image(categories, values, title, y_titles)
            resources.redis.set(question.strip() + "<-split->cache", json.dumps(chart_config, ensure_ascii=False), ex=3600)
            print(f"[DEBUG] ✅ 画图配置生成成功，已写入 Redis！决定使用的图表类型: {merry}")
            
    except Exception as e:
        # 如果报错了，一定要把它打印出来让你看见，而不是默默 pass！
        print(f"========== [ERROR] ❌ 执行后台画图任务报错 ==========")
        print(f"报错详情: {str(e)}")
        import traceback
        traceback.print_exc()
        pass
    finally:
        resources.redis.delete(question.strip() + "_task_cache")