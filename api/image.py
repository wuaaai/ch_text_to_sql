import json
from fastapi import APIRouter, Request, Form, Depends
from dependencies import get_res, AppResources
from prompts import PROMPT2, DEMO_DATA1, DEMO_DATA2
from core.visual import create_image
from config import API_CONFIG

router = APIRouter()

@router.post("/get_image_info", summary="获取图片信息")
async def get_image_info(
        request: Request,
        data: str = Form(description='返回的查询数据', default=""),
        table_name: str = Form(description='中文表名', default=''),
        question: str = Form(description='用户问题'),
        res: AppResources = Depends(get_res)
):
    cache_data = res.redis.get(question.strip() + "<-split->cache")
    if cache_data:
        return {"status": "success", "data": cache_data}
    else:
        task_status = res.redis.get(question.strip() + "_task_cache")
        if task_status:
            return {"status": "error", "data": task_status}
        else:
            if data and table_name:
                try:
                    prompt = PROMPT2.format(data=data, table_name=table_name, question=question, demo_data1=DEMO_DATA1, demo_data2=DEMO_DATA2)
                    create_params2 = {
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": question},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 8192,
                        "stream": False,
                        "timeout": API_CONFIG["timeout"]
                    }
                    response2 = res.chat.chat.completions.create(**create_params2)
                    ret2 = response2.choices[0].message.content
                    
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
                        res.redis.set(question.strip() + "<-split->cache", json.dumps(chart_config, ensure_ascii=False), ex=3600)
                        return {"status": "success", "data": chart_config}
                    else:
                        return {"status": "error", "data": "数据格式不支持绘图！"}
                except Exception as e:
                    return {"status": "error", "data": str(e)}
            else:
                return {"status": "error", "data": "数据格式不支持绘图！"}