import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware

from dependencies import init_resources, close_resources
from api.chat import router as chat_router
from api.image import router as image_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print("应用启动：正在加载模型、缓存与数据库资源...")
    init_resources()
    yield
    # 退出时执行
    print("应用关闭：正在释放资源...")
    close_resources()

app = FastAPI(
    debug=True,
    title="知识库+api",
    description="目前支持的表有：软件著作权表-software_copyright、农合机构核心经营指标表-nonghe_institution_indicators、企业财务数据表-bus_financial_data等", 
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# 挂载路由
app.include_router(chat_router, prefix="/v1")
app.include_router(image_router)

if __name__ == '__main__':
    uvicorn.run('main:app', host='192.168.100.160', port=8792, workers=1)