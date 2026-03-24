from fastapi import FastAPI
from fastapi.params import Body
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import json
from starlette.middleware.cors import CORSMiddleware
from modelscope import AutoModelForCausalLM, AutoTokenizer



app = FastAPI(debug=True,
              title="知识库+api",
              description="目前支持的表有：软件著作权表-software_copyright、农合机构核心经营指标表-nonghe_institution_indicators、企业财务数据表-bus_financial_data、企业主要人员表-key_people、工商信息+基础信息表-business_information、企业对外投资表-out_invest、作品著作权表-creation_copyright、企业变更记录表-change_log、专利信息表-patent_information、企业间接对外投资表-out_invest_indirect、股东信息表-shareholder_information、资质证书表-certification", )



@app.on_event("startup")
async def load_models():
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
    app.state.sql_tokenizer = AutoTokenizer.from_pretrained(local_path)




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

@app.post("/text2sql")
async def text2sql(
        question: str = Body(description='用户问题'),
        demo: str = Body(description='schema'),
        evidence: str = Body(description='用户问题'),
        ):
    try:
        demo = json.loads(demo)
        prompt = app.state.nl2sqlite_template_cn.format(
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
        return {"status":"success","sql_query": sql_query}
    except Exception as e:
        return {"status":"error","message": str(e)}




app.add_middleware(  # 解决跨域问题
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('model_main:app', host=f'192.168.100.160', port=8989, workers=1)
