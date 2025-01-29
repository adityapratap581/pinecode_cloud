from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from uuid import uuid4
from typing import List, Optional
import json
# from .service import embed_and_insert_email_data, query_retrieval
from .upsert import embed_and_insert_email_data
from .query import query_retrieval
import os
from pinecone import Pinecone, ServerlessSpec
import tempfile
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

class QueryRequest(BaseModel):
    query_text: str

class QueryResponse(BaseModel):
    result: str

pinecone_cred = os.environ.get("PINECONE_CREDENTIAL")
os.environ["PINECONE_API_KEY"] = pinecone_cred
pc = Pinecone(api_key=pinecone_cred)
index_name = 'boldpenguin'
model = SentenceTransformer("all-mpnet-base-v2")
model_z = "gpt-4o-mini"


@app.get('/create_index')
def create_index():
    pc.create_index(
    name=index_name,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud='aws', 
        region='us-east-1'
        )
    )
    return {'message': 'index created'}


@app.post("/upload-email/")
async def upload_email(file: UploadFile = File(...)):
    temp_dir = r'temp'
    file_path = os.path.join(temp_dir, file.filename)
    print(file_path)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    index = pc.Index('boldpenguin')
    transaction_id = await embed_and_insert_email_data(file_path, index,model= model,model_z=model_z)

    email_info = {file.filename: transaction_id}

    print('email_info : ', email_info)

    json_file_path = r'email_info.json'
    
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                data = {} 
    else:
        data = {}

    data.update(email_info)

    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    os.remove(file_path)

    return {"transaction_id": transaction_id}


@app.post("/query-pinecone/", response_model=QueryResponse)
async def query_pinecone(request: QueryRequest):
    json_file_path = r'email_info.json'
    
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        latest_transaction_id = list(data.values())[-1]
    else:
        return {"error": "No transaction data found"}

    index = pc.Index('boldpenguin')

    result = query_retrieval(index, request.query_text, transaction_id=latest_transaction_id,model=model)
    
    return QueryResponse(result=result)