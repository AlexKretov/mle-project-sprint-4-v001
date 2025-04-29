# main.py
from fastapi import FastAPI, HTTPException, Body
import boto3
import pandas as pd
import os
from typing import List
from fastapi.responses import JSONResponse
# music_recommender_ultimate.py
import numpy as np
from implicit.als import AlternatingLeastSquares
import json
from scipy.sparse import csr_matrix, save_npz, load_npz
import polars as pl
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse 
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import aioboto3  # <-- Исправлено здесь
import logging


app = FastAPI()
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],
#    allow_methods=["*"],
#    allow_headers=["*"],
#)
# Добавляем в начало файла (после импортов)
#app.mount("/static", StaticFiles(directory=r"E:\Data\ML-engineer\mle-project-sprint-4-v001\flask\static"), name="static")

# Модифицируем существующий роут /
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
    <html>
        <head>
            <title>Music Recommender</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; }
                .container { padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; }
                button { background: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer; }
                #results { margin-top: 20px; }
                .track { padding: 10px; border-bottom: 1px solid #eee; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Здесь ничего сделать нельзя, запросы отправляются с другой веб-странцицы</h1>
            </div>
        </body>
    </html>
    """

# Конфигурация S3
S3_CONFIG = {
    "service_name": "s3",
    "endpoint_url": "https://storage.yandexcloud.net",
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY")
    #"bucket": os.getenv("S3_BUCKET"),
    #"key": "recsys/data/similar.parquet"
}
# Глобальная переменная для кеширования данных
recommedations = None
def preload_data():
    """Синхронная обертка для асинхронной загрузки данных"""
    import asyncio
    asyncio.run(load_data())

async def load_data():
    global recommedations
    # Если данные уже загружены - пропускаем
    if recommedations is not None:
        return

    client_config = {
        k: v for k, v in S3_CONFIG.items() 
        if k in ['service_name', 'endpoint_url', 
                'aws_access_key_id', 'aws_secret_access_key']
    }
    
    # Ваша логика загрузки данных из S3
    # Например, через aioboto3:
    buffer = BytesIO()

    async with aioboto3.Session().client(**client_config) as s3:
        #s3.download_fileobj(Bucket='s3-student-mle-20250130-833968fcc1', Key='recsys/data/similar_items.parquet', Fileobj=buffer)
        await s3.download_fileobj(Bucket='s3-student-mle-20250130-833968fcc1', Key='recsys/data/cb_predictions.parquet', Fileobj=buffer)
        buffer.seek(0)
        recommedations = pd.read_parquet(buffer)




@app.post("/recommend")
async def test_endpoint(data: dict = Body(...)):
    print(f"Generating recommendations for user: {data['user_id']}")
    logging.debug(f"Generating recommendations for user: {data['user_id']}")
    user_id = data['user_id']
    logging.debug(f"[1] Loading data: {len(recommedations)}")
    print(f"[1] Loading data: {len(recommedations)}")
    cb_recommedations = recommedations[recommedations['user_id'] == user_id]
    print(f"[2] CB recommendations: {len(cb_recommedations)}")
    logging.debug(f"[2] CB recommendations: {len(cb_recommedations)}")
    recd_ids = cb_recommedations.nlargest(10, 'cb_score')['track_id'].to_list()
    print(f"[3] Recd track IDs: {recd_ids}")
    logging.debug(f"[3] Recd track IDs: {recd_ids}")

    
    return {
        "status": "processed",
        "track_ids": recd_ids
    }


if __name__ == "__main__":
    import uvicorn
    preload_data()

    uvicorn.run(app, host="127.0.0.1", port=7000)
