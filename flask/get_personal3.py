from fastapi import FastAPI
import requests
import pandas as pd
import aioboto3
import os
from io import BytesIO
S3_CONFIG = {
    "service_name": "s3",
    "endpoint_url": "https://storage.yandexcloud.net",
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY")
}
recommedations = None
app = FastAPI()
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
@app.get("/get_tracks")
def get_tracks(user_id: int):
    # Отправляем запрос к другому микросервису
    

    # Находим строки, где track_id is in track_list
    try:
        result = recommedations[recommedations['user_id']==user_id].nlargest(10, 'cb_score')['track_id'].to_list()
    except:
        return {'status': 'failed', 'data': []}
    return {'status': 'success', 'data': result}

if __name__ == "__main__":
    import uvicorn
    preload_data()

    uvicorn.run(app, host="127.0.0.1", port=7000)
