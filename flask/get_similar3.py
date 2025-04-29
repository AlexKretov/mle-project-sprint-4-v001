from fastapi import FastAPI,Body
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
similars = None
app = FastAPI()
def preload_data():
    """Синхронная обертка для асинхронной загрузки данных"""
    import asyncio
    asyncio.run(load_data())

async def load_data():
    global similars
    # Если данные уже загружены - пропускаем
    if similars is not None:
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
        await s3.download_fileobj(Bucket='s3-student-mle-20250130-833968fcc1', Key='recsys/data/similar_items.parquet', Fileobj=buffer)
        buffer.seek(0)
        similars = pd.read_parquet(buffer)
@app.get("/get_tracks")
def get_tracks(track_list: list):
    fresh = track_list[:3]
    try:
        result = similars.query('item_id_1 in @fresh').nlargest(20, 'score')['item_id_2'].to_list()
        similar_ids = [x for x in result if x not in track_list][:10]
    except:
        return {'status': 'failed', 'data': []}
    return {'status': 'success', 'data': similar_ids}



if __name__ == "__main__":
    import uvicorn
    preload_data()

    uvicorn.run(app, host="127.0.0.1", port=6000)
