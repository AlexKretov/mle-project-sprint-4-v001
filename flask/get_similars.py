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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Добавляем в начало файла (после импортов)
app.mount("/static", StaticFiles(directory=r"E:\Data\ML-engineer\mle-project-sprint-4-v001\flask\static"), name="static")

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
                <h1>Music Track Recommender</h1>
                <input type="text" id="trackIds" placeholder="Enter track IDs separated by commas (e.g. 123,456,789)">
                <button onclick="getRecommendations()">Get Recommendations</button>
                <div id="results"></div>
            </div>
            
            <script>
                function getRecommendations() {
                    const input = document.getElementById('trackIds').value;
                    const trackIds = input.split(',').map(id => parseInt(id.trim())).filter(id => !isNaN(id));
                    
                    if(trackIds.length === 0) {
                        alert("Please enter valid track IDs");
                        return;
                    }

                    fetch('/recommend', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(trackIds)
                    })
                    .then(response => {
                        if(!response.ok) throw new Error('Network error');
                        return response.json();
                    })
                    .then(data => {
                        const resultsDiv = document.getElementById('results');
                        resultsDiv.innerHTML = data.recommendations
                            .map(track => `
                                <div class="track">
                                    <strong>${track[0]}</strong> - ${track[1]}
                                </div>
                            `)
                            .join('');
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error getting recommendations: ' + error.message);
                    });
                }
            </script>
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
cached_similarities = None
def preload_data():
    """Синхронная обертка для асинхронной загрузки данных"""
    import asyncio
    asyncio.run(load_data())

async def load_data():
    global cached_similarities
    global tracks
    # Если данные уже загружены - пропускаем
    if cached_similarities is not None:
        return

    client_config = {
        k: v for k, v in S3_CONFIG.items() 
        if k in ['service_name', 'endpoint_url', 
                'aws_access_key_id', 'aws_secret_access_key']
    }
    
    # Ваша логика загрузки данных из S3
    # Например, через aioboto3:
    buffer = BytesIO()
    bu = BytesIO()
    async with aioboto3.Session().client(**client_config) as s3:
        #s3.download_fileobj(Bucket='s3-student-mle-20250130-833968fcc1', Key='recsys/data/similar_items.parquet', Fileobj=buffer)
        await s3.download_fileobj(Bucket='s3-student-mle-20250130-833968fcc1', Key='recsys/data/similar_items.parquet', Fileobj=buffer)
        buffer.seek(0)
        cached_similarities = pd.read_parquet(buffer)
    async with aioboto3.Session().client(**client_config) as s3:
        #s3.download_fileobj(Bucket='s3-student-mle-20250130-833968fcc1', Key='recsys/data/items.parquet', Fileobj=bu)
        await s3.download_fileobj(Bucket='s3-student-mle-20250130-833968fcc1', Key='recsys/data/items.parquet', Fileobj=bu)
        bu.seek(0)
        tracks = pd.read_parquet(bu)



@app.post("/test")
async def test_endpoint(data: dict = Body(...)):
    print(f"Received track IDs: {data['track_ids']}")
    got_ids = data['track_ids']
    last = got_ids[-3:]
    similar_ids = cached_similarities.query('item_id_1 in @last').nlargest(3, 'score')['item_id_2'].to_list()
    similar_ids = [x for x in similar_ids if x not in got_ids][:3]
    print(f"[5] Similar track IDs: {similar_ids}")
    
    return {
        "status": "processed",
        "track_ids": similar_ids
    }

@app.post("/recommend")
async def recommend(track_ids: List[int]):
#async def recommend(request_json):
    #track_ids = request_json['track_ids']
    #print(f"[1] Received request with track_ids: {track_ids}")
    
    try:
        # Загрузка данных
        print("[2] Loading data...")
        #tracks, similar = await load_data()
        #print(f"[3] Loaded: {len(tracks)} tracks, {len(similar)} similarities")
        
        
        # Получение рекомендаций
        print("[4] Getting similar tracks...")
        last = track_ids[-3:]
        similar_ids = cached_similarities.query('item_id_1 in @last').nlargest(3, 'score')['item_id_2'].to_list()
        similar_ids = [x for x in similar_ids if x not in track_ids][:3]
        print(f"[5] Similar track IDs: {similar_ids}")
        
        # Формирование результатов
        print("[6] Querying tracks...")
        results = tracks.query('track_id in @similar_ids')
        cool_results = [(row['track_name'], row['artists']) for _, row in results.iterrows()]
        print(f"[7] Final results: {cool_results}")
        
        #return {"recommendations": cool_results}
        return {"track_ids": similar_ids}
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    preload_data()

    uvicorn.run(app, host="127.0.0.1", port=8000)
