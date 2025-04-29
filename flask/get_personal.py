# main.py
from fastapi import FastAPI, HTTPException, Request, Body, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import conlist
from typing import List
import os
import pandas as pd
import aioboto3
import uvicorn
from io import BytesIO
import logging

# Конфигурация
class Config:
    S3 = {
        "endpoint_url": "https://storage.yandexcloud.net",
        "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "bucket": "s3-student-mle-20250130-833968fcc1",
        "paths": {
            "tracks": "recsys/data/items.parquet",
            "predictions": "recsys/data/cb_predictions.parquet"
        }
    }
    CACHE_SIZE = 100
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Хранилище данных
class DataStore:
    def __init__(self):
        self.tracks = None
        self.predictions = None
        self.logger = logging.getLogger(__name__)

data_store = DataStore()

# Инициализация приложения
app = FastAPI(title="Music Recommender API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статические файлы
app.mount("/static", StaticFiles(
    directory=r"E:\Data\ML-engineer\mle-project-sprint-4-v001\flask\static"
), name="static")

# Обработчики жизненного цикла
@app.on_event("startup")
async def initialize_data():
    """Асинхронная инициализация данных при запуске"""
    try:
        session = aioboto3.Session()
        async with session.client(
            service_name='s3',
            endpoint_url=Config.S3["endpoint_url"],
            aws_access_key_id=Config.S3["access_key"],
            aws_secret_access_key=Config.S3["secret_key"]
        ) as s3_client:
            
            
            # Загрузка предвычисленных предсказаний
            pred_buffer = BytesIO()
            await s3_client.download_fileobj(
                Bucket=Config.S3["bucket"],
                Key=Config.S3["paths"]["predictions"],
                Fileobj=pred_buffer
            )
            pred_buffer.seek(0)
            global data_store
            data_store.predictions = pd.read_parquet(pred_buffer)

        data_store.logger.info("Data loading completed")
        logging.debug(f"Data loading completed: {len(data_store.predictions)}")
    except Exception as e:
        data_store.logger.error(f"Data loading failed: {str(e)}")
        raise RuntimeError("Initialization failed")

# Роуты
@app.get("/", response_class=HTMLResponse)
async def render_interface(request: Request):
    """Главная страница с интерфейсом"""
    return """
    <html>
        <head>
            <title>CB-recomedator. Send response elsewhere</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; }
                .container { padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; }
                button { background: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer; }
                #results { margin-top: 20px; }
                .track { padding: 10px; border-bottom: 1px solid #eee; }
            </style>
                    <body>
            <div class="container">
                <h1>Здесь ничего сделать нельзя, запросы отправляются с другой веб-странцицы</h1>
            </div>
        </body>
        </head>

    </html>
    """

@app.post("/recommend")
async def generate_recommendations(
    data: dict = Body(...)): 
    """Генерация рекомендаций для пользователя"""
    try:
        # Валидация данных
        #if not data_store.predictions or not data_store.tracks:
        #    raise HTTPException(503, "Service initializing")

        # Основная логика
        main_user_id = data['user_id']
        logging.debug(f"Generating recommendations for user: {main_user_id}")
        # Поиск рекомендаций
        ascii
        predictions = data_store.predictions[
            data_store.predictions.user_id == main_user_id
        ]
        logging.debug(f"Dataset for this user is that long: {len(predictions)}")
        if predictions.empty:
            return JSONResponse({"status": "empty","recommendations": []})
        
        top_tracks = predictions.nlargest(10, 'cb_score')['track_id'].tolist()
        logging.debug(f"Got top tracks {top_tracks}")
        
        return JSONResponse({"status": "processed","track_ids": top_tracks})

    except HTTPException:
        raise
    except Exception as e:
        data_store.logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Internal server error")

# Точка входа
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=7000
    )
