import logging
from fastapi import FastAPI, Body
import requests
import pandas as pd
import boto3
import os
from io import BytesIO
from dotenv import load_dotenv
import time

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

df = None
app = FastAPI()
load_dotenv(r'flask\.env')

def preload_data():
    """Синхронная обертка для асинхронной загрузки данных"""
    logger.info("Starting data preloading...")
    try:
        import asyncio
        asyncio.run(load_data())
        logger.info("Data preloading completed successfully")
    except Exception as e:
        logger.error(f"Data preloading failed: {str(e)}", exc_info=True)

async def load_data():
    global df
    logger.debug("Initializing S3 client")
    try:
        s3 = boto3.client('s3', 
                        endpoint_url='https://storage.yandexcloud.net',
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
        
        logger.info(f"Checking S3 bucket connection: {os.getenv('S3_BUCKET')}")
        s3.head_bucket(Bucket=os.getenv("S3_BUCKET"))
        logger.info("S3 bucket connection successful")
        
        buffer = BytesIO()
        logger.debug("Starting file download from S3")
        start_time = time.time()
        
        s3.download_fileobj(
            Bucket=os.getenv("S3_BUCKET"),
            Key='recsys/data/items.parquet',
            Fileobj=buffer
        )
        
        logger.info(f"File downloaded successfully in {time.time() - start_time:.2f}s")
        buffer.seek(0)
        
        logger.debug("Reading parquet data")
        df = pd.read_parquet(buffer)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}", exc_info=True)
        raise

def get_tracks(user_id: int):
    logger.debug(f"Getting tracks for user {user_id}")
    try:
        url = f"http://127.0.0.1:7000/get_tracks?user_id={user_id}"
        logger.info(f"Requesting: {url}")
        response = requests.get(url)
        logger.debug(f"Response status: {response.status_code}")
        return response.json()
    except Exception as e:
        logger.error(f"Error in get_tracks: {str(e)}")
        return {'data': []}

def get_history(user_id: int):
    logger.debug(f"Getting history for user {user_id}")
    try:
        url = f"http://127.0.0.1:8000/get_tracks?user_id={user_id}"
        logger.info(f"Requesting: {url}")
        response = requests.get(url)
        logger.debug(f"Response status: {response.status_code}")
        return response.json()
    except Exception as e:
        logger.error(f"Error in get_history: {str(e)}")
        return {'data': []}

def similars(track_ids):
    logger.debug(f"Getting similar tracks for {len(track_ids)} tracks")
    try:
        url = "http://127.0.0.1:6000/get_tracks"
        logger.info(f"Requesting: {url} with {len(track_ids)} track IDs")
        response = requests.get(url, json=track_ids)
        logger.debug(f"Response status: {response.status_code}")
        return response.json()
    except Exception as e:
        logger.error(f"Error in similars: {str(e)}")
        return {'data': []}

def get_pops(track_ids):
    logger.debug(f"Getting popular tracks for {len(track_ids)} tracks")
    try:
        url = "http://127.0.0.1:8000/get_pop_tracks"
        logger.info(f"Requesting: {url}")
        response = requests.get(url, json=track_ids)
        logger.debug(f"Response status: {response.status_code}")
        return response.json()
    except Exception as e:
        logger.error(f"Error in get_pops: {str(e)}")
        return {'data': []}

@app.get("/get_playlist")
def get_playlist(user_id: int):
    logger.info(f"Received playlist request for user {user_id}")
    stats = {'personal':0,'similars':0, 'popular':0}
    
    try:
        if df is None:
            logger.warning("Dataframe not loaded! Returning empty playlist")
            return {'playlist': [], 'stats': stats}
        
        logger.debug("Fetching recommendation data")
        rex = get_tracks(user_id)
        hix = get_history(user_id)
        sims = similars(hix['data'])
        pops = get_pops(hix['data'])
        
        logger.info(f"Recommendation counts - Rex: {len(rex['data'])}, Hix: {len(hix['data'])}")
        
        out = []
        if len(rex['data']) >= 7:
            logger.debug("Using primary recommendation strategy")
            out = rex['data'][:7] + sims['data'][:3]
            stats['personal'] = 7
            stats['similars'] = 3
        elif len(rex['data']) == 0:
            logger.debug("Using fallback strategy")
            if len(hix['data']) == 10:
                out = sims['data'][:10]
                stats['similars'] = 10
            else:
                out = pops
                stats['popular'] = 10
        
        logger.debug(f"Final track IDs: {len(out)}")
        logger.info("Constructing playlist names")
        
        result_df = df.query('track_id in @out')
        logger.debug(f"Found {len(result_df)} tracks in dataframe")
        
        cool_result = [
            f"{row['track_name']}-{row['artists']}"
            for _, row in result_df.iterrows()
        ]
        
        logger.info(f"Returning playlist with {len(cool_result)} tracks")
        return {'playlist': cool_result, 'stats': stats}
    
    except Exception as e:
        logger.error(f"Error processing playlist request: {str(e)}", exc_info=True)
        return {'playlist': [], 'stats': stats}

if __name__ == "__main__":
    logger.info("Starting application...")
    try:
        preload_data()
        logger.info("Starting Uvicorn server")
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=4000)
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        raise
