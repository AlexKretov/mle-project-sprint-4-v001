import logging
from datetime import datetime
import requests
import random
import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_service.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ServiceTester')

# Конфигурация теста (без изменений)
TEST_CONFIG = {
    'base_url': 'http://127.0.0.1:4000',
    'num_requests': 200,
    'user_id_range': (1, 10768620),
    'timeout': 5
}

def test_service(config):
    # Инициализация лога теста
    logger.info(f"\n{'#'*40}")
    logger.info(f"Starting new test session at {datetime.now()}")
    logger.info(f"Test configuration: {config}")
    
    stats = defaultdict(int)
    response_times = []
    track_counts = []
    stat_df = pd.DataFrame(columns=['user_id','personal','similars','popular'])
    
    for _ in tqdm(range(config['num_requests']), desc='Testing'):
        user_id = random.randint(*config['user_id_range'])
        url = f"{config['base_url']}/get_playlist?user_id={user_id}"
        
        # Логирование деталей запроса
        logger.debug(f"Requesting user_id: {user_id} | URL: {url}")
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=config['timeout'])
            elapsed = time.time() - start_time
            response_times.append(elapsed)
            
            if response.status_code == 200:
                stats['success'] += 1
                data = response.json()
                track_counts.append(len(data.get('playlist', [])))
                st = data.get('stats', {})
                stat_df.loc[len(stat_df)] = [user_id, st['personal'], st['similars'], st['popular']]
                
                # Детальное логирование успешных ответов
                logger.info(f"SUCCESS: user_id={user_id} | "
                          f"Tracks={len(data.get('playlist', []))} | "
                          f"Response_time={elapsed:.3f}s")
                
            else:
                stats[f'error_{response.status_code}'] += 1
                logger.error(f"HTTP ERROR {response.status_code}: user_id={user_id} | "
                            f"Response: {response.text[:200]}...")
                
        except Exception as e:
            elapsed = time.time() - start_time
            stats['exceptions'] += 1
            stats[f'exception_{type(e).__name__}'] += 1
            logger.exception(f"EXCEPTION: {type(e).__name__} occurred for user_id={user_id} | "
                           f"Error: {str(e)[:200]}...")
        finally:
            response_times.append(elapsed)
    
    # Логирование финальной статистики
    stats['avg_response_time'] = sum(response_times) / len(response_times)
    stats['min_response_time'] = min(response_times)
    stats['max_response_time'] = max(response_times)
    stats['avg_tracks_per_response'] = sum(track_counts) / len(track_counts) if track_counts else 0
    
    logger.info(f"\nTest Session Statistics:")
    logger.info(f"Total requests: {config['num_requests']}")
    logger.info(f"Success rate: {stats['success'] / config['num_requests'] * 100:.2f}%")
    logger.info(f"Response times - Avg: {stats['avg_response_time']:.3f}s | "
              f"Min: {stats['min_response_time']:.3f}s | Max: {stats['max_response_time']:.3f}s")
    logger.info(f"Average tracks per response: {stats['avg_tracks_per_response']:.1f}")
    
    # Сохранение сырых данных
    stat_df.to_csv('test_results.csv', index=False)
    logger.info(f"Raw test data saved to test_results.csv")
    
    return stats, stat_df

def print_stats(stats):
    print("\nTest Results:")
    print(f"Total requests: {TEST_CONFIG['num_requests']}")
    print(f"Success rate: {stats['success'] / TEST_CONFIG['num_requests'] * 100:.2f}%")
    print(f"Average response time: {stats['avg_response_time']:.3f}s")
    print(f"Min/Max response time: {stats['min_response_time']:.3f}s / {stats['max_response_time']:.3f}s")
    print(f"Average tracks per response: {stats['avg_tracks_per_response']:.1f}")
    
    print("\nErrors and Exceptions:")
    for key, value in stats.items():
        if key.startswith('error_') or key.startswith('exception_'):
            print(f"{key}: {value}")

if __name__ == "__main__":
    test_stats, df = test_service(TEST_CONFIG)
    print_stats(test_stats)
    print(df.describe())
