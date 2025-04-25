# music_recommender_ultimate.py
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from implicit.als import AlternatingLeastSquares
import json
from scipy.sparse import csr_matrix, save_npz, load_npz
import polars as pl
from collections import defaultdict
# ======================
# КОНФИГУРАЦИЯ ПРИЛОЖЕНИЯ
# ======================
st.set_page_config(
    page_title="MusicWizard 🎧",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# ГЕНЕРАЦИЯ ДАННЫХ
# ======================
@st.cache_data
def preprocess_data(tracks):
    tracks['track_artist'] = tracks['track_name'] + " - " + tracks['artists']
    track_to_id = tracks.set_index('track_artist')['track_id'].astype(int).to_dict()
    track_id_to_info = tracks.set_index('track_id')[['track_name', 'artists']].to_dict(orient='index')
    return tracks, track_to_id, track_id_to_info

@st.cache_data
def load_data():
    # Параллельная загрузка данных
    tracks = pd.read_parquet('../data/items.parquet', engine='pyarrow')
    similar = pd.read_parquet('../data/similar.parquet', engine='pyarrow')
    # Загружаем csr_matrix из файла
    user_item_matrix_train = load_npz('../data/user_item_matrix_train.npz')
    
    
    # Оптимизация структуры данных
    similar_map = (
        similar.sort_values(by='score', ascending=False)
        .groupby('item_id')['user_id']
        .agg(lambda x: list(x)[:10])  # Предварительно сохраняем только топ-10
        .to_dict()
    )
    
    tracks, track_to_id, track_id_to_info = preprocess_data(tracks)
    tracks.set_index('track_id', inplace=True)
    return tracks, similar_map, track_to_id, track_id_to_info

@st.cache_resource
def load_model():
    als_model = AlternatingLeastSquares()
    als_model = als_model.load('../data/model.npz')
    return CatBoostClassifier().load_model('../catboost_model.cbm'), als_model

@st.cache_data(ttl=3600, show_spinner=False)
def get_instant_similar_tracks(user_input, similar_map):
    return list({
        similar_id 
        for track in user_input[-3:] 
        for similar_id in similar_map.get(track, [])[:10]
    })[:3]
def get_sexy_recommendations(history, tracks):
    
    # Загрузка маппингов
    with open('../data/user_map.json') as f:
        user_map = json.load(f)
    with open('../data/track_map.json') as f:
        track_map = json.load(f)
    als_model = AlternatingLeastSquares()
    als_model = als_model.load('../data/model.npz')
    # Поиск похожих пользователей
    similar_users = find_similar_user(
        history,
        als_model,
        user_map,
        track_map,
        n_neighbors=1
    )
    user_item_matrix_train = load_npz('../data/user_item_matrix_train.npz')

    als_score = get_recommendations_als(user_item_matrix_train, als_model, similar_users[0][0], include_seen=False, n=10)
    events = pd.DataFrame(columns=['user_id', 'track_id'])
    events['user_id'] = [similar_users[0][0]] * len(als_score)
    events['track_id'] = als_score['track_id']
    factors = add_recommendation_features(events, tracks)
    factors['als_score'] = als_score['score']
    cb_model = CatBoostClassifier().load_model('../catboost_model.cbm')
    features = ['als_score', 'user_genre_affinity', 'user_artist_ratio']
    #target = 'target'
    inference_data = Pool(data=np.array(factors[features]))
    predictions = cb_model.predict_proba(inference_data)
    factors['cb_score']=predictions[:, 1]
    threshold = factors['cb_score'].nlargest(10).min()
    top_10 = factors[factors['cb_score'] >= threshold]['track_id'].to_list()
    return top_10

def add_recommendation_features(events, items):
    """
    Добавляет 2 новых признака для рекомендательной системы:
    1. user_genre_affinity - нормированная частота прослушиваний жанра
    2. user_artist_ratio - доля прослушиваний исполнителя
    
    Параметры:
    events - DataFrame с колонками ['user_id', 'track_id']
    items - DataFrame с колонками ['track_id', 'artist', 'genre']
    
    Возвращает:
    Модифицированный DataFrame events с добавленными признаками
    """
    
    # 1. Создание быстрых lookup-таблиц
    print(items.columns)
    genre_lookup = items['genres'].to_dict()
    artist_lookup = items['track_artist'].to_dict()
    
    # 2. Инициализация структур для хранения статистик
    user_stats = {
        'genres': defaultdict(lambda: defaultdict(int)),
        'track_artist': defaultdict(lambda: defaultdict(int)),
        'total': defaultdict(int)
    }
    
    # 3. Однопроходный расчет статистик
    for _, row in events.iterrows():
        user_id = row['user_id']
        track_id = row['track_id']
        
        # Получаем жанр и исполнителя
        genre = genre_lookup.get(track_id)
        artist = artist_lookup.get(track_id)
        
        # Обновляем счетчики
        if genre:
            user_stats['genres'][user_id][genre] += 1
        if artist:
            user_stats['track_artist'][user_id][artist] += 1
        
        user_stats['total'][user_id] += 1
    
    # 4. Функции для расчета признаков
    def calc_genre_affinity(row):
        user_id = row['user_id']
        track_id = row['track_id']
        genre = genre_lookup.get(track_id)
        
        if not genre or user_stats['total'][user_id] == 0:
            return 0.0
        return user_stats['genres'][user_id][genre] / user_stats['total'][user_id]
    
    def calc_artist_ratio(row):
        user_id = row['user_id']
        track_id = row['track_id']
        artist = artist_lookup.get(track_id)
        
        if not artist or user_stats['total'][user_id] == 0:
            return 0.0
        return user_stats['track_artist'][user_id][artist] / user_stats['total'][user_id]
    
    # 5. Добавление признаков к данным
    events = events.copy()
    events['user_genre_affinity'] = events.apply(calc_genre_affinity, axis=1)
    events['user_artist_ratio'] = events.apply(calc_artist_ratio, axis=1)
    
    return events

def find_similar_user(new_user_tracks, model, user_map, track_map, n_neighbors=5):
    """
    Находит топ-N наиболее похожих пользователей по истории прослушиваний
    
    Параметры:
    new_user_tracks - список track_id нового пользователя
    model - обученная ALS модель
    user_map - словарь: оригинальный user_id -> внутренний id
    track_map - словарь: оригинальный track_id -> внутренний id
    n_neighbors - количество возвращаемых похожих пользователей
    
    Возвращает:
    Список кортежей (оригинальный user_id, оценка сходства)
    """
    # 1. Преобразование треков в внутренние индексы
    track_indices = [track_map[str(t)] for t in new_user_tracks if str(t) in track_map]
    #print(track_indices)
    if not track_indices:
        return []
    
    # 2. Получаем эмбеддинги для прослушанных треков
    item_embeddings = model.item_factors[track_indices]
    
    # 3. Агрегация эмбеддингов пользователя (взвешенное среднее)
    user_embedding = item_embeddings.mean(axis=0)
    
    # 4. Нормализация эмбеддингов
    user_embedding /= np.linalg.norm(user_embedding) + 1e-6
    
    # 5. Вычисление сходства со всеми пользователями
    similarities = model.user_factors.dot(user_embedding)
    
    # 6. Топ-N наиболее похожих пользователей
    top_indices = np.argpartition(similarities, -n_neighbors)[-n_neighbors:]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]
    
    # 7. Преобразование к оригинальным ID
    reverse_user_map = {v: k for k, v in user_map.items()}
    return [(reverse_user_map[i], similarities[i]) for i in top_indices if i in reverse_user_map]    
def get_recommendations_als(user_item_matrix, model, user_id, include_seen=True, n=5):
    """
    Возвращает отранжированные рекомендации для заданного пользователя
    """
    with open('../data/user_map.json', 'r') as f:
        user_map = json.load(f)
    with open('../data/track_map.json', 'r') as f:
        track_map = json.load(f)
    # Получаем закодированный user_id
    user_id_enc = user_map[user_id]
    
    # Создаем обратный словарь для track_map
    inv_track_map = {v: int(k) for k, v in track_map.items()}
    
    # Получаем рекомендации от модели
    recommendations = model.recommend(
        user_id_enc, 
        user_item_matrix[user_id_enc], 
        filter_already_liked_items=not include_seen,
        N=n
    )
    
    # Разбираем рекомендации на track_idx и score
    track_indices = recommendations[0]
    scores = recommendations[1]
    
    # Преобразуем track_idx в track_id с помощью inv_track_map
    track_ids = np.array([inv_track_map.get(idx, -1) for idx in track_indices])
    
    # Создаем итоговый DataFrame с помощью Polars
    recommendations_df = pl.DataFrame({
        "track_id": track_ids,
        "score": scores
    })
    
    return recommendations_df
# ======================
# РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА
# ======================
def main():
    st.title('🎵 Персональный музыкальный рекомендатор 🎵')
    st.markdown("Выберите 10 треков, которые вам нравятся, и получите персональные рекомендации!")

    # Параллельная загрузка данных и модели
    tracks, similar_map, track_to_id, track_id_to_info = load_data()
    model, als_model = load_model()

    # Сессионный state с ограничением истории
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Кэшированный список опций для multiselect
    @st.cache_data
    def get_track_options():
        return list(track_to_id.keys())

    # Виджет выбора треков
    st.subheader("1. Выберите ваши любимые треки")
    selected_tracks = st.multiselect(
        "Начните вводить название трека или исполнителя",
        options=get_track_options(),
        max_selections=10
    )

    if st.button("Сформировать рекомендации") and len(selected_tracks) >= 5:
        # Быстрое преобразование выбранных треков
        selected_ids = [track_to_id[x] for x in selected_tracks]
        
        # Ограниченная история сессии
        st.session_state.history = (st.session_state.history + selected_ids)[-100:]  # Храним только последние 100
        
        # Асинхронная обработка
        with st.spinner('Генерируем рекомендации...'):
            st.success(f"Добавлено {len(selected_ids)} треков в вашу историю!")
            
            # Разделение на колонки для параллельного отображения
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("2. Треки похожие на ваши 3 последних прослушанных")
                similar_ids = get_instant_similar_tracks(st.session_state.history[-3:], similar_map)
                top_10 = get_sexy_recommendations(st.session_state.history, tracks)
                # Мгновенный вывод через кэшированные данные
                similars  = [
                    f"- {track_id_to_info[id]['track_name']} - {track_id_to_info[id]['artists']}"
                    for id in similar_ids if id in track_id_to_info
                ]
                
                st.markdown("\n".join(similars))

                st.subheader("3. Ваши персональные рекомендации")
                model_rec = [
                    f"- {track_id_to_info[id]['track_name']} - {track_id_to_info[id]['artists']}"
                    for id in top_10 if id in track_id_to_info
                ]
                
                st.markdown("\n".join(model_rec))

if __name__ == "__main__":
    main()
