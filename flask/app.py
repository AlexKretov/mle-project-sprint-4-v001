from flask import Flask, request, render_template, jsonify, session
import psycopg2
import requests
from dotenv import load_dotenv
import os
import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
load_dotenv('flask\.env')
def get_connection():
    return psycopg2.connect(
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )

def perform_search(query):
    query = query.lower()
    # Connect to the database
    conn = get_connection()
    c = conn.cursor()

    # Execute the search query
    c.execute(f"SELECT * FROM tracks WHERE LOWER(track_name) LIKE '{'%'+query+'%'}' OR LOWER(artists) LIKE '{'%'+query+'%'}'")
    results = c.fetchall()

    # Close the connection
    conn.close()

    return results
@app.route('/add_to_selection', methods=['POST'])
def add_to_selection():
    data = request.json
    track_data = {
        'track_id': data['track_id'],
        'track_name': data['track_name'],
        'artists': data['artists']
    }
    
    # Инициализируем список выбранных треков в сессии
    if 'selected_tracks' not in session:
        session['selected_tracks'] = []
    
    # Проверяем, не добавлен ли трек уже
    if not any(t['track_id'] == track_data['track_id'] for t in session['selected_tracks']):
        session['selected_tracks'].append(track_data)
        session.modified = True  # Явно указываем на изменение сессии
    
    return jsonify({
        'status': 'success',
        'selected_count': len(session['selected_tracks'])
    })
@app.route('/get_selection')
def get_selection():
    return jsonify(session.get('selected_tracks', []))
@app.route('/process', methods=['POST'])
def process_selected_tracks():
    try:
        selected_tracks = session.get('selected_tracks', [])
        track_ids = [t['track_id'] for t in selected_tracks if 'track_id' in t]

        # Запрос к микросервису
        try:
            response = requests.post(
                "http://127.0.0.1:8000/test",
                json={"track_ids": track_ids},
                timeout=10
            )
            response.raise_for_status()
            recommended_data = response.json() if response.headers.get('Content-Type') == 'application/json' else {"track_ids": []}
        
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Ошибка соединения: {str(e)}")
            return jsonify({'status': 'error', 'message': 'Сервис недоступен'}), 503

        # Безопасный запрос к БД
        conn = get_connection()
        c = conn.cursor()
        try:
            c.execute(
                "SELECT track_name, artists FROM tracks WHERE track_id = ANY(%s)",
                (recommended_data['track_ids'],)
            )
            results = [[row[0], row[1]] for row in c.fetchall()]
        finally:
            conn.close()

        return jsonify({
            'status': 'success',
            'processed_tracks': len(track_ids),
            'recommendations': results  # Данные в формате [[name, artist], ...]
        })

    except Exception as e:
        app.logger.error(f"Ошибка обработки: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Внутренняя ошибка сервера'}), 500

@app.route('/remove_from_selection', methods=['POST'])
def remove_from_selection():
    track_id = request.json['track_id']
    
    if 'selected_tracks' in session:
        session['selected_tracks'] = [t for t in session['selected_tracks'] if t['track_id'] != track_id]
        session.modified = True
    
    return jsonify({
        'status': 'success',
        'selected_count': len(session.get('selected_tracks', []))
    })
@app.route('/')
def search_page():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    conn = get_connection()
    c = conn.cursor()
    
    search_pattern = f"%{query.lower()}%"
    c.execute("""
        SELECT track_id, track_name, artists 
        FROM tracks 
        WHERE LOWER(track_name) LIKE %s 
        OR LOWER(artists) LIKE %s
    """, (search_pattern, search_pattern))
    
    columns = [desc[0] for desc in c.description]
    results = [dict(zip(columns, row)) for row in c.fetchall()]
    
    # Добавляем информацию о выборе
    selected_ids = {t['track_id'] for t in session.get('selected_tracks', [])}
    for track in results:
        track['is_selected'] = track['track_id'] in selected_ids
    
    conn.close()
    return jsonify(results)

if __name__ == '__main__':
    app.run()