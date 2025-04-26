from flask import Flask, request, render_template, jsonify
import psycopg2
from dotenv import load_dotenv
import os
import hashlib
app = Flask(__name__)
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

@app.route('/')
def search_page():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    conn = get_connection()
    c = conn.cursor()
    
    # Используем параметризованный запрос для безопасности
    search_pattern = f"%{query.lower()}%"
    c.execute("""
        SELECT track_id, track_name, artists 
        FROM tracks 
        WHERE LOWER(track_name) LIKE %s 
        OR LOWER(artists) LIKE %s
    """, (search_pattern, search_pattern))
    
    # Конвертируем результат в список словарей
    columns = [desc[0] for desc in c.description]
    results = [dict(zip(columns, row)) for row in c.fetchall()]
    
    conn.close()
    return jsonify(results)  # Возвращаем данные в JSON формате

if __name__ == '__main__':
    app.run()