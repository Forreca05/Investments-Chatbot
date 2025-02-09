import os
import sqlite3

# Configuração do Banco de Dados SQLite
DB_PATH = "chat_history.db"

# Função de limpeza ao final da sessão
def cleanup_db():
    """Apaga o banco de dados SQLite após a sessão terminar."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def initialize_db():
    """Cria a tabela do histórico de chat no SQLite se não existir."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(role, content):
    """Salva uma nova mensagem no banco de dados SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def load_history_from_db():
    """Carrega o histórico do chat do SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history ORDER BY timestamp ASC")
    history = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return history
