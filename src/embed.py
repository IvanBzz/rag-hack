from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


MODEL = 'intfloat/multilingual-e5-large'  # Хорошая мультиязычная модель

# Инициализация модели
model = SentenceTransformer(MODEL)

def embed_chunks(input_path, output_path):
    df = pd.read_csv(input_path)
    if 'text' not in df.columns:
        raise ValueError('Входной файл не содержит столбца "chunk".')
    df = df.dropna(subset=['text'])
    if df.empty:
        raise ValueError("В файле с чанками нет доступных текстов для эмбеддинга.")
    chunks = df['text'].astype(str).str.strip()
    chunks = chunks[chunks != ""]
    if chunks.empty:
        raise ValueError("Все чанки пусты после очистки.")
    embeddings = model.encode(chunks.tolist(), show_progress_bar=True)
    np.save(output_path, embeddings)


if __name__ == '__main__':
    embed_chunks('data/chunks.csv', 'data/chunk_embeddings.npy')
