import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "DeepPavlov/rubert-base-cased-sentence"
model = SentenceTransformer(MODEL_NAME)


def retrieve_top_k(questions_path, chunks_path, embeddings_path, index_path, k=200):
    df_q = pd.read_csv(questions_path)
    df_chunks = pd.read_csv(chunks_path)
    index = faiss.read_index(index_path)
    embeddings = np.load(embeddings_path)

    query_embeddings = model.encode(df_q['query'].tolist(), show_progress_bar=True)
    D, I = index.search(query_embeddings, k)

    results = []
    for qid, indices in zip(df_q['q_id'], I):
        for rank, idx in enumerate(indices):
            results.append({
                'q_id': qid,
                'web_id': df_chunks.iloc[idx]['web_id'],
                'rank': rank + 1
            })

    pd.DataFrame(results).to_csv('data/retrieved.csv', index=False)


if __name__ == '__main__':
    retrieve_top_k('data/questions_clean.csv', 'data/chunks.csv', 'data/chunk_embeddings.npy', 'data/index.faiss')
