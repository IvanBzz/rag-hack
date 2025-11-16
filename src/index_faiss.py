import faiss
import numpy as np
import pandas as pd

def build_faiss_index(embeddings_path, index_path):
    embeddings = np.load(embeddings_path)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)


if __name__ == '__main__':
    build_faiss_index('data/chunk_embeddings.npy', 'data/index.faiss')
