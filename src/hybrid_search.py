import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = 'intfloat/multilingual-e5-large' 

class HybridRetriever:
    """
    Гибридный ретривер: комбинирует dense (embeddings) и sparse (BM25/TF-IDF) поиск.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.tfidf = None
        self.tfidf_matrix = None
        self.chunks_df = None
        self.faiss_index = None
        
    def fit(self, chunks_path: str, embeddings_path: str, index_path: str):
        """
        Инициализирует оба индекса: dense и sparse.
        
        Args:
            chunks_path: Путь к CSV с чанками
            embeddings_path: Путь к эмбеддингам
            index_path: Путь к FAISS индексу
        """
        logger.info("Загрузка данных...")
        self.chunks_df = pd.read_csv(chunks_path)
        self.chunks_df = self.chunks_df.dropna(subset=['text'])
        
        # Загружаем dense index (FAISS)
        logger.info("Загрузка FAISS индекса...")
        self.faiss_index = faiss.read_index(index_path)
        
        # Создаем sparse index (TF-IDF)
        logger.info("Создание TF-IDF индекса...")
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # uni и bi-граммы
            min_df=2,
            max_df=0.8,
            strip_accents='unicode',
            lowercase=True
        )
        chunks_text = self.chunks_df['text'].astype(str).tolist()
        self.tfidf_matrix = self.tfidf.fit_transform(chunks_text)
        
        logger.info(f"Индексы готовы. Всего чанков: {len(self.chunks_df)}")
    
    def search_dense(self, queries: List[str], k: int = 100) -> List[List[Tuple[int, float]]]:
        """Dense поиск через FAISS."""
        query_embeddings = self.model.encode(queries, show_progress_bar=False)
        distances, indices = self.faiss_index.search(query_embeddings, k)
        
        results = []
        for dist_row, idx_row in zip(distances, indices):
            results.append([(int(idx), float(1.0 / (1.0 + dist))) for idx, dist in zip(idx_row, dist_row)])
        
        return results
    
    def search_sparse(self, queries: List[str], k: int = 100) -> List[List[Tuple[int, float]]]:
        """Sparse поиск через TF-IDF."""
        query_vectors = self.tfidf.transform(queries)
        
        results = []
        for i in range(len(queries)):
            query_vec = query_vectors[i]
            
            # Косинусная близость с TF-IDF матрицей
            scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
            
            # Сортируем и берем топ-k
            top_indices = np.argsort(scores)[::-1][:k]
            top_scores = scores[top_indices]
            
            results.append([(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)])
        
        return results
    
    def hybrid_search(self, queries: List[str], k: int = 200, 
                     dense_weight: float = 0.7, sparse_weight: float = 0.3) -> pd.DataFrame:
        """
        Гибридный поиск: комбинирует dense и sparse результаты.
        
        Args:
            queries: Список запросов
            k: Количество результатов для получения от каждого метода
            dense_weight: Вес dense поиска
            sparse_weight: Вес sparse поиска
            
        Returns:
            DataFrame с результатами (q_id, web_id, rank, dense_score, sparse_score, combined_score)
        """
        logger.info(f"Гибридный поиск для {len(queries)} запросов...")
        logger.info(f"Веса: dense={dense_weight}, sparse={sparse_weight}")
        
        # Получаем результаты от обоих методов
        dense_results = self.search_dense(queries, k=k)
        sparse_results = self.search_sparse(queries, k=k)
        
        all_results = []
        
        for q_idx, (query, dense_res, sparse_res) in enumerate(zip(queries, dense_results, sparse_results)):
            # Собираем все уникальные индексы чанков
            all_chunk_indices = set()
            dense_scores = {}
            sparse_scores = {}
            
            for chunk_idx, score in dense_res:
                all_chunk_indices.add(chunk_idx)
                dense_scores[chunk_idx] = score
            
            for chunk_idx, score in sparse_res:
                all_chunk_indices.add(chunk_idx)
                sparse_scores[chunk_idx] = score
            
            # Нормализуем и комбинируем скоры
            chunk_scores = []
            for chunk_idx in all_chunk_indices:
                d_score = dense_scores.get(chunk_idx, 0.0)
                s_score = sparse_scores.get(chunk_idx, 0.0)
                
                # Комбинированный скор
                combined = dense_weight * d_score + sparse_weight * s_score
                
                web_id = self.chunks_df.iloc[chunk_idx]['web_id']
                
                chunk_scores.append({
                    'q_id': q_idx + 1,
                    'web_id': web_id,
                    'chunk_idx': chunk_idx,
                    'dense_score': d_score,
                    'sparse_score': s_score,
                    'combined_score': combined
                })
            
            # Сортируем по комбинированному скору
            chunk_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Добавляем ранг
            for rank, item in enumerate(chunk_scores, start=1):
                item['rank'] = rank
                all_results.append(item)
        
        results_df = pd.DataFrame(all_results)
        
        # Дедупликация web_id для каждого запроса (оставляем лучший)
        results_df = results_df.sort_values(['q_id', 'combined_score'], ascending=[True, False])
        results_df = results_df.drop_duplicates(subset=['q_id', 'web_id'], keep='first')
        
        # Пересчитываем ранги после дедупликации
        results_df['rank'] = results_df.groupby('q_id').cumcount() + 1
        
        logger.info(f"Найдено {len(results_df)} уникальных результатов")
        
        return results_df[['q_id', 'web_id', 'rank', 'dense_score', 'sparse_score', 'combined_score']]


def retrieve_hybrid(questions_path: str, chunks_path: str, embeddings_path: str, 
                   index_path: str, output_path: str, k: int = 200,
                   dense_weight: float = 0.7, sparse_weight: float = 0.3):
    """
    Выполняет гибридный поиск и сохраняет результаты.
    
    Args:
        questions_path: Путь к вопросам
        chunks_path: Путь к чанкам
        embeddings_path: Путь к эмбеддингам
        index_path: Путь к FAISS индексу
        output_path: Путь для сохранения результатов
        k: Количество результатов
        dense_weight: Вес dense поиска
        sparse_weight: Вес sparse поиска
    """
    # Загружаем вопросы
    df_q = pd.read_csv(questions_path)
    queries = df_q['query'].tolist()
    
    # Инициализируем ретривер
    retriever = HybridRetriever()
    retriever.fit(chunks_path, embeddings_path, index_path)
    
    # Выполняем поиск
    results_df = retriever.hybrid_search(queries, k=k, 
                                        dense_weight=dense_weight, 
                                        sparse_weight=sparse_weight)
    
    # Сохраняем
    results_df.to_csv(output_path, index=False)
    logger.info(f"Результаты сохранены в {output_path}")
    
    # Статистика
    logger.info(f"\nСтатистика:")
    logger.info(f"  Запросов: {df_q['q_id'].nunique()}")
    logger.info(f"  Результатов: {len(results_df)}")
    logger.info(f"  Среднее на запрос: {len(results_df) / df_q['q_id'].nunique():.2f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Гибридный поиск (dense + sparse)')
    parser.add_argument('--questions', default='data/questions_clean.csv')
    parser.add_argument('--chunks', default='data/clean_answer_data.csv')
    parser.add_argument('--embeddings', default='data/chunk_embeddings.npy')
    parser.add_argument('--index', default='data/index.faiss')
    parser.add_argument('--output', default='data/retrieved.csv')
    parser.add_argument('--k', type=int, default=200, help='Результатов на запрос')
    parser.add_argument('--dense-weight', type=float, default=0.7, help='Вес dense поиска')
    parser.add_argument('--sparse-weight', type=float, default=0.3, help='Вес sparse поиска')
    
    args = parser.parse_args()
    
    retrieve_hybrid(
        questions_path=args.questions,
        chunks_path=args.chunks,
        embeddings_path=args.embeddings,
        index_path=args.index,
        output_path=args.output,
        k=args.k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight
    )
