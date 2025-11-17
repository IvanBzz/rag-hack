"""
Полный интегрированный RAG Pipeline со всеми улучшениями
Запуск: python complete_pipeline.py --mode full
"""
import os
import argparse
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: УЛУЧШЕННЫЙ ЧАНКИНГ С ПЕРЕКРЫТИЕМ
# ============================================================================

class ImprovedChunker:
    """Чанкинг с перекрытием для сохранения контекста"""
    
    def __init__(self, max_length: int = 500, overlap: int = 100):
        self.max_length = max_length
        self.overlap = overlap
        self._ensure_punkt()
    
    def _ensure_punkt(self):
        """Гарантируем наличие punkt tokenizer"""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("Downloading punkt tokenizer...")
            nltk.download("punkt", quiet=True)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Разбивает текст на чанки с перекрытием
        
        Args:
            text: Исходный текст
            
        Returns:
            Список чанков
        """
        if not isinstance(text, str):
            if pd.isna(text):
                return []
            text = str(text)
        
        text = text.strip()
        if not text:
            return []
        
        sentences = sent_tokenize(text, language='russian')
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(sent)
            
            if current_len + sent_len <= self.max_length:
                current_chunk.append(sent)
                current_len += sent_len
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Добавляем перекрытие из предыдущего чанка
                overlap_text = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_text.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                
                current_chunk = overlap_text + [sent]
                current_len = sum(len(s) for s in current_chunk)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_chunks_with_metadata(self, input_path: str, output_path: str):
        """
        Создает чанки для low_filtered.csv формата
        
        Args:
            input_path: Путь к low_filtered.csv
            output_path: Путь для сохранения чанков
        """
        logger.info(f"Создание улучшенных чанков из {input_path}...")
        df = pd.read_csv(input_path)
        
        data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
            chunks = self.chunk_text(row['text'])
            
            for chunk in chunks:
                chunk_data = {
                    'web_id': row['web_id'],
                    'chunk': chunk,
                    'enriched_chunk': chunk  # Для low_filtered.csv используем просто chunk
                }
                
                data.append(chunk_data)
        
        result_df = pd.DataFrame(data)
        result_df.to_csv(output_path, index=False)
        logger.info(f"Создано {len(result_df)} чанков → {output_path}")
        
        return result_df


# ============================================================================
# PART 2: QUERY EXPANSION
# ============================================================================

class QueryExpander:
    """Расширение запросов для более широкого поиска"""
    
    def __init__(self):
        self.stop_words = {
            'и', 'в', 'на', 'с', 'по', 'для', 'как', 'что', 'это', 'а', 'о',
            'к', 'от', 'из', 'у', 'за', 'до', 'при', 'не', 'же', 'бы', 'ли',
            'или', 'но', 'если', 'когда', 'то', 'так', 'только', 'уже'
        }
    
    def expand_query(self, query: str) -> List[str]:
        """
        Создает вариации запроса
        
        Args:
            query: Исходный запрос
            
        Returns:
            Список вариаций запроса
        """
        expanded = [query]
        
        # Извлекаем ключевые термины
        key_terms = self._extract_key_terms(query)
        if key_terms and key_terms != query.lower().split():
            expanded.append(' '.join(key_terms))
        
        # Проверяем наличие вопросительных слов
        question_words = ['как', 'что', 'где', 'когда', 'почему', 'какой', 'сколько']
        has_question = any(word in query.lower() for word in question_words)
        
        if not has_question and len(query.split()) <= 5:
            # Добавляем вопросительные формы
            expanded.append(f"Как {query}?")
            expanded.append(f"Что такое {query}?")
        
        return expanded
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Извлекает ключевые термины"""
        words = text.lower().split()
        key_terms = [w for w in words if w not in self.stop_words and len(w) > 2]
        return key_terms


# ============================================================================
# PART 3: ГИБРИДНЫЙ РЕТРИВЕР С RRF
# ============================================================================

class HybridRetriever:
    """
    Гибридный ретривер с:
    - Dense search (embeddings)
    - Sparse search (TF-IDF/BM25)
    - Reciprocal Rank Fusion
    - Query Expansion
    """
    
    def __init__(self, 
                 embedding_model: str = 'intfloat/multilingual-e5-large',
                 use_query_expansion: bool = True):
        """
        Args:
            embedding_model: Модель для эмбеддингов
            use_query_expansion: Использовать ли расширение запросов
        """
        logger.info(f"Инициализация HybridRetriever...")
        logger.info(f"Модель эмбеддингов: {embedding_model}")
        
        self.embedding_model_name = embedding_model
        self.embed_model = SentenceTransformer(embedding_model)
        self.use_query_expansion = use_query_expansion
        
        if use_query_expansion:
            self.query_expander = QueryExpander()
        
        self.tfidf = None
        self.tfidf_matrix = None
        self.chunks_df = None
        self.faiss_index = None
        self.use_enriched = False
    
    def _add_e5_prefix(self, text: str, is_query: bool = False) -> str:
        """Добавляет префиксы для E5 моделей"""
        if 'e5' in self.embedding_model_name.lower():
            prefix = "query: " if is_query else "passage: "
            return prefix + text
        return text
    
    def fit(self, chunks_path: str, embeddings_path: Optional[str] = None,
            index_path: Optional[str] = None, recreate_index: bool = False):
        """
        Инициализация индексов
        
        Args:
            chunks_path: Путь к CSV с чанками
            embeddings_path: Путь к сохраненным эмбеддингам
            index_path: Путь к FAISS индексу
            recreate_index: Пересоздать индекс с нуля
        """
        logger.info("="*60)
        logger.info("Загрузка и индексация данных")
        logger.info("="*60)
        
        # Загружаем чанки
        logger.info(f"Загрузка чанков из {chunks_path}...")
        self.chunks_df = pd.read_csv(chunks_path)
        self.chunks_df = self.chunks_df.dropna(subset=['chunk'])
        
        # Проверяем наличие обогащенных чанков
        if 'enriched_chunk' in self.chunks_df.columns:
            self.use_enriched = True
            logger.info("Используем обогащенные чанки")
            chunk_column = 'enriched_chunk'
        else:
            chunk_column = 'chunk'
        
        chunks_text = self.chunks_df[chunk_column].astype(str).tolist()
        logger.info(f"Загружено чанков: {len(chunks_text)}")
        
        # Dense index (FAISS)
        if recreate_index or embeddings_path is None or not os.path.exists(embeddings_path):
            logger.info("Создание эмбеддингов с префиксами E5...")
            prefixed_chunks = [self._add_e5_prefix(text, is_query=False) for text in chunks_text]
            
            embeddings = self.embed_model.encode(
                prefixed_chunks,
                show_progress_bar=True,
                batch_size=32,
                normalize_embeddings=True  # Нормализация для косинусной близости
            )
            
            logger.info("Создание FAISS индекса...")
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Сохраняем
            if embeddings_path:
                np.save(embeddings_path, embeddings)
                logger.info(f"Эмбеддинги сохранены: {embeddings_path}")
            if index_path:
                faiss.write_index(self.faiss_index, index_path)
                logger.info(f"FAISS индекс сохранен: {index_path}")
        else:
            logger.info(f"Загрузка существующего FAISS индекса: {index_path}")
            self.faiss_index = faiss.read_index(index_path)
        
        # Sparse index (TF-IDF)
        logger.info("Создание TF-IDF индекса...")
        self.tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.85,
            strip_accents='unicode',
            lowercase=True,
            analyzer='word'
        )
        self.tfidf_matrix = self.tfidf.fit_transform(chunks_text)
        
        logger.info("="*60)
        logger.info("Индексация завершена")
        logger.info(f"  Чанков: {len(self.chunks_df)}")
        logger.info(f"  Размерность эмбеддингов: {self.faiss_index.d}")
        logger.info(f"  TF-IDF features: {len(self.tfidf.get_feature_names_out())}")
        logger.info("="*60)
    
    def search_dense(self, queries: List[str], k: int = 100) -> List[List[Tuple[int, float]]]:
        """Dense поиск через FAISS"""
        # Добавляем префиксы E5
        prefixed_queries = [self._add_e5_prefix(q, is_query=True) for q in queries]
        
        query_embeddings = self.embed_model.encode(
            prefixed_queries,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        scores, indices = self.faiss_index.search(query_embeddings.astype('float32'), k)
        
        results = []
        for score_row, idx_row in zip(scores, indices):
            results.append([(int(idx), float(score)) for idx, score in zip(idx_row, score_row)])
        
        return results
    
    def search_sparse(self, queries: List[str], k: int = 100) -> List[List[Tuple[int, float]]]:
        """Sparse поиск через TF-IDF"""
        query_vectors = self.tfidf.transform(queries)
        
        results = []
        for i in range(len(queries)):
            query_vec = query_vectors[i]
            scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
            top_indices = np.argsort(scores)[::-1][:k]
            top_scores = scores[top_indices]
            results.append([(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)])
        
        return results
    
    def reciprocal_rank_fusion(self, rankings: List[List[Tuple[int, float]]], 
                                k: int = 60) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion для комбинирования результатов
        
        Args:
            rankings: Список ранжирований от разных методов
            k: Параметр RRF (обычно 60)
            
        Returns:
            Объединенное ранжирование
        """
        rrf_scores = {}
        
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank)
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs
    
    def hybrid_search(self, queries: List[str], k_retrieve: int = 300,
                     k_final: int = 100) -> pd.DataFrame:
        """
        Гибридный поиск с RRF и query expansion
        
        Args:
            queries: Список запросов
            k_retrieve: Количество результатов от каждого метода
            k_final: Финальное количество результатов
            
        Returns:
            DataFrame с результатами
        """
        logger.info(f"Гибридный поиск для {len(queries)} запросов...")
        
        all_results = []
        
        for q_idx, query in enumerate(tqdm(queries, desc="Hybrid search")):
            # Query expansion
            if self.use_query_expansion:
                expanded_queries = self.query_expander.expand_query(query)
            else:
                expanded_queries = [query]
            
            # Собираем результаты от всех вариаций запроса
            all_rankings = []
            
            for exp_query in expanded_queries:
                dense_res = self.search_dense([exp_query], k=k_retrieve)[0]
                sparse_res = self.search_sparse([exp_query], k=k_retrieve)[0]
                all_rankings.extend([dense_res, sparse_res])
            
            # RRF комбинирование
            combined = self.reciprocal_rank_fusion(all_rankings)
            
            # Берем топ результатов
            for chunk_idx, rrf_score in combined[:k_final]:
                chunk_idx = int(chunk_idx)
                if chunk_idx >= len(self.chunks_df):
                    continue
                
                web_id = self.chunks_df.iloc[chunk_idx]['web_id']
                
                all_results.append({
                    'q_id': q_idx + 1,
                    'web_id': web_id,
                    'chunk_idx': chunk_idx,
                    'rrf_score': rrf_score
                })
        
        results_df = pd.DataFrame(all_results)
        
        # Дедупликация по web_id для каждого запроса
        results_df = results_df.sort_values(['q_id', 'rrf_score'], ascending=[True, False])
        results_df = results_df.drop_duplicates(subset=['q_id', 'web_id'], keep='first')
        
        logger.info(f"Найдено уникальных результатов: {len(results_df)}")
        
        return results_df


# ============================================================================
# PART 4: RERANKER
# ============================================================================

class Reranker:
    """Cross-encoder для переранжировки"""
    
    def __init__(self, model_name: str = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
                 batch_size: int = 32):
        """
        Args:
            model_name: Модель cross-encoder
            batch_size: Размер батча
        """
        logger.info(f"Загрузка reranking модели: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        self.batch_size = batch_size
    
    def rerank(self, queries: List[str], candidates_df: pd.DataFrame,
               chunks_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """
        Переранжировка кандидатов
        
        Args:
            queries: Список запросов
            candidates_df: DataFrame с кандидатами (должен содержать q_id, web_id, chunk_idx)
            chunks_df: DataFrame с чанками
            top_k: Количество финальных результатов
            
        Returns:
            DataFrame с переранжированными результатами
        """
        logger.info(f"Переранжировка кандидатов...")
        
        reranked = []
        
        for q_id in tqdm(candidates_df['q_id'].unique(), desc="Reranking"):
            query = queries[q_id - 1]
            query_candidates = candidates_df[candidates_df['q_id'] == q_id].copy()
            
            # Получаем тексты чанков
            chunk_texts = []
            valid_indices = []
            
            for idx, row in query_candidates.iterrows():
                chunk_idx = int(row['chunk_idx'])
                if chunk_idx < len(chunks_df):
                    chunk_texts.append(chunks_df.iloc[chunk_idx]['chunk'])
                    valid_indices.append(idx)
            
            if not chunk_texts:
                continue
            
            # Создаем пары (запрос, документ)
            pairs = [(query, text) for text in chunk_texts]
            
            # Получаем скоры
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Добавляем скоры
            for idx, score in zip(valid_indices, scores):
                query_candidates.loc[idx, 'rerank_score'] = float(score)
            
            # Сортируем и берем топ-k
            query_candidates = query_candidates.sort_values('rerank_score', ascending=False).head(top_k)
            query_candidates['rank'] = range(1, len(query_candidates) + 1)
            
            reranked.append(query_candidates)
        
        result_df = pd.concat(reranked, ignore_index=True)
        
        logger.info(f"Переранжировка завершена: {len(result_df)} результатов")
        
        return result_df


# ============================================================================
# PART 5: ПОСТОБРАБОТКА
# ============================================================================

class PostProcessor:
    """Постобработка результатов"""
    
    def __init__(self, diversity_weight: float = 0.2):
        """
        Args:
            diversity_weight: Вес штрафа за повторение
        """
        self.diversity_weight = diversity_weight
    
    def process(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Постобработка результатов
        
        Args:
            results_df: Результаты поиска
            
        Returns:
            Обработанные результаты
        """
        logger.info("Постобработка результатов...")
        
        # Для low_filtered.csv просто возвращаем результаты
        # без разнообразия по типам контента
        results_df['final_score'] = results_df.get('rerank_score', results_df.get('rrf_score', 0))
        results_df = results_df.sort_values(['q_id', 'final_score'], ascending=[True, False])
        
        # Нумеруем ранги
        results_df['rank'] = results_df.groupby('q_id').cumcount() + 1
        
        logger.info(f"После постобработки: {len(results_df)} результатов")
        
        return results_df


# ============================================================================
# PART 6: ПОЛНЫЙ ПАЙПЛАЙН
# ============================================================================

class CompletePipeline:
    """Полный RAG пайплайн со всеми улучшениями"""
    
    def __init__(self,
                 embedding_model: str = 'intfloat/multilingual-e5-large',
                 rerank_model: str = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
                 chunk_max_length: int = 500,
                 chunk_overlap: int = 100,
                 use_query_expansion: bool = True,
                 diversity_weight: float = 0.2):
        """
        Инициализация полного пайплайна
        
        Args:
            embedding_model: Модель для эмбеддингов
            rerank_model: Модель для переранжировки
            chunk_max_length: Максимальная длина чанка
            chunk_overlap: Перекрытие чанков
            use_query_expansion: Использовать расширение запросов
            diversity_weight: Вес для разнообразия результатов
        """
        self.chunker = ImprovedChunker(chunk_max_length, chunk_overlap)
        self.retriever = HybridRetriever(embedding_model, use_query_expansion)
        self.reranker = Reranker(rerank_model)
        self.postprocessor = PostProcessor(diversity_weight)
        
        self.chunks_df = None
    
    def prepare_data(self, 
                    websites_path: str,
                    chunks_output_path: str,
                    recreate_chunks: bool = False):
        """
        Подготовка данных: чанкинг
        
        Args:
            websites_path: Путь к low_filtered.csv
            chunks_output_path: Путь для сохранения чанков
            recreate_chunks: Пересоздать чанки
        """
        logger.info("="*60)
        logger.info("ПОДГОТОВКА ДАННЫХ")
        logger.info("="*60)
        
        # Создаем/загружаем чанки
        if recreate_chunks or not os.path.exists(chunks_output_path):
            self.chunks_df = self.chunker.create_chunks_with_metadata(
                websites_path,
                chunks_output_path
            )
        else:
            logger.info(f"Загрузка существующих чанков: {chunks_output_path}")
            self.chunks_df = pd.read_csv(chunks_output_path)
        
        logger.info(f"Всего чанков: {len(self.chunks_df)}")
    
    def build_index(self,
                   chunks_path: str,
                   embeddings_path: str,
                   index_path: str,
                   recreate_index: bool = False):
        """
        Построение индексов
        
        Args:
            chunks_path: Путь к чанкам
            embeddings_path: Путь для эмбеддингов
            index_path: Путь для FAISS индекса
            recreate_index: Пересоздать индекс
        """
        logger.info("="*60)
        logger.info("ПОСТРОЕНИЕ ИНДЕКСОВ")
        logger.info("="*60)
        
        self.retriever.fit(
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            index_path=index_path,
            recreate_index=recreate_index
        )
        
        if self.chunks_df is None:
            self.chunks_df = pd.read_csv(chunks_path)
    
    def search(self,
              queries: List[str],
              k_retrieve: int = 300,
              k_rerank: int = 50,
              k_final: int = 5,
              use_reranking: bool = True,
              use_postprocessing: bool = True) -> pd.DataFrame:
        """
        Полный поиск
        
        Args:
            queries: Список запросов
            k_retrieve: Количество кандидатов на первом этапе
            k_rerank: Количество кандидатов для переранжировки
            k_final: Финальное количество результатов
            use_reranking: Использовать переранжировку
            use_postprocessing: Использовать постобработку
            
        Returns:
            DataFrame с результатами
        """
        logger.info("="*60)
        logger.info("НАЧАЛО ПОИСКА")
        logger.info("="*60)
        logger.info(f"Запросов: {len(queries)}")
        logger.info(f"k_retrieve: {k_retrieve}")
        logger.info(f"k_rerank: {k_rerank}")
        logger.info(f"k_final: {k_final}")
        logger.info(f"Reranking: {use_reranking}")
        logger.info(f"Postprocessing: {use_postprocessing}")
        logger.info("="*60)
        
        # Этап 1: Гибридный поиск
        logger.info("\n[1/4] Гибридный поиск (Dense + Sparse + RRF)...")
        candidates = self.retriever.hybrid_search(
            queries=queries,
            k_retrieve=k_retrieve,
            k_final=k_rerank if use_reranking else k_final
        )
        
        logger.info(f"  Найдено кандидатов: {len(candidates)}")
        logger.info(f"  Среднее на запрос: {len(candidates) / len(queries):.2f}")
        
        results = candidates
        
        # Этап 2: Переранжировка
        if use_reranking:
            logger.info("\n[2/4] Переранжировка (Cross-Encoder)...")
            results = self.reranker.rerank(
                queries=queries,
                candidates_df=candidates,
                chunks_df=self.chunks_df,
                top_k=k_final
            )
            logger.info(f"  После reranking: {len(results)} результатов")
        else:
            logger.info("\n[2/4] Переранжировка пропущена")
            results = results.groupby('q_id').head(k_final).copy()
        
        # Этап 3: Постобработка
        if use_postprocessing:
            logger.info("\n[3/4] Постобработка...")
            results = self.postprocessor.process(results)
            logger.info(f"  После постобработки: {len(results)} результатов")
        else:
            logger.info("\n[3/4] Постобработка пропущена")
        
        # Этап 4: Финализация
        logger.info("\n[4/4] Финализация результатов...")
        results = results.sort_values(['q_id', 'rank'])
        
        logger.info("="*60)
        logger.info("ПОИСК ЗАВЕРШЕН")
        logger.info("="*60)
        logger.info(f"Итоговых результатов: {len(results)}")
        logger.info(f"Результатов на запрос: {len(results) / len(queries):.2f}")
        
        return results
    
    def create_submission(self, results_df: pd.DataFrame, output_path: str):
        """
        Создание submission файла
        
        Args:
            results_df: Результаты поиска
            output_path: Путь для сохранения
        """
        logger.info(f"\nСоздание submission файла: {output_path}")
        
        # Группируем по q_id и берем топ-5 web_id
        submission = results_df.sort_values(['q_id', 'rank']).groupby('q_id')['web_id'].apply(
            lambda x: list(x.head(5))
        ).reset_index()
        
        submission.columns = ['q_id', 'web_list']
        submission['web_list'] = submission['web_list'].apply(str)
        
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission сохранен: {output_path}")
        logger.info(f"  Запросов в submission: {len(submission)}")


# ============================================================================
# PART 7: ЭКСПЕРИМЕНТЫ И ОПТИМИЗАЦИЯ
# ============================================================================

class Experimenter:
    """Класс для запуска экспериментов с разными параметрами"""
    
    def __init__(self, pipeline: CompletePipeline):
        self.pipeline = pipeline
    
    def test_embedding_models(self, queries: List[str], models: List[str],
                             output_dir: str = 'experiments/models'):
        """
        Тестирование разных моделей эмбеддингов
        
        Args:
            queries: Список запросов
            models: Список моделей для тестирования
            output_dir: Директория для сохранения результатов
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("="*60)
        logger.info(f"ТЕСТИРОВАНИЕ {len(models)} МОДЕЛЕЙ")
        logger.info("="*60)
        
        results = {}
        
        for model_name in models:
            logger.info(f"\nТестирование модели: {model_name}")
            
            # Создаем новый ретривер с этой моделью
            self.pipeline.retriever = HybridRetriever(
                embedding_model=model_name,
                use_query_expansion=True
            )
            
            # Переиндексируем данные
            self.pipeline.retriever.fit(
                chunks_path='data/chunks_improved.csv',
                embeddings_path=f'{output_dir}/embeddings_{model_name.replace("/", "_")}.npy',
                index_path=f'{output_dir}/index_{model_name.replace("/", "_")}.faiss',
                recreate_index=True
            )
            
            # Запускаем поиск
            search_results = self.pipeline.search(queries)
            
            # Сохраняем результаты
            output_path = f'{output_dir}/submission_{model_name.replace("/", "_")}.csv'
            self.pipeline.create_submission(search_results, output_path)
            
            results[model_name] = output_path
        
        logger.info("\n" + "="*60)
        logger.info("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        logger.info("="*60)
        for model, path in results.items():
            logger.info(f"  {model}: {path}")
    
    def grid_search(self, queries: List[str], param_grid: Dict,
                   output_dir: str = 'experiments/grid_search'):
        """
        Grid search по параметрам
        
        Args:
            queries: Список запросов
            param_grid: Словарь с параметрами
            output_dir: Директория для результатов
        """
        from itertools import product
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("="*60)
        logger.info("GRID SEARCH")
        logger.info("="*60)
        logger.info(f"Параметры: {param_grid}")
        
        # Генерируем все комбинации
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        logger.info(f"Всего комбинаций: {len(combinations)}")
        
        results = []
        
        for idx, combination in enumerate(combinations, 1):
            params = dict(zip(keys, combination))
            logger.info(f"\n[{idx}/{len(combinations)}] Тестирование: {params}")
            
            # Запускаем поиск с этими параметрами
            search_results = self.pipeline.search(queries, **params)
            
            # Сохраняем
            output_path = f'{output_dir}/submission_{idx}.csv'
            self.pipeline.create_submission(search_results, output_path)
            
            results.append({
                'combination': idx,
                'params': params,
                'output': output_path,
                'num_results': len(search_results)
            })
        
        # Сохраняем метаданные экспериментов
        meta_df = pd.DataFrame(results)
        meta_df.to_csv(f'{output_dir}/experiments_meta.csv', index=False)
        
        logger.info("\n" + "="*60)
        logger.info("GRID SEARCH ЗАВЕРШЕН")
        logger.info("="*60)
        logger.info(f"Результаты сохранены в: {output_dir}")


# ============================================================================
# PART 8: ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Полный RAG Pipeline со всеми улучшениями'
    )
    
    # Пути к данным
    parser.add_argument('--websites', default='data/low_filtered.csv',
                       help='Путь к файлу low_filtered.csv')
    parser.add_argument('--questions', default='data/questions_clean.csv',
                       help='Путь к вопросам')
    parser.add_argument('--chunks', default='data/chunks_improved.csv',
                       help='Путь к чанкам')
    parser.add_argument('--embeddings', default='data/embeddings_improved.npy',
                       help='Путь к эмбеддингам')
    parser.add_argument('--index', default='data/index_improved.faiss',
                       help='Путь к FAISS индексу')
    parser.add_argument('--output', default='submission_final.csv',
                       help='Путь для submission')
    
    # Режим работы
    parser.add_argument('--mode', default='full',
                       choices=['full', 'prepare', 'index', 'search', 'experiment'],
                       help='Режим работы')
    
    # Параметры моделей
    parser.add_argument('--embedding-model', 
                       default='intfloat/multilingual-e5-large',
                       help='Модель для эмбеддингов')
    parser.add_argument('--rerank-model',
                       default='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
                       help='Модель для переранжировки')
    
    # Параметры чанкинга
    parser.add_argument('--chunk-length', type=int, default=500,
                       help='Максимальная длина чанка')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                       help='Перекрытие чанков')
    
    # Параметры поиска
    parser.add_argument('--k-retrieve', type=int, default=300,
                       help='Кандидатов на первом этапе')
    parser.add_argument('--k-rerank', type=int, default=50,
                       help='Кандидатов для переранжировки')
    parser.add_argument('--k-final', type=int, default=5,
                       help='Финальных результатов')
    
    # Флаги
    parser.add_argument('--no-rerank', action='store_true',
                       help='Отключить переранжировку')
    parser.add_argument('--no-expansion', action='store_true',
                       help='Отключить расширение запросов')
    parser.add_argument('--no-postprocess', action='store_true',
                       help='Отключить постобработку')
    parser.add_argument('--recreate-chunks', action='store_true',
                       help='Пересоздать чанки')
    parser.add_argument('--recreate-index', action='store_true',
                       help='Пересоздать индекс')
    
    # Эксперименты
    parser.add_argument('--experiment-models', action='store_true',
                       help='Тестировать разные модели')
    parser.add_argument('--grid-search', action='store_true',
                       help='Запустить grid search')
    
    args = parser.parse_args()
    
    # Инициализация пайплайна
    logger.info("="*60)
    logger.info("ИНИЦИАЛИЗАЦИЯ ПОЛНОГО RAG PIPELINE")
    logger.info("="*60)
    
    pipeline = CompletePipeline(
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
        chunk_max_length=args.chunk_length,
        chunk_overlap=args.chunk_overlap,
        use_query_expansion=not args.no_expansion,
        diversity_weight=0.2
    )
    
    # Загружаем вопросы
    df_questions = pd.read_csv(args.questions)
    queries = df_questions['query'].tolist()
    logger.info(f"Загружено запросов: {len(queries)}")
    
    # Режимы работы
    if args.mode == 'prepare' or args.mode == 'full':
        pipeline.prepare_data(
            websites_path=args.websites,
            chunks_output_path=args.chunks,
            recreate_chunks=args.recreate_chunks
        )
    
    if args.mode == 'index' or args.mode == 'full':
        pipeline.build_index(
            chunks_path=args.chunks,
            embeddings_path=args.embeddings,
            index_path=args.index,
            recreate_index=args.recreate_index
        )
    
    if args.mode == 'search' or args.mode == 'full':
        results = pipeline.search(
            queries=queries,
            k_retrieve=args.k_retrieve,
            k_rerank=args.k_rerank,
            k_final=args.k_final,
            use_reranking=not args.no_rerank,
            use_postprocessing=not args.no_postprocess
        )
        
        pipeline.create_submission(results, args.output)
        
        # Статистика
        logger.info("\n" + "="*60)
        logger.info("ИТОГОВАЯ СТАТИСТИКА")
        logger.info("="*60)
        logger.info(f"Запросов обработано: {len(queries)}")
        logger.info(f"Результатов получено: {len(results)}")
        logger.info(f"Среднее на запрос: {len(results) / len(queries):.2f}")
        if 'rerank_score' in results.columns:
            logger.info(f"Средний rerank score: {results['rerank_score'].mean():.4f}")
            logger.info(f"Min/Max rerank score: {results['rerank_score'].min():.4f} / {results['rerank_score'].max():.4f}")
        logger.info("="*60)
    
    if args.mode == 'experiment':
        experimenter = Experimenter(pipeline)
        
        if args.experiment_models:
            # Тестируем разные модели
            models_to_test = [
                'intfloat/multilingual-e5-large',
                'intfloat/multilingual-e5-base',
                'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                'cointegrated/rubert-tiny2'
            ]
            experimenter.test_embedding_models(queries, models_to_test)
        
        if args.grid_search:
            # Grid search по параметрам
            param_grid = {
                'k_retrieve': [200, 300, 400],
                'k_rerank': [30, 50, 70],
                'k_final': [5]
            }
            experimenter.grid_search(queries, param_grid)


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ
# ============================================================================

def quick_test(questions_path: str = 'data/questions_clean.csv',
               chunks_path: str = 'data/chunks_improved.csv',
               n_queries: int = 10):
    """
    Быстрый тест на небольшом количестве запросов
    
    Args:
        questions_path: Путь к вопросам
        chunks_path: Путь к чанкам
        n_queries: Количество запросов для теста
    """
    logger.info("="*60)
    logger.info(f"БЫСТРЫЙ ТЕСТ НА {n_queries} ЗАПРОСАХ")
    logger.info("="*60)
    
    # Загружаем данные
    df_q = pd.read_csv(questions_path)
    queries = df_q['query'].head(n_queries).tolist()
    
    # Простой пайплайн
    pipeline = CompletePipeline(
        embedding_model='intfloat/multilingual-e5-base',  # Быстрая модель
        chunk_max_length=500,
        chunk_overlap=100
    )
    
    # Индексация
    pipeline.build_index(
        chunks_path=chunks_path,
        embeddings_path='data/test_embeddings.npy',
        index_path='data/test_index.faiss',
        recreate_index=True
    )
    
    # Поиск
    results = pipeline.search(
        queries=queries,
        k_retrieve=100,
        k_rerank=20,
        k_final=5,
        use_reranking=True
    )
    
    # Сохраняем
    pipeline.create_submission(results, 'test_submission.csv')
    
    logger.info("="*60)
    logger.info("БЫСТРЫЙ ТЕСТ ЗАВЕРШЕН")
    logger.info("="*60)


def analyze_results(submission_path: str, ground_truth_path: Optional[str] = None):
    """
    Анализ результатов submission
    
    Args:
        submission_path: Путь к submission файлу
        ground_truth_path: Путь к правильным ответам (если есть)
    """
    import ast
    
    df = pd.read_csv(submission_path)
    
    logger.info("="*60)
    logger.info("АНАЛИЗ РЕЗУЛЬТАТОВ")
    logger.info("="*60)
    logger.info(f"Запросов: {len(df)}")
    
    # Конвертируем web_list из строки в список
    df['web_list'] = df['web_list'].apply(ast.literal_eval)
    df['num_results'] = df['web_list'].apply(len)
    
    logger.info(f"Среднее результатов на запрос: {df['num_results'].mean():.2f}")
    logger.info(f"Запросов с <5 результатами: {(df['num_results'] < 5).sum()}")
    
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt = pd.read_csv(ground_truth_path)
        gt['web_list'] = gt['web_list'].apply(ast.literal_eval)
        
        # Считаем Hit@5
        hits = 0
        for idx, row in df.iterrows():
            pred_set = set(row['web_list'][:5])
            true_set = set(gt.loc[gt['q_id'] == row['q_id'], 'web_list'].iloc[0])
            
            if len(pred_set & true_set) > 0:
                hits += 1
        
        hit_at_5 = hits / len(df) * 100
        logger.info(f"\nHit@5: {hit_at_5:.2f}%")
    
    logger.info("="*60)


if __name__ == '__main__':
    # Запуск основной программы
    main()
    
