from sentence_transformers import CrossEncoder
import pandas as pd
from tqdm import tqdm
import numpy as np
import logging
from typing import List, Tuple, Optional
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Available reranking models (ordered by quality/speed tradeoff)
RERANK_MODELS = {
    # Best for Russian - specialized cross-encoders
    'ru_large': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',  # Multilingual, good for RU
    'ru_medium': 'cross-encoder/ms-marco-MiniLM-L-6-v2',  # Fast, decent quality
    
    # Experimental Russian models
    'ru_experimental': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    
    # Fallback options
    'fast': 'cross-encoder/ms-marco-TinyBERT-L-2-v2',  # Fastest, lower quality
    'balanced': 'cross-encoder/ms-marco-MiniLM-L-12-v2',  # Good balance
}


class Reranker:
    """
    Cross-encoder based reranker with batch processing and optimization.
    """
    
    def __init__(self, model_name: str = 'ru_large', batch_size: int = 32, 
                 device: Optional[str] = None):
        """
        Initialize reranker with specified model.
        
        Args:
            model_name: Key from RERANK_MODELS or direct model path
            batch_size: Batch size for inference (tune based on GPU memory)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = RERANK_MODELS.get(model_name, model_name)
        self.batch_size = batch_size
        
        logger.info(f"Loading cross-encoder model: {self.model_name}")
        
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = CrossEncoder(self.model_name, device=device, max_length=512)
        logger.info(f"Model loaded on device: {device}")
        
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Score query-document pairs using cross-encoder.
        
        Args:
            pairs: List of (query, document) tuples
            
        Returns:
            Array of relevance scores
        """
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return scores
    
    def rerank_candidates(self, query: str, candidates: List[str], 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Rerank candidate documents for a single query.
        
        Args:
            query: Query text
            candidates: List of candidate document texts
            top_k: Number of top results to return
            
        Returns:
            List of (index, score) tuples sorted by score descending
        """
        if not candidates:
            return []
        
        # Create query-document pairs
        pairs = [(query, doc) for doc in candidates]
        
        # Score all pairs
        scores = self.score_pairs(pairs)
        
        # Sort by score and return top_k
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in ranked_indices]
        
        return results


def rerank(retrieved_path: str, questions_path: str, chunks_path: str, 
           output_path: str, model_name: str = 'ru_large', 
           top_k: int = 5, batch_size: int = 32,
           save_intermediate: bool = True):
    """
    Rerank retrieved documents using cross-encoder.
    
    Args:
        retrieved_path: Path to retrieved candidates CSV
        questions_path: Path to questions CSV
        chunks_path: Path to chunks CSV
        output_path: Path to save reranked results
        model_name: Model to use for reranking
        top_k: Number of top results to keep per query
        batch_size: Batch size for inference
        save_intermediate: Save full scores before filtering to top_k
    """
    logger.info("="*60)
    logger.info("Starting reranking process")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading data...")
    df_retr = pd.read_csv(retrieved_path)
    df_q = pd.read_csv(questions_path)
    df_chunks = pd.read_csv(chunks_path)
    
    logger.info(f"Retrieved candidates: {len(df_retr)} rows")
    logger.info(f"Questions: {len(df_q)} queries")
    logger.info(f"Chunks: {len(df_chunks)} chunks")
    
    # Create web_id to chunk mapping for fast lookup
    logger.info("Creating chunk lookup index...")
    chunk_map = df_chunks.set_index('web_id')['chunk'].to_dict()
    
    # Initialize reranker
    reranker = Reranker(model_name=model_name, batch_size=batch_size)
    
    # Process each query
    reranked = []
    query_ids = df_retr['q_id'].unique()
    
    logger.info(f"Reranking {len(query_ids)} queries...")
    
    for qid in tqdm(query_ids, desc="Reranking queries"):
        # Get query text
        qtext = df_q.loc[df_q['q_id'] == qid, 'query'].values[0]
        
        # Get candidates for this query
        candidates = df_retr[df_retr['q_id'] == qid].copy()
        
        # Get chunk texts
        candidate_texts = []
        valid_indices = []
        
        for idx, row in candidates.iterrows():
            web_id = row['web_id']
            if web_id in chunk_map:
                candidate_texts.append(chunk_map[web_id])
                valid_indices.append(idx)
            else:
                logger.warning(f"web_id {web_id} not found in chunks")
        
        if not candidate_texts:
            logger.warning(f"No valid candidates for query {qid}")
            continue
        
        # Rerank using cross-encoder
        ranked_results = reranker.rerank_candidates(
            query=qtext,
            candidates=candidate_texts,
            top_k=min(top_k, len(candidate_texts))
        )
        
        # Add scores to candidates DataFrame
        scores = np.zeros(len(valid_indices))
        ranks = np.arange(len(valid_indices)) + 1
        
        for rank, (orig_idx, score) in enumerate(ranked_results):
            idx = valid_indices[orig_idx]
            candidates.loc[idx, 'rerank_score'] = score
            candidates.loc[idx, 'rerank_rank'] = rank + 1
        
        # Keep only top_k
        candidates_with_scores = candidates.loc[valid_indices]
        candidates_with_scores = candidates_with_scores.sort_values(
            'rerank_score', ascending=False
        ).head(top_k)
        
        reranked.append(candidates_with_scores)
    
    # Combine all results
    logger.info("Combining results...")
    result_df = pd.concat(reranked, ignore_index=True)
    
    # Save results
    logger.info(f"Saving reranked results to {output_path}")
    result_df.to_csv(output_path, index=False)
    
    # Save intermediate results with all scores if requested
    if save_intermediate:
        intermediate_path = output_path.replace('.csv', '_full_scores.csv')
        logger.info(f"Saving full scores to {intermediate_path}")
        result_df.to_csv(intermediate_path, index=False)
    
    # Statistics
    logger.info("="*60)
    logger.info("Reranking complete!")
    logger.info(f"Total results: {len(result_df)}")
    logger.info(f"Queries processed: {len(query_ids)}")
    logger.info(f"Avg results per query: {len(result_df) / len(query_ids):.2f}")
    if 'rerank_score' in result_df.columns:
        logger.info(f"Score range: [{result_df['rerank_score'].min():.4f}, "
                   f"{result_df['rerank_score'].max():.4f}]")
        logger.info(f"Mean score: {result_df['rerank_score'].mean():.4f}")
    logger.info("="*60)


def compare_models(retrieved_path: str, questions_path: str, chunks_path: str,
                   output_dir: str = 'data/rerank_comparison',
                   models: Optional[List[str]] = None):
    """
    Compare different reranking models.
    
    Args:
        retrieved_path: Path to retrieved candidates CSV
        questions_path: Path to questions CSV
        chunks_path: Path to chunks CSV
        output_dir: Directory to save comparison results
        models: List of model names to compare (None = all)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if models is None:
        models = list(RERANK_MODELS.keys())
    
    logger.info(f"Comparing {len(models)} models: {models}")
    
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing model: {model_name}")
        logger.info(f"{'='*60}")
        
        output_path = os.path.join(output_dir, f'submission_{model_name}.csv')
        
        try:
            rerank(
                retrieved_path=retrieved_path,
                questions_path=questions_path,
                chunks_path=chunks_path,
                output_path=output_path,
                model_name=model_name,
                save_intermediate=False
            )
            results[model_name] = output_path
        except Exception as e:
            logger.error(f"Error with model {model_name}: {e}")
            continue
    
    logger.info("\n" + "="*60)
    logger.info("Model comparison complete!")
    logger.info("="*60)
    logger.info("Generated submissions:")
    for model_name, path in results.items():
        logger.info(f"  {model_name}: {path}")
    logger.info("\nRun eval.py on each submission to compare Hit@5 scores")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Rerank retrieved documents')
    parser.add_argument('--retrieved', default='data/retrieved.csv',
                       help='Path to retrieved candidates')
    parser.add_argument('--questions', default='data/questions_clean.csv',
                       help='Path to questions')
    parser.add_argument('--chunks', default='data/chunks.csv',
                       help='Path to chunks')
    parser.add_argument('--output', default='submission.csv',
                       help='Path to save submission')
    parser.add_argument('--model', default='ru_large',
                       choices=list(RERANK_MODELS.keys()) + ['compare'],
                       help='Model to use or "compare" to test all')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results per query')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    if args.model == 'compare':
        compare_models(
            retrieved_path=args.retrieved,
            questions_path=args.questions,
            chunks_path=args.chunks
        )
    else:
        rerank(
            retrieved_path=args.retrieved,
            questions_path=args.questions,
            chunks_path=args.chunks,
            output_path=args.output,
            model_name=args.model,
            top_k=args.top_k,
            batch_size=args.batch_size
        )