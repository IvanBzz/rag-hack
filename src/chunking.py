import os
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text_with_overlap(text: str, 
                            max_length: int = 768,
                            overlap: int = 192,
                            min_chunk_length: int = 50) -> List[str]:
    """
    Чанкинг с перекрытием для сохранения контекста
    
    Args:
        text: Исходный текст
        max_length: Максимальная длина чанка в символах
        overlap: Размер перекрытия между чанками
        min_chunk_length: Минимальная длина чанка
        
    Returns:
        Список чанков
    """
    if not isinstance(text, str):
        if pd.isna(text):
            return []
        text = str(text)
    
    text = text.strip()
    if not text or len(text) < min_chunk_length:
        return []
    
    # Если текст короткий, возвращаем целиком
    if len(text) <= max_length:
        return [text]
    
    try:
        sentences = sent_tokenize(text, language='russian')
    except:
        # Если не удалось токенизировать, разбиваем по предложениям вручную
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in sentences:
        sent_len = len(sent)
        
        # Если одно предложение больше max_length, разбиваем его
        if sent_len > max_length:
            # Сохраняем текущий чанк если есть
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Разбиваем длинное предложение на части
            words = sent.split()
            temp_chunk = []
            temp_length = 0
            
            for word in words:
                word_len = len(word) + 1  # +1 для пробела
                if temp_length + word_len > max_length and temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                    # Перекрытие: берём последние слова
                    overlap_words = []
                    overlap_len = 0
                    for w in reversed(temp_chunk):
                        if overlap_len + len(w) + 1 <= overlap:
                            overlap_words.insert(0, w)
                            overlap_len += len(w) + 1
                        else:
                            break
                    temp_chunk = overlap_words
                    temp_length = overlap_len
                
                temp_chunk.append(word)
                temp_length += word_len
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
            continue
        
        # Обычная логика для нормальных предложений
        if current_length + sent_len + 1 <= max_length:
            current_chunk.append(sent)
            current_length += sent_len + 1
        else:
            # Сохраняем текущий чанк
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Создаём перекрытие
            overlap_chunk = []
            overlap_length = 0
            
            for s in reversed(current_chunk):
                s_len = len(s) + 1
                if overlap_length + s_len <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_length += s_len
                else:
                    break
            
            # Начинаем новый чанк с перекрытием
            current_chunk = overlap_chunk + [sent]
            current_length = overlap_length + sent_len + 1
    
    # Добавляем последний чанк
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= min_chunk_length:
            chunks.append(chunk_text)
    
    return chunks


def create_chunks_improved(input_path: str, 
                          output_path: str,
                          max_length: int = 512,
                          overlap: int = 128,
                          min_chunk_length: int = 50):
    """
    Создание чанков с улучшенной стратегией
    
    Args:
        input_path: Путь к исходным данным
        output_path: Путь для сохранения чанков
        max_length: Максимальная длина чанка
        overlap: Размер перекрытия
        min_chunk_length: Минимальная длина чанка
    """
    logger.info(f"Загрузка данных из {input_path}")
    df = pd.read_csv(input_path)
    
    logger.info(f"Создание чанков с параметрами:")
    logger.info(f"  max_length: {max_length}")
    logger.info(f"  overlap: {overlap}")
    logger.info(f"  min_chunk_length: {min_chunk_length}")
    
    data = []
    total_chunks = 0
    docs_with_chunks = 0
    
    for idx, row in df.iterrows():
        chunks = chunk_text_with_overlap(
            row['text'],
            max_length=max_length,
            overlap=overlap,
            min_chunk_length=min_chunk_length
        )
        
        if chunks:
            docs_with_chunks += 1
            for chunk in chunks:
                data.append({
                    'web_id': row['web_id'],
                    'text': chunk,
                    'chunk_length': len(chunk)
                })
                total_chunks += 1
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Обработано документов: {idx + 1}/{len(df)}")
    
    df_chunks = pd.DataFrame(data)
    df_chunks.to_csv(output_path, index=False)
    
    # Статистика
    logger.info("\n" + "="*60)
    logger.info("СТАТИСТИКА ЧАНКИНГА:")
    logger.info(f"  Исходных документов: {len(df)}")
    logger.info(f"  Документов с чанками: {docs_with_chunks}")
    logger.info(f"  Всего чанков: {total_chunks}")
    logger.info(f"  Среднее чанков на документ: {total_chunks/docs_with_chunks:.2f}")
    logger.info(f"  Средняя длина чанка: {df_chunks['chunk_length'].mean():.0f}")
    logger.info(f"  Медиана длины чанка: {df_chunks['chunk_length'].median():.0f}")
    logger.info(f"  Min/Max длина: [{df_chunks['chunk_length'].min()}, {df_chunks['chunk_length'].max()}]")
    logger.info("="*60)
    
    return df_chunks


def analyze_chunking_strategy(input_path: str, 
                              max_lengths: List[int] = [256, 384, 512, 768],
                              overlaps: List[int] = [64, 128, 192]):
    """
    Анализ различных стратегий чанкинга
    """
    logger.info("\n" + "="*60)
    logger.info("АНАЛИЗ СТРАТЕГИЙ ЧАНКИНГА")
    logger.info("="*60)
    
    df = pd.read_csv(input_path)
    sample_text = df.iloc[0]['text']
    
    logger.info(f"\nПример текста (первые 500 символов):")
    logger.info(f"{sample_text[:500]}...\n")
    logger.info(f"Полная длина: {len(sample_text)} символов\n")
    
    results = []
    
    for max_len in max_lengths:
        for overlap in overlaps:
            if overlap >= max_len * 0.5:  # Перекрытие не должно быть больше половины
                continue
            
            chunks = chunk_text_with_overlap(sample_text, max_len, overlap)
            
            results.append({
                'max_length': max_len,
                'overlap': overlap,
                'num_chunks': len(chunks),
                'avg_length': sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                'overlap_ratio': overlap / max_len
            })
    
    results_df = pd.DataFrame(results)
    logger.info("\nРезультаты анализа:")
    logger.info(results_df.to_string(index=False))
    
    # Рекомендации
    logger.info("\n💡 РЕКОМЕНДАЦИИ:")
    logger.info("  Для коротких документов (<2000 символов): max_length=512, overlap=128")
    logger.info("  Для средних документов (2000-10000): max_length=768, overlap=192")
    logger.info("  Для длинных документов (>10000): max_length=1024, overlap=256")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Улучшенный чанкинг с перекрытием')
    parser.add_argument('--input', default='data/clean_answer_data.csv')
    parser.add_argument('--output', default='data/chunks.csv')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--min-chunk-length', type=int, default=50)
    parser.add_argument('--analyze', action='store_true',
                       help='Анализ стратегий чанкинга')
    
    args = parser.parse_args()
    
    # Загружаем punkt если нужно
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Скачивание punkt tokenizer...")
        nltk.download('punkt')
    
    if args.analyze:
        analyze_chunking_strategy(args.input)
    else:
        create_chunks_improved(
            args.input,
            args.output,
            max_length=args.max_length,
            overlap=args.overlap,
            min_chunk_length=args.min_chunk_length
        )
