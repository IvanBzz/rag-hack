import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import time  # Добавляем импорт модуля time

def e5_large_pipeline():
    """
    Пайплайн с моделью intfloat/multilingual-e5-large
    Улучшения:
    - E5: отбор TOP_K_CANDIDATES кандидатов (по умолчанию 20)
    - CrossEncoder rerank на этих кандидатах
    - Финальная сортировка: финальный скор = ALPHA * cross_norm + (1-ALPHA) * e5_score
    - В итоговый csv сохраняются только q_id и web_list
    """
    print("=== ЗАПУСК С МОДЕЛЬЮ multilingual-e5-large ===")
    
    # Загрузка данных
    questions = pd.read_csv('data/questions_clean.csv')
    documents = pd.read_csv('data/filtered_file.csv')
    
    print(f"Загружено: {len(questions)} вопросов, {len(documents)} документов")
    
    # Специальная предобработка для E5 модели
    def e5_preprocess(text, text_type="query"):
        """
        E5 модель требует специального формата:
        - Запросы: "query: {текст}"
        - Документы: "passage: {текст}"
        """
        if not isinstance(text, str):
            text = ""
        
        # Базовая очистка
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Добавляем префикс в зависимости от типа текста
        if text_type == "query":
            return f"query: {text}"
        else:  # passage
            return f"passage: {text}"
    
    # Предобработка данных для E5
    print("Специальная предобработка для E5 модели...")
    # Убедимся, что колонки text и query есть
    if 'text' not in documents.columns:
        # Используем первую текстовую колонку, если 'text' отсутствует
        text_cols = [col for col in documents.columns if documents[col].dtype == 'object' and col != 'web_id']
        documents['text'] = documents[text_cols[0]] if text_cols else ''
    
    if 'query' not in questions.columns:
        # Используем первую текстовую колонку, если 'query' отсутствует
        query_cols = [col for col in questions.columns if questions[col].dtype == 'object' and col != 'q_id']
        questions['query'] = questions[query_cols[0]] if query_cols else ''
    
    documents['e5_text'] = documents['text'].apply(lambda x: e5_preprocess(x, "passage"))
    questions['e5_query'] = questions['query'].apply(lambda x: e5_preprocess(x, "query"))
    
    # Загрузка E5 модели
    print("Загрузка модели intfloat/multilingual-e5-large...")
    try:
        model = SentenceTransformer(
            'intfloat/multilingual-e5-large',
            trust_remote_code=True
        )
        print("✓ Модель успешно загружена")
    except Exception as e:
        print(f"✗ Ошибка загрузки E5 модели: {e}")
        print("Пробуем альтернативные модели...")
        return fallback_to_other_models(questions, documents)
    
    # Проверка работы модели
    try:
        test_texts = ["query: test", "passage: test document"]
        test_embeddings = model.encode(test_texts, normalize_embeddings=True)
        print(f"✓ Тест модели пройден, размерность: {test_embeddings.shape[1]}")
    except Exception as e:
        print(f"✗ Ошибка тестирования модели: {e}")
        return fallback_to_other_models(questions, documents)
    
    # Векторизация документов с учетом ограничений памяти
    print("Векторизация документов для E5...")
    
    doc_embeddings_list = []
    batch_size = 8
    
    i = 0
    while i < len(documents):
        batch_end = min(i + batch_size, len(documents))
        batch_texts = documents['e5_text'].iloc[i:batch_end].tolist()
        
        try:
            batch_embeddings = model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_tensor=False
            )
            doc_embeddings_list.append(batch_embeddings)
            print(f"Векторизовано документов: {batch_end}/{len(documents)}")
            i = batch_end  # Успешно обработали батч, переходим к следующему
        except Exception as e:
            print(f"Ошибка в батче {i}: {e}")
            # Пробуем еще меньший батч
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"Уменьшаем размер батча до {batch_size}")
            else:
                # Если даже с batch_size=1 не получается, пропускаем документ
                print(f"Пропускаем документ {i}")
                i += 1
    
    if not doc_embeddings_list:
        print("Не удалось векторизовать документы, используем альтернативную модель")
        return fallback_to_other_models(questions, documents)
    
    doc_embeddings = np.vstack(doc_embeddings_list)
    print(f"✓ Векторизация завершена, размер: {doc_embeddings.shape}")
    
    # Обработка вопросов
    print("Обработка вопросов с E5 моделью...")
    results = []
    TOP_K = 5
    TOP_K_CANDIDATES = 20
    
    question_batch_size = 4
    
    for i in range(0, len(questions), question_batch_size):
        batch_end = min(i + question_batch_size, len(questions))
        batch_questions = questions.iloc[i:batch_end]
        
        try:
            # Векторизация батча вопросов
            batch_queries = batch_questions['e5_query'].tolist()
            q_embeddings = model.encode(
                batch_queries,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_tensor=False
            )
            
            # Расчет схожестей
            batch_similarities = cosine_similarity(q_embeddings, doc_embeddings)
            
            for j, (idx, row) in enumerate(batch_questions.iterrows()):
                similarities = batch_similarities[j]
                
                # Берем TOP_K_CANDIDATES кандидатов
                top_indices = np.argsort(similarities)[-TOP_K_CANDIDATES:][::-1]
                candidate_web_ids = documents.iloc[top_indices]['web_id'].tolist()
                candidate_texts = documents.iloc[top_indices]['text'].tolist()
                e5_scores = similarities[top_indices].tolist()
                
                # Гарантируем длину
                while len(candidate_web_ids) < TOP_K_CANDIDATES:
                    candidate_web_ids.append("")
                    candidate_texts.append("")
                    e5_scores.append(0.0)
                
                results.append({
                    'q_id': row['q_id'],
                    'query': row.get('query', ''),
                    'candidate_web_ids': candidate_web_ids[:TOP_K_CANDIDATES],
                    'candidate_texts': candidate_texts[:TOP_K_CANDIDATES],
                    'e5_scores': e5_scores[:TOP_K_CANDIDATES],
                    'model_used': 'intfloat/multilingual-e5-large'
                })
            
            print(f"Обработано вопросов: {batch_end}/{len(questions)}")
            
        except Exception as e:
            print(f"Ошибка при обработке батча вопросов {i}: {e}")
            # Обрабатываем вопросы по одному
            for j in range(i, batch_end):
                try:
                    single_result = process_single_question_e5(
                        questions.iloc[j], documents, model, doc_embeddings, TOP_K_CANDIDATES
                    )
                    results.append({
                        'q_id': questions.iloc[j]['q_id'],
                        'query': questions.iloc[j].get('query', ''),
                        'candidate_web_ids': single_result['candidate_web_ids'],
                        'candidate_texts': single_result['candidate_texts'],
                        'e5_scores': single_result['e5_scores'],
                        'model_used': 'intfloat/multilingual-e5-large'
                    })
                except Exception as e2:
                    print(f"Ошибка при обработке вопроса {questions.iloc[j]['q_id']}: {e2}")
                    # Добавляем пустой результат
                    results.append({
                        'q_id': questions.iloc[j]['q_id'],
                        'query': questions.iloc[j].get('query', ''),
                        'candidate_web_ids': [""] * TOP_K_CANDIDATES,
                        'candidate_texts': [""] * TOP_K_CANDIDATES,
                        'e5_scores': [0.0] * TOP_K_CANDIDATES,
                        'model_used': 'intfloat/multilingual-e5-large'
                    })
    
    # === RERANKING БЛОК (CrossEncoder) ===
    print("Загрузка CrossEncoder для реранжирования...")
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        print("✓ Реранкер загружен")
    except Exception as e:
        print(f"Ошибка загрузки CrossEncoder: {e}")
        # Если реранкер не загрузился, просто используем E5 scores
        print("Пропускаем реранкинг, используем только E5 scores")
        for r in results:
            top_indices = np.argsort(r['e5_scores'])[-TOP_K:][::-1]
            r['web_list'] = [r['candidate_web_ids'][idx] for idx in top_indices]
            # Гарантируем 5 элементов
            while len(r['web_list']) < TOP_K:
                r['web_list'].append("")
        
        result_df = pd.DataFrame(results)
        result_df = result_df[['q_id', 'web_list']]
        result_df.to_csv('result_e5_large.csv', index=False)
        print("Результаты сохранены в result_e5_large.csv")
        return result_df

    # Функция очистки текста перед подачей в CrossEncoder
    def clean_text(s):
        if not isinstance(s, str):
            return ""
        s = re.sub(r'<[^>]+>', ' ', s)  # удалить HTML
        s = re.sub(r'http\S+', ' ', s)  # удалить URL
        s = s.replace('\n', ' ').replace('\r', ' ')
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    print("Применяем реранжирование кандидатов...")
    ALPHA = 0.7

    # Добавляем отслеживание времени выполнения реранжирования
    rerank_start_time = time.time()
    total_rerank = len(results)
    
    for idx, r in enumerate(results):
        query_text = r.get('query', '')
        candidate_texts = r.get('candidate_texts', [])
        candidate_ids = r.get('candidate_web_ids', [])
        e5_scores = r.get('e5_scores', [])
        
        # Фильтруем пустые кандидаты
        valid_indices = [idx for idx, (web_id, text) in enumerate(zip(candidate_ids, candidate_texts)) 
                        if web_id and text and isinstance(text, str)]
        
        if not valid_indices:
            r['web_list'] = [""] * TOP_K
            continue
            
        valid_web_ids = [candidate_ids[i] for i in valid_indices]
        valid_texts = [candidate_texts[i] for i in valid_indices]
        valid_e5_scores = [e5_scores[i] for i in valid_indices]
        
        # Очистка
        q_clean = clean_text(query_text)
        pairs = [[q_clean, clean_text(d)] for d in valid_texts]

        try:
            cross_scores = reranker.predict(pairs, batch_size=16)
        except Exception as e:
            print(f"Ошибка при реранке для q_id={r['q_id']}: {e}")
            # Используем только E5 scores при ошибке
            top_indices = np.argsort(valid_e5_scores)[-TOP_K:][::-1]
            r['web_list'] = [valid_web_ids[idx] for idx in top_indices]
            while len(r['web_list']) < TOP_K:
                r['web_list'].append("")
            continue

        # Нормализуем e5_scores на [0,1]
        e5_arr = np.array(valid_e5_scores, dtype=float)
        if len(e5_arr) > 0:
            e5_min, e5_max = e5_arr.min(), e5_arr.max()
            if e5_max - e5_min > 1e-6:
                e5_norm = (e5_arr - e5_min) / (e5_max - e5_min)
            else:
                e5_norm = np.zeros_like(e5_arr)
        else:
            e5_norm = np.array([])

        # Нормализуем cross_scores
        cross_arr = np.array(cross_scores, dtype=float)
        if len(cross_arr) > 0:
            cross_min, cross_max = cross_arr.min(), cross_arr.max()
            if cross_max - cross_min > 1e-6:
                cross_norm = (cross_arr - cross_min) / (cross_max - cross_min)
            else:
                cross_norm = np.zeros_like(cross_arr)
        else:
            cross_norm = np.array([])

        if len(e5_norm) > 0 and len(cross_norm) > 0:
            fin_scores = ALPHA * cross_norm + (1 - ALPHA) * e5_norm
        elif len(cross_norm) > 0:
            fin_scores = cross_norm
        else:
            fin_scores = e5_norm

        combined = list(zip(valid_web_ids, fin_scores))
        combined.sort(key=lambda x: x[1], reverse=True)

        # Берём топ-K в финальный web_list
        top_final = combined[:TOP_K]
        final_webs = [w for w, s in top_final]

        # Записываем в результат (дополняем если нужно)
        while len(final_webs) < TOP_K:
            final_webs.append("")

        r['web_list'] = final_webs
        
        # Расчет и вывод оставшегося времени
        current_time = time.time()
        elapsed_time = current_time - rerank_start_time
        progress = (idx + 1) / total_rerank
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total_time - elapsed_time
        
        # Форматируем время в читаемый вид
        if remaining_time > 3600:
            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            time_str = f"{hours}ч {minutes}м"
        elif remaining_time > 60:
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            time_str = f"{minutes}м {seconds}с"
        else:
            time_str = f"{int(remaining_time)}с"
        
        # Выводим прогресс каждые 10% или для каждого вопроса, если их мало
        if (idx + 1) % max(1, total_rerank // 10) == 0 or (idx + 1) == total_rerank:
            print(f"Реранжирование: {idx + 1}/{total_rerank} ({progress:.1%}) | Осталось: {time_str}")

    # Статистика
    all_e5_scores = []
    e5_top1_list = []
    for r in results:
        e5 = r.get('e5_scores', [])
        if len(e5) > 0:
            all_e5_scores.extend(e5)
            e5_top1_list.append(e5[0])
    
    avg_similarity = np.mean(all_e5_scores) if all_e5_scores else 0
    avg_top1_similarity = np.mean(e5_top1_list) if e5_top1_list else 0

    print(f"\n=== СТАТИСТИКА (E5 кандидаты) ===")
    print(f"Обработано вопросов: {len(results)}")
    print(f"Средняя схожесть E5 (все кандидаты): {avg_similarity:.3f}")
    print(f"Средняя схожесть E5 (top1): {avg_top1_similarity:.3f}")
    
    # Сохранение результатов
    result_df = pd.DataFrame(results)
    if 'web_list' not in result_df.columns:
        result_df['web_list'] = result_df.get('candidate_web_ids', [[] for _ in range(len(result_df))]).apply(
            lambda lst: lst[:TOP_K] if isinstance(lst, list) else [""]*TOP_K)
    
    result_df = result_df[['q_id', 'web_list']]
    result_df.to_csv('result_e5_large.csv', index=False)
    print("Результаты сохранены в result_e5_large.csv")

    return result_df

def process_single_question_e5(question_row, documents, model, doc_embeddings, top_k_candidates=20):
    """Обработка одного вопроса с E5 моделью (резервная функция)"""
    query = question_row['e5_query']
    
    try:
        q_embedding = model.encode([query], normalize_embeddings=True, convert_to_tensor=False)
        similarities = cosine_similarity(q_embedding, doc_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k_candidates:][::-1]
        
        web_list = documents.iloc[top_indices]['web_id'].tolist()
        candidate_texts = documents.iloc[top_indices]['text'].tolist()
        top_similarities = similarities[top_indices].tolist()
        
        # Гарантируем нужное количество элементов
        while len(web_list) < top_k_candidates:
            web_list.append("")
            candidate_texts.append("")
            top_similarities.append(0.0)
        
        return {
            'candidate_web_ids': web_list[:top_k_candidates],
            'candidate_texts': candidate_texts[:top_k_candidates],
            'e5_scores': top_similarities[:top_k_candidates]
        }
    except Exception as e:
        print(f"Ошибка в process_single_question_e5: {e}")
        return {
            'candidate_web_ids': [""] * top_k_candidates,
            'candidate_texts': [""] * top_k_candidates,
            'e5_scores': [0.0] * top_k_candidates
        }

def fallback_to_other_models(questions, documents):
    """Резервный вариант с другими моделями"""
    print("Запуск резервного варианта...")
    
    MODEL_OPTIONS = [
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    ]
    
    for model_name in MODEL_OPTIONS:
        try:
            print(f"Пробуем модель: {model_name}")
            model = SentenceTransformer(model_name)
            
            # Подготовка текстов документов
            if 'text' not in documents.columns:
                text_cols = [col for col in documents.columns if documents[col].dtype == 'object' and col != 'web_id']
                doc_texts = documents[text_cols[0]].tolist() if text_cols else [''] * len(documents)
            else:
                doc_texts = documents['text'].tolist()
            
            doc_embeddings = model.encode(
                doc_texts, 
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_tensor=False
            )
            
            results = []
            TOP_K = 5
            
            for i, row in questions.iterrows():
                query_text = row.get('query', '')
                q_embedding = model.encode([query_text], normalize_embeddings=True, convert_to_tensor=False)
                similarities = cosine_similarity(q_embedding, doc_embeddings)[0]
                top_indices = np.argsort(similarities)[-TOP_K:][::-1]
                
                web_list = documents.iloc[top_indices]['web_id'].tolist()
                while len(web_list) < TOP_K:
                    web_list.append("")
                
                results.append({
                    'q_id': row['q_id'], 
                    'web_list': web_list[:TOP_K],
                    'model_used': model_name
                })
                
                
                if (i + 1) % 50 == 0:
                    print(f"Обработано: {i + 1}/{len(questions)}")
            
            result_df = pd.DataFrame(results)
            result_df[['q_id', 'web_list']].to_csv('result_fallback.csv', index=False)
            print("Резервные результаты сохранены в result_fallback.csv")
            return result_df
            
        except Exception as e:
            print(f"Ошибка с моделью {model_name}: {e}")
            continue
    
    print("Все резервные модели не сработали")
    return None

if __name__ == "__main__":
    e5_large_pipeline()
