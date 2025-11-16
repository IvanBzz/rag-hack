import pandas as pd  
import argparse


def format_submission(input_path: str, output_path: str, top_k: int = 5):
    """
    Преобразует развернутый формат с rerank результатами в формат submission.
    
    Args:
        input_path: Путь к файлу с rerank результатами (q_id, web_id, rank, rerank_score, rerank_rank)
        output_path: Путь для сохранения submission файла (q_id, web_list)
        top_k: Количество результатов на запрос (по умолчанию 5)
    """
    # Загружаем данные
    df = pd.read_csv(input_path)
    
    print(f"Загружено {len(df)} строк")
    print(f"Уникальных запросов: {df['q_id'].nunique()}")
    
    # Сортируем по q_id и rerank_rank (или rerank_score если rank не заполнен)
    if 'rerank_rank' in df.columns and df['rerank_rank'].notna().any():
        df_sorted = df.sort_values(['q_id', 'rerank_rank'])
    elif 'rerank_score' in df.columns:
        df_sorted = df.sort_values(['q_id', 'rerank_score'], ascending=[True, False])
    else:
        # Фоллбэк на rank если ничего нет
        df_sorted = df.sort_values(['q_id', 'rank'])
    
    # Группируем по q_id и берем top_k web_id для каждого запроса
    submission = df_sorted.groupby('q_id')['web_id'].apply(
        lambda x: list(x.head(top_k))
    ).reset_index()
    
    # Переименовываем колонку
    submission.columns = ['q_id', 'web_list']
    
    # Преобразуем списки в строковый формат
    submission['web_list'] = submission['web_list'].apply(str)
    
    # Сохраняем
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission файл сохранен: {output_path}")
    print(f"Количество запросов: {len(submission)}")
    print(f"\nПример первых строк:")
    print(submission.head(3))
    
    # Проверяем количество результатов на запрос
    submission['list_len'] = submission['web_list'].apply(lambda x: len(eval(x)))
    print(f"\nСтатистика количества результатов на запрос:")
    print(submission['list_len'].describe())


def auto_format_if_needed(filepath: str, top_k: int = 5):
    """
    Автоматически форматирует файл, если он в развернутом формате.
    
    Args:
        filepath: Путь к файлу
        top_k: Количество результатов на запрос
    """
    df = pd.read_csv(filepath)
    
    # Проверяем, нужно ли форматирование
    if 'web_list' not in df.columns and 'web_id' in df.columns:
        print(f"\nОбнаружен развернутый формат. Выполняется форматирование...")
        output_path = filepath.replace('.csv', '_formatted.csv')
        format_submission(filepath, output_path, top_k)
        print(f"Используйте файл {output_path} для submission")
        return output_path
    else:
        print(f"\nФайл уже в правильном формате")
        return filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Преобразование rerank результатов в формат submission'
    )
    parser.add_argument(
        '--input', 
        default='submission.csv',
        help='Путь к файлу с rerank результатами'
    )
    parser.add_argument(
        '--output', 
        default='submission_formatted.csv',
        help='Путь для сохранения submission файла'
    )
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=5,
        help='Количество результатов на запрос'
    )
    
    args = parser.parse_args()
    
    format_submission(args.input, args.output, args.top_k)