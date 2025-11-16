import pandas as pd
import argparse


def convert_to_submission_format(input_path: str, output_path: str, top_k: int = 5):
    """
    Преобразует формат (q_id, web_id, rank) в формат submission (q_id, web_list).
    
    Args:
        input_path: Путь к файлу с колонками q_id, web_id, rank
        output_path: Путь для сохранения submission файла
        top_k: Количество результатов на запрос (по умолчанию 5)
    """
    # Загружаем данные
    df = pd.read_csv(input_path)
    
    print(f"Загружено {len(df)} строк")
    print(f"Колонки: {list(df.columns)}")
    print(f"Уникальных запросов: {df['q_id'].nunique()}")
    
    # Проверяем наличие нужных колонок
    if 'q_id' not in df.columns or 'web_id' not in df.columns:
        raise ValueError(f"Файл должен содержать колонки 'q_id' и 'web_id'. Найденные: {list(df.columns)}")
    
    # Сортируем по q_id и rank (если есть)
    if 'rank' in df.columns:
        df_sorted = df.sort_values(['q_id', 'rank'])
    else:
        df_sorted = df.sort_values('q_id')
    
    # Группируем по q_id и собираем списки web_id (убираем дубликаты)
    def aggregate_web_ids(group):
        # Удаляем дубликаты, сохраняя порядок
        seen = set()
        result = []
        for web_id in group['web_id']:
            if web_id not in seen:
                seen.add(web_id)
                result.append(int(web_id))
        return result[:top_k]
    
    submission = df_sorted.groupby('q_id').apply(aggregate_web_ids).reset_index()
    submission.columns = ['q_id', 'web_list']
    
    # Преобразуем списки в строковый формат
    submission['web_list'] = submission['web_list'].apply(str)
    
    # Сохраняем
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ Submission файл сохранен: {output_path}")
    print(f"✓ Количество запросов: {len(submission)}")
    print(f"\nПример первых 5 строк:")
    print(submission.head(5).to_string(index=False))
    
    # Проверяем количество результатов на запрос
    submission['list_len'] = submission['web_list'].apply(lambda x: len(eval(x)))
    print(f"\nСтатистика количества результатов на запрос:")
    print(f"  Минимум: {submission['list_len'].min()}")
    print(f"  Максимум: {submission['list_len'].max()}")
    print(f"  Среднее: {submission['list_len'].mean():.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Конвертация формата (q_id, web_id, rank) в submission формат (q_id, web_list)'
    )
    parser.add_argument(
        '--input',
        default='data/retrieved.csv',
        help='Путь к входному файлу с колонками q_id, web_id, rank'
    )
    parser.add_argument(
        '--output',
        default='submission1.csv',
        help='Путь для сохранения submission файла'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Количество результатов на запрос'
    )
    
    args = parser.parse_args()
    
    convert_to_submission_format(args.input, args.output, args.top_k)
