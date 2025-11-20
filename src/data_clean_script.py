import re
import html
import pandas as pd
from bs4 import BeautifulSoup


# ----------------------------------------
# Очистка HTML (удаление тегов)
# ----------------------------------------
def clean_html(text):
    if not isinstance(text, str):
        return ""

    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    cleaned = soup.get_text(separator=" ")
    return cleaned


# ----------------------------------------
# Нормализация обычного текста
# ----------------------------------------
def normalize(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    # Удалить markdown таблицы
    text = re.sub(r"\|.*\|", " ", text)

    # Удалить мусор типа "-----", "=====", "#####"
    text = re.sub(r"[-_=]{3,}", " ", text)

    # Удалить повторяющиеся символы
    text = re.sub(r"(.)\1{4,}", r"\1", text)

    # Убрать лишние пробелы и переносы
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------------------
# Главная функция очистки
# ----------------------------------------
def clean_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required_columns = ["web_id", "url", "kind", "title", "text"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"В CSV нет обязательной колонки: {col}")

    cleaned_rows = []

    for idx, row in df.iterrows():
        clean_row = row.copy()

        clean_row["title"] = normalize(clean_html(row["title"]))

        text = row["text"]

        if "<" in str(text) and ">" in str(text):
            text = clean_html(text)

        text = normalize(text)

        clean_row["text"] = text

        cleaned_rows.append(clean_row)

    cleaned_df = pd.DataFrame(cleaned_rows)

    # Сохраняем в ТОМ ЖЕ формате
    cleaned_df.to_csv(output_csv, index=False)

    print("✔ Очистка завершена")
    print(f"Файл сохранён: {output_csv}")


# ----------------------------------------
# Запуск
# ----------------------------------------
if __name__ == "__main__":
    clean_csv(
        input_csv="data/raw/websites_updated.csv",
        output_csv="data/processed/clean.csv"
    )
