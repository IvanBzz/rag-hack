import re
import html
import pandas as pd
from bs4 import BeautifulSoup

# ------------------------------------------------------------
# 1. Удаление HTML
# ------------------------------------------------------------
def clean_html(text):
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ")

# ------------------------------------------------------------
# 2. Нормализация текста (как в clean_answer_data.csv)
# ------------------------------------------------------------
def normalize(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    # Удаляем markdown-таблицы
    text = re.sub(r"\|.*?\|", " ", text)

    # длинные символы ----, =====
    text = re.sub(r"[-_=]{3,}", " ", text)

    # повторяющиеся символы
    text = re.sub(r"(.)\1{4,}", r"\1", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ------------------------------------------------------------
# 3. Финальная очистка (нижний регистр + удаление пунктуации)
# ------------------------------------------------------------
def final_clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# ------------------------------------------------------------
# 4. Очистка CSV — полностью воссозданная логика
# ------------------------------------------------------------
def clean_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required = ["web_id", "url", "kind", "title", "text"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Нет нужной колонки: {col}")

    cleaned = []

    for _, row in df.iterrows():
        row2 = row.copy()

        # title
        t = row2["title"]
        t = clean_html(t)
        t = normalize(t)
        t = final_clean(t)
        row2["title"] = t

        # text
        x = row2["text"]
        if "<" in str(x) and ">" in str(x):
            x = clean_html(x)
        x = normalize(x)
        x = final_clean(x)
        row2["text"] = x

        cleaned.append(row2)

    df2 = pd.DataFrame(cleaned)

    # выбросы длины
    df2["len"] = df2["text"].astype(str).str.len()
    df2 = df2[(df2["len"] >= 10) & (df2["len"] <= 30000)]
    df2 = df2.drop(columns=["len"])

    df2.to_csv(output_csv, index=False)
    print("✔ clean_csv complete →", output_csv)

# ------------------------------------------------------------
# 5. Фильтрация
# ------------------------------------------------------------
def filter_columns(input_csv, output_csv, columns=None):
    if columns is None:
        columns = ["web_id", "text"]

    df = pd.read_csv(input_csv)

    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].apply(final_clean)

    df = df[columns]
    df.to_csv(output_csv, index=False)
    print("✔ filter_columns complete →", output_csv)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    clean_csv(
        input_csv="data/raw/websites_updated.csv",
        output_csv="data/processed/clean.csv"
    )

    filter_columns(
        input_csv="data/processed/clean.csv",
        output_csv="data/processed/filtered_file.csv",
        columns=["web_id", "text"]
    )