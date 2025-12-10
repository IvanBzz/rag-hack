import re
import html
import pandas as pd
from bs4 import BeautifulSoup


# ----------------------------------------
# 1. Классификация строки
# ----------------------------------------
def classify(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "empty"

    t = text.strip()

    # HTML
    if "<" in t and ">" in t:
        return "html"

    # QA pattern
    if re.search(r"(вопрос|question)[:\s]|(ответ|answer)[:\s]", t.lower()):
        return "qa"

    # Логи / ошибки
    if re.search(r"(error|exception|traceback|fatal|stack)", t.lower()):
        return "log"

    # Слишком короткий текст = шум
    if len(t) < 20:
        return "junk"

    # Повторяющиеся символы
    if re.search(r"(.)\1{4,}", t):  # aaaaa, !!!!!!
        return "junk"

    return "text"


# ----------------------------------------
# 2. Очистка HTML
# ----------------------------------------
def clean_html(text):
    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ")


# ----------------------------------------
# 3. Нормализация полезного текста
# ----------------------------------------
def normalize_text(text):
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "")
    return text.strip()


# ----------------------------------------
# 4. Извлечение QA пар
# ----------------------------------------
def extract_qa(text):
    q = re.search(r"(вопрос|question)[:\s]*(.*)", text, re.I)
    a = re.search(r"(ответ|answer)[:\s]*(.*)", text, re.I)

    question = q.group(2).strip() if q else None
    answer = a.group(2).strip() if a else None

    return question, answer


# ----------------------------------------
# 5. Чанкование текста
# ----------------------------------------
def chunk_text(text, max_tokens=200):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])


# ----------------------------------------
# 6. ГЛАВНАЯ ФУНКЦИЯ
# ----------------------------------------
def clean_csv_rag(input_csv, text_column, chunks_out, qa_out):
    df = pd.read_csv(input_csv)

    if text_column not in df.columns:
        raise ValueError(f"В CSV нет колонки '{text_column}'")

    clean_chunks = []
    qa_pairs = []

    for row in df[text_column]:
        block = str(row).strip()
        btype = classify(block)

        if btype in ["empty", "junk", "log"]:
            continue

        # HTML → чистый текст
        if btype == "html":
            block = clean_html(block)

        # QA-пары → структура
        if btype == "qa":
            q, a = extract_qa(block)
            if q and a:
                qa_pairs.append({"question": q, "answer": a})
            continue

        # Обычный текст
        if btype == "text":
            block = normalize_text(block)

            if len(block) < 20:
                continue

            # чанкование
            for ch in chunk_text(block):
                if 20 < len(ch) < 2500:
                    clean_chunks.append({"chunk": ch})

    # Сохраняем
    pd.DataFrame(clean_chunks).drop_duplicates().to_csv(chunks_out, index=False)
    pd.DataFrame(qa_pairs).drop_duplicates().to_csv(qa_out, index=False)

    print("✔ Очистка завершена")
    print(f"✔ Чанки: {chunks_out}")
    print(f"✔ QA: {qa_out}")


# ----------------------------------------
# Запуск
# ----------------------------------------
if __name__ == "__main__":
    clean_csv_rag(
        input_csv="data/websites_updated.csv",
        text_column="text",          # ← ИМЯ КОЛОНКИ С ТЕКСТОМ
        chunks_out="data/clean_chunks.csv",
        qa_out="data/clean_qa.csv",
    )
