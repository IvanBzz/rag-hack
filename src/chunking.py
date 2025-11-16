import os

import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd


# def _ensure_punkt():
#     """Guarantee that the Punkt tokenizer data is available without forcing a download."""
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     known_locations = [
#         os.environ.get("NLTK_DATA"),
#         os.path.join(project_root, ".venv", "nltk_data"),
#         os.path.join(project_root, "nltk_data"),
#     ]
#     for location in known_locations:
#         if location and os.path.isdir(location):
#             nltk.data.path.append(location)
#     try:
#         nltk.data.find("tokenizers/punkt")
#     except LookupError:
#         nltk.download("punkt")


def chunk_text(text, max_length=500):
    if not isinstance(text, str):
        if pd.isna(text):
            return []
        text = str(text)
    text = text.strip()
    if not text:
        return []
    sentences = sent_tokenize(text, language='russian')
    chunks, current_chunk = [], []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) <= max_length:
            current_chunk.append(sent)
            current_len += len(sent)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sent]
            current_len = len(sent)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def create_chunks(input_path, output_path):
    df = pd.read_csv(input_path)
    data = []
    for _, row in df.iterrows():
        for chunk in chunk_text(row['text']):
            data.append({'web_id': row['web_id'], 'chunk': chunk})
    pd.DataFrame(data).to_csv(output_path, index=False)


if __name__ == '__main__':
    create_chunks('data/filtered_file.csv', 'data/chunks.csv')
