# 🔍 RAG Hackathon Solution | Альфа-Банк Хакатон 2025

> **Retrieval-Augmented Generation система для поиска релевантных фрагментов из веб-источников**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8.0-green.svg)](https://github.com/facebookresearch/faiss)

---

## 📊 Результаты

### Метрика Hit@5: **25.0%** на private лидерборде

- 🏆 **82 место из 180 команд**
- 📈 Улучшение метрики с ~0.79% (в **~30 раз**)
- 🎯 Лучший результат нашей команды

---

## 👥 Команда

| Роль                        | Описание                                                       |
| --------------------------- | -------------------------------------------------------------- |
| **ML Engineer / Team Lead** | Разработка ML-пайплайна, эксперименты с моделями, координация команды |
| **ML Engineer**             | Гибридный поиск, оптимизация индексов                         |
| **Data Analyst**            | Очистка данных, анализ результатов                             |
| **Data Analyst**            | Предобработка, валидация данных                                |

---

## 📖 Описание проекта

Это решение для хакатона **Альфа-Банка (2025)**, где требовалось разработать систему поиска релевантных фрагментов из веб-источников для ответов на вопросы пользователей.

### Ключевые особенности

- ✅ **Гибридный поиск**: комбинация dense (embeddings) и sparse (TF-IDF) методов
- ✅ **Мультиязычная модель**: `intfloat/multilingual-e5-large` для качественных эмбеддингов
- ✅ **FAISS индексация**: быстрый векторный поиск
- ✅ **Оптимизированный пайплайн**: от очистки данных до финального submission

> ⚠️ **Примечание**: Эксперименты с chunking и reranking показали ухудшение результатов, поэтому финальное решение использует простой гибридный поиск без дополнительных этапов.

---

## 🏗️ Структура проекта

```text
rag-hack/
│
├── 📁 data/
│   ├── raw/                          # Исходные данные
│   └── processed/                    # Обработанные данные
│
├── 📁 src/                           # Основной пайплайн
│   ├── 01_clean_data.py              # Очистка и нормализация данных
│   ├── 02_embed.py                   # Генерация эмбеддингов
│   ├── 03_index_faiss.py             # Построение FAISS индекса
│   ├── 04_hybrid_search.py           # Гибридный поиск (dense + sparse)
│   └── 05_convert_to_submission.py   # Конвертация в формат submission
│
├── 📁 experiments/                   # Эксперименты (не использовались в финале)
│   ├── chunking.py                   # Стратегии разбиения текста
│   └── rerank.py                     # Reranking модели
│
├── 📄 requirements.txt               # Зависимости проекта
├── 📄 Makefile                       # Команды для запуска пайплайна
└── 📄 README.md                      # Документация
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных

Убедитесь, что исходные данные находятся в `data/raw/`:

- `questions_clean.csv` - вопросы
- `websites_updated.csv` - веб-источники

### 3. Запуск полного пайплайна

```bash
make run-final
```

Или вручную:

```bash
python src/01_clean_data.py
python src/02_embed.py
python src/03_index_faiss.py
python src/04_hybrid_search.py --dense-weight 0.7 --sparse-weight 0.3
python src/05_convert_to_submission.py
```

---

## 🔧 Описание компонентов

### 1. Очистка данных (`01_clean_data.py`)

- Удаление HTML-тегов и скриптов
- Нормализация текста (unescape, удаление markdown-таблиц)
- Фильтрация по длине (10-30000 символов)
- Приведение к нижнему регистру

**Вход:** `data/raw/websites_updated.csv`  
**Выход:** `data/processed/clean.csv`, `data/processed/filtered_file.csv`

---

### 2. Генерация эмбеддингов (`02_embed.py`)

**Модель:** `intfloat/multilingual-e5-large`

- Мультиязычная модель с поддержкой русского языка
- Размерность эмбеддингов: 1024
- Batch processing с progress bar

**Вход:** `data/processed/filtered_file.csv`  
**Выход:** `data/processed/embeddings.npy`

---

### 3. FAISS индексация (`03_index_faiss.py`)

**Тип индекса:** `IndexFlatL2` (точный поиск)

- L2 расстояние для поиска ближайших соседей
- Быстрый поиск для больших объемов данных

**Вход:** `data/processed/embeddings.npy`  
**Выход:** `data/processed/index.faiss`

---

### 4. Гибридный поиск (`04_hybrid_search.py`)

**Класс:** `HybridRetriever`

#### Методы поиска

1. **Dense Search** (вес: 0.7)
   - Векторный поиск через FAISS
   - Использует эмбеддинги запросов и документов
   - Косинусная близость

2. **Sparse Search** (вес: 0.3)
   - TF-IDF векторное представление
   - Uni- и bi-граммы
   - Лексическое совпадение

#### Комбинирование результатов

```python
combined_score = 0.7 * dense_score + 0.3 * sparse_score
```

**Параметры:**

- `--k`: количество кандидатов (по умолчанию: 200)
- `--dense-weight`: вес dense поиска (по умолчанию: 0.7)
- `--sparse-weight`: вес sparse поиска (по умолчанию: 0.3)

**Выход:** `data/processed/retrieved.csv`

---

### 5. Конвертация в submission (`05_convert_to_submission.py`)

Преобразует формат `(q_id, web_id, rank)` в формат submission `(q_id, web_list)`.

**Параметры:**

- `--top-k`: количество результатов на запрос (по умолчанию: 5)

**Вход:** `data/processed/retrieved.csv`  
**Выход:** `submission_retrived_final.csv`

---

## ⚙️ Конфигурация

Рекомендуемые параметры (оптимизированы на валидации):

```python
dense_weight = 0.7    # Вес dense поиска
sparse_weight = 0.3  # Вес sparse поиска
k = 200              # Количество кандидатов
top_k = 5            # Финальное количество результатов
```

Модель эмбеддингов задается в `src/02_embed.py`:

```python
MODEL = 'intfloat/multilingual-e5-large'
```

---

## 🔍 Технические детали

### Архитектура пайплайна

```text
Raw Data → Cleaning → Embedding → Indexing → Hybrid Search → Submission
```

### Оптимизации

1. **Batch processing**: обработка данных батчами для ускорения
2. **Дедупликация**: удаление дубликатов web_id для каждого запроса
3. **Нормализация скоров**: приведение скоров к единому диапазону

---

## 📝 Формат данных

### Входные данные

**questions_clean.csv:**

```csv
q_id,query
1,"Как открыть счет в банке?"
2,"Какие документы нужны для кредита?"
```

**websites_updated.csv:**

```csv
web_id,url,kind,title,text
1,https://...,article,Заголовок,Текст статьи...
```

### Выходные данные

**submission_retrived_final.csv:**

```csv
q_id,web_list
1,"[123, 456, 789, 101, 112]"
2,"[234, 567, 890, 111, 223]"
```

---

Made with ❤️ by Titanic 2 for Alfa Hackathon 2025

