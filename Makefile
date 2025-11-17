#makefile
# Базовый пайплайн (оригинальный метод)
run:
	python3 src/chunking.py && \
	python3 src/embed.py && \
	python3 src/index_faiss.py && \
	python3 src/search.py && \
	python3 src/rerank.py && \
	python3 src/convert_to_submission.py && \
	python3 src/eval.py


# Гибридный поиск (dense + sparse) - лучшая комбинация
run-hybrid:
	python3 src/chunking.py --strategy overlap --max-length 400 --overlap 100 && \
	python3 src/embed.py && \
	python3 src/index_faiss.py && \
	python3 src/hybrid_search.py --dense-weight 0.7 --sparse-weight 0.3 && \
	python3 src/eval.py && \
	python3 src/rerank.py && \
	python3 src/convert_to_submission.py

run-hybrid2:
	python3 src/embed.py && \
	python3 src/index_faiss.py && \
	python3 src/hybrid_search.py --dense-weight 0.7 --sparse-weight 0.3 && \
	python3 src/eval.py


# Очистка промежуточных файлов
clean:
	rm -f data/chunks.csv data/chunk_embeddings.npy data/index.faiss data/retrieved.csv
