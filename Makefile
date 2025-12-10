#makefile
# Базовый пайплайн (оригинальный метод)
run:
	python3 experiments/chunking.py && \
	python3 src/02_embed.py && \
	python3 src/index_faiss.py && \
	python3 src/search.py && \
	python3 experiments/rerank.py && \
	python3 src/05_convert_to_submission.py && \
	python3 src/06_eval.py

run-final:
	python3 src/01_clean_data.py
	python3 src/02_embed.py && \
	python3 src/03_index_faiss.py && \
	python3 src/04_hybrid_search.py --dense-weight 0.7 --sparse-weight 0.3 && \
	python3 src/05_convert_to_submission.py


# Очистка промежуточных файлов
clean:
	rm -f data/processed/*.csv