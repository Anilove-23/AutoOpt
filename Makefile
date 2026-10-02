.PHONY: install test analyze synthesize recommend clean

install:
	pip install -e .

test:
	pytest tests/ -v

synthesize:
	autoopt synthesize -n 100 -o benchmarks/synthesized

recommend:
	autoopt recommend benchmarks/custom/matrix_mult.c

analyze:
	autoopt analyze benchmarks/custom/matrix_mult.c

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.exe" -delete
	find . -type f -name "*.s" -delete
