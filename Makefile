.PHONY: install dev-install test lint analyze synthesize recommend benchmark sequences clean help

# ── Installation ────────────────────────────────────────
install:
	pip install -e .
	@echo ""
	@echo "✓ AutoOpt installed. You can now use: autoopt --help"

dev-install:
	pip install -e ".[dev]"
	@echo "✓ AutoOpt installed with dev extras (pytest, shap, optuna)."

# ── Testing ─────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	python -m py_compile autoopt/cli.py autoopt/__init__.py autoopt/models/predictor.py
	@echo "✓ Syntax OK"

# ── CLI demos ───────────────────────────────────────────
recommend:
	autoopt recommend benchmarks/custom/matrix_mult.c

analyze:
	autoopt analyze benchmarks/custom/matrix_mult.c

benchmark:
	autoopt benchmark benchmarks/custom/matrix_mult.c --runs 3

sequences:
	autoopt sequences

synthesize:
	autoopt synthesize -n 100 -o benchmarks/synthesized

run-demo:
	autoopt run benchmarks/custom/matrix_mult.c

# ── Cleanup ─────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	find . -type d -name "__pycache__" -delete 2>/dev/null; true
	find . -type f -name "*.exe" -delete 2>/dev/null; true
	find . -type f -name "*.s" -delete 2>/dev/null; true
	@echo "✓ Cleaned."

# ── Help ────────────────────────────────────────────────
help:
	@echo "AutoOpt Development Makefile"
	@echo ""
	@echo "  make install       Install autoopt CLI globally (pip install -e .)"
	@echo "  make dev-install   Install with dev extras"
	@echo "  make test          Run test suite"
	@echo "  make recommend     Demo: recommend flags for matrix_mult.c"
	@echo "  make analyze       Demo: structural analysis of matrix_mult.c"
	@echo "  make benchmark     Demo: sweep all flag-sets on matrix_mult.c"
	@echo "  make sequences     List all optimization sequences"
	@echo "  make run-demo      Demo: compile & run matrix_mult.c"
	@echo "  make synthesize    Generate 100 benchmark programs"
	@echo "  make clean         Remove build artifacts"
