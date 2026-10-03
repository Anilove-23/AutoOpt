# AutoOpt Developer Guide

Welcome to the AutoOpt developer documentation. This guide provides instructions on how to set up the development environment, run tests, and contribute to the project.

## Development Environment Setup

To contribute to AutoOpt, you'll need Python 3.9+ and a working C/C++ compiler (GCC/G++ or Clang/Clang++).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anilove-23/AutoOpt.git
   cd AutoOpt
   ```

2. **Install in development mode (with extra dependencies):**
   ```bash
   make dev-install
   # Alternatively: pip install -e ".[dev]"
   ```
   This installs AutoOpt along with `pytest`, `shap`, and `optuna` for testing and ML tuning.

## Project Architecture Overview

AutoOpt is structured into several core modules:

*   **`autoopt.analyzer`**: Contains the LLVM-inspired static analysis engine. 
    *   `cfg_analyzer.py`: Control Flow Graph topology, dominance trees.
    *   `loop_analyzer.py`: Loop geometry, nesting depths, induction variables.
    *   `memory_analyzer.py`: Memory access patterns, alias analysis.
    *   `inst_counter.py`: Instruction opcodes and execution mix.
*   **`autoopt.synthesizer`**: Procedural C++ kernel generation (used for training the models on millions of code variants).
*   **`autoopt.engine`**: The compiler and execution runner logic.
    *   `compiler.py`: Wraps GCC/Clang, handles MinGW anomalies (like `-mconsole`), emits assemblies.
    *   `runner.py`: High-precision benchmark execution with outlier rejection.
    *   `sequences.py`: Defines the 12 LLVM/GCC optimization sequences (`O2_unroll_vec`, `O3_fast_math`, etc.).
*   **`autoopt.models`**: ML models and inference.
    *   `ensemble.py`: Random Forest + XGBoost prediction logic.
    *   `predictor.py`: High-level inference API.
*   **`autoopt.cli`**: The rich command-line interface implementation.

## Common Development Workflows

We use a standard `Makefile` to orchestrate common development tasks.

### Running Tests
We use `pytest` for unit testing.
```bash
make test
```

### Formatting and Linting
To check for syntax errors before committing:
```bash
make lint
```

### Cleaning Build Artifacts
During development, `.exe`, `.s`, and `__pycache__` files can clutter the workspace.
```bash
make clean
```

### Testing the CLI Locally
You can test the CLI changes without re-installing by running it as a Python module:
```bash
python -m autoopt --help
python -m autoopt run benchmarks/custom/matrix_mult.c
```
Alternatively, test specific components using the Make targets:
```bash
make run-demo
make analyze
make benchmark
```

## Adding New Optimization Sequences

If you want to add a new compiler optimization pass preset:
1. Open `autoopt/engine/sequences.py`.
2. Add a new `OptimizationSequence` object to the `OPTIMIZATION_SEQUENCES` list.
3. Ensure it defines both `gcc_flags` and `clang_flags`.
4. Assign a unique, non-colliding `id`.

## Retraining the Machine Learning Model

AutoOpt relies on pre-trained serialized `.pkl` models. If you modify the analyzer features, you must retrain the model.

1. **Synthesize a new dataset:**
   ```bash
   autoopt synthesize -n 100000 -o data/generated
   ```
2. **Extract features and measure runtimes:**
   *(Use the provided scripts in `scripts/`)*
   ```bash
   python scripts/build_dataset_large.py
   ```
3. **Train the models:**
   ```bash
   python autoopt/models/train_1m.py
   ```

## Contributing Guidelines

1. **Feature Branches:** Always create a new branch for your feature or bugfix (`git checkout -b feature/my-new-feature`).
2. **Tests:** Ensure existing tests pass (`make test`) and add new tests for new functionality in the `tests/` directory.
3. **Documentation:** Update `README.md` or this guide if CLI commands or APIs change.
4. **Pull Requests:** Submit PRs against the `main` branch with a clear description of the problem solved.

Thank you for contributing to AutoOpt!
