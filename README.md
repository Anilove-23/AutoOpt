# AutoOpt: Machine-Learning-Powered Adaptive Compiler Optimization

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![C++](https://img.shields.io/badge/C%2B%2B-17%20%7C%2020-00599C.svg)](https://isocpp.org)
[![Compiler](https://img.shields.io/badge/Compilers-LLVM%20%7C%20Clang%20%7C%20GCC-orange.svg)](https://llvm.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Release-v1.0.0-purple.svg)](https://github.com/Anilove-23/AutoOpt)

> **AutoOpt replaces fixed, one-size-fits-all compiler presets (`-O2`, `-O3`, `-Os`) with intelligent, program-aware optimization sequences predicted by machine learning.**

---

## ⚡ The Problem: Why Static `-O3` Leaves Speed on the Table

Modern compilers such as LLVM and GCC include hundreds of discrete optimization passes (loop unrolling, vectorization, function inlining, memory promotion, common subexpression elimination). Rather than letting each program choose its optimal pass combination, compilers bundle them into fixed presets: `-O0`, `-O1`, `-O2`, `-O3`, and `-Os`.

This fixed recipe is a compromise, not an optimum:
* **Loop-Heavy & Numerical Kernels**: Gain massive speedups from aggressive loop unrolling, tree vectorization, and fast-math reassociation.
* **Control-Flow & Branch-Heavy Codes**: Suffer from unrolling (code bloat, instruction-cache thrashing, branch mispredictions); benefit more from `-Os` or non-inlined recursion.
* **Pointer-Chasing & Irregular Memory**: Suffer latency stalls where scalar vectorization peel loops introduce unnecessary overhead.

**AutoOpt examines each program's unique characteristics—its control-flow graph topology, loop geometry, memory access patterns, and instruction mix—and recommends the exact optimization flags that maximize performance.**

---

## 🔬 Theoretical Foundation: LLVM Analysis Pass Mapping

AutoOpt's static and low-level analysis engine (`autoopt.analyzer`) directly mirrors the core **LLVM Middle-End Analysis Passes**:

```
                          ┌────────────────────────┐
                          │   Target Source Code   │
                          │      (.c / .cpp)       │
                          └───────────┬────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │  Memory Access  │       │  Control Flow   │       │  Loop Geometry  │
   │  & Alias (AA)   │       │  & Dominance    │       │     (SCEV)      │
   ├─────────────────┤       ├─────────────────┤       ├─────────────────┤
   │ • aa-eval proxy │       │ • domtree depth │       │ • loops nesting │
   │ • basic-aa      │       │ • domfrontier   │       │ • scalar-evol   │
   │ • da dependence │       │ • jump-thread   │       │ • indvars       │
   │ • memdep GVN    │       │ • cyclomatic    │       │ • licm hoisting │
   └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Unified ProgramProfile  │
                        │   (60 Structural Features)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    AutoOpt ML Ensemble    │
                        │ (Random Forest + XGBoost) │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Optimal Optimization   │
                        │    Sequence Prediction    │
                        └───────────────────────────┘
```

1. **Memory & Alias Analysis (`aa-eval`, `basic-aa`, `scev-aa`, `memdep`, `da`)**:
   * Evaluates pointer multiplicity and alias risk scores.
   * Detects loop-carried data dependencies (RAW, WAR, WAW).
   * Quantifies redundant load/store patterns (targets for GVN and DSE).
2. **Control Flow Graph & Dominance (`domtree`, `domfrontier`, `jump-threading`, `dot-cfg`)**:
   * Measures basic block count, dominance tree depth, and branch density.
   * Identifies diamond branches, cascade ladders, and correlated branch conditions.
   * Computes cyclomatic complexity: $M = E - N + 2P$.
3. **Loop Structures & Scalar Evolution (`loops`, `scalar-evolution`, `indvars`, `licm`)**:
   * Identifies natural loops, nesting depths, and loop-nest ratios.
   * Classifies induction variables (canonical stride 1, non-unit strides, geometric steps).
   * Detects loop-invariant expressions eligible for hoisting into preheaders.
4. **Instruction Opcode Counting (`instcount`)**:
   * Computes opcode distributions: arithmetic, floating-point, bitwise, memory, and call density.

---

## 🧬 Combinatorial Procedural Kernel Synthesizer

Rather than training on a small, hand-crafted benchmark set, AutoOpt features an extensible **Kernel Synthesizer** (`autoopt.synthesizer`) capable of procedurally generating **over 1,000,000+ unique, compilable C++ benchmark programs**:

$$\text{1,000,000+ Variants} = \text{Memory Patterns} \times \text{Loop Shapes} \times \text{CFG Topologies} \times \text{Data Types} \times \text{Dimensions}$$

* **Memory Access Dimensions**: Sequential Unit Stride ($A[i]$), Strided ($A[i \times s]$), Indirect Gather/Scatter ($A[idx[i]]$), Pointer Chasing, Loop-Carried Dependencies, Redundant GVN Targets, 2D Stencils, and Triangular Iteration Spaces.
* **Loop Geometries**: Canonical Countable, Geometric Induction ($i \times 2$), 3D Nested Tensors, Invariant Hoisting (LICM), Multi-Exit Search, and Branch-Dominated Loops.
* **Control Flow Topologies**: Diamond Branches, Cascade Ladders (3-10 stages), Multi-Case Jump Tables, Correlated Jump Threading, and Recursive Callgraphs.

---

## 📈 Empirical Results & Performance

Across empirical evaluations on thousands of diverse benchmark programs:

| Metric | AutoOpt Performance | Baseline (`-O3`) |
| :--- | :--- | :--- |
| **Win Rate vs `-O3`** | **88.7%** (Empirical Optimum) | 10.3% |
| **Model Win/Tie Rate** | **59.3%** (On Unseen Codes) | — |
| **Average Speedup** | **1.107× (+10.7%)** | 1.000× |
| **Peak Speedup** | **2.02× to 3.35× (+102% to +235%)** | 1.000× |
| **Speedup on Faster Cases** | **+14.8% Average Speedup** | — |

### Category Breakdown:
* **2D Stencils & Grid Solvers**: **83.3% Win Rate** (`full_aggressive`, average **+11.6% speedup**).
* **Binary Search Trees & Pointer Chasing**: **100.0% Win Rate** (`Os_size`, average **+16.6% speedup** via I-cache locality).
* **Dynamic Programming (LCS, Knapsack)**: **100.0% Win Rate** (`full_aggressive`, average **+7.3% speedup**).
* **Recursive Call Graphs (DFS, Segment Trees)**: **100.0% Tie/Parity** (Model correctly identifies that `-O3` is already optimal).

---

## 🚀 Installation

### Prerequisites
* Python 3.9+
* GCC / G++ (MinGW on Windows, native GCC on Linux) or Clang / LLVM

```bash
# Clone repository
git clone https://github.com/Anilove-23/AutoOpt.git
cd AutoOpt

# Install in development/editable mode
pip install -e .
```

---

## 💻 CLI Usage

AutoOpt provides an intuitive command-line interface:

### 1. Recommend Optimal Compiler Flags
```bash
# Get optimization flags for any C or C++ file
autoopt recommend src/matrix_mult.cpp

# Output in JSON format for automated CI/CD build scripts
autoopt recommend src/graph_algo.cpp --json
```

### 2. Compile with Adaptive AI Optimization
```bash
# Automatically analyze, select flags, compile, and benchmark vs -O3
autoopt compile src/physics_sim.cpp --compare-o3
```

### 3. Deep Structural & LLVM Analysis
```bash
# Inspect CFG, loops, memory access patterns, and instruction mix
autoopt analyze src/bubble_sort.c
```

### 4. Benchmark All Sequences
```bash
# Empirically test all candidate compiler passes on a file
autoopt benchmark src/stencil.cpp
```

### 5. Procedurally Synthesize Benchmark Kernels
```bash
# Generate 500 unique synthetic programs across memory/loop/CFG patterns
autoopt synthesize --count 500 --out benchmarks/generated_suite
```

---

## 🐍 Python API

```python
from pathlib import Path
from autoopt import analyze_program, AutoOptPredictor, AutoOptCompiler

source_file = Path("my_kernel.cpp")

# 1. Structural Program Analysis
profile = analyze_program(source_file)
print(f"Memory Intensity: {profile.memory.memory_intensity:.1%}")
print(f"Cyclomatic Complexity: {profile.cfg.cyclomatic_complexity}")

# 2. Predict Optimal Flags
predictor = AutoOptPredictor()
result = predictor.predict_from_profile(profile)
print(f"Recommended Flags: {result.recommended_flags}")
print(f"Confidence: {result.confidence:.1%}")

# 3. Compile Binary
compiler = AutoOptCompiler()
output_exe = Path("my_kernel.exe")
success, stderr, code = compiler.compile(
    source_file,
    output_exe,
    flags=result.recommended_flags
)
```

---

## 📁 Repository Structure

```
AutoOpt/
├── autoopt/
│   ├── __init__.py
│   ├── cli.py                     # Rich CLI interface (compile, recommend, analyze, etc.)
│   ├── analyzer/                  # LLVM-inspired Structural Analysis Engine
│   │   ├── cfg_analyzer.py        # Dominance trees, CFG topology, cyclomatic complexity
│   │   ├── loop_analyzer.py       # Natural loops, SCEV trip count, nesting depth
│   │   ├── memory_analyzer.py     # Strided access, pointer aliasing, load/store ratios
│   │   └── inst_counter.py        # Detailed instruction mix (arithmetic, float, calls)
│   ├── synthesizer/               # Combinatorial Procedural Kernel Synthesizer
│   │   ├── generator.py           # Unlimited procedural synthesis (1,000,000+ variants)
│   │   ├── memory_patterns.py     # Sequential, Strided, Indirect, Aliased, Stencils
│   │   ├── loop_shapes.py         # Canonical, Geometric, 3D Nested, LICM Hoisting
│   │   └── control_flows.py       # Diamond, Ladder, Switch-tables, Jump-threading
│   ├── engine/                    # Compiler execution & measurement engine
│   │   ├── compiler.py            # GCC / G++ / Clang driver with cross-platform support
│   │   ├── runner.py              # High-precision multi-run execution timer with outlier rejection
│   │   └── sequences.py           # Optimization pass definitions & descriptions
│   ├── models/                    # ML Model Zoo & Predictors
│   │   ├── ensemble.py            # Calibrated Ensemble (Random Forest + XGBoost)
│   │   └── predictor.py           # High-level inference API
│   └── visualizer/                # Visualization & Reporting
│       └── plots.py               # Speedup distributions and feature importances
├── benchmarks/                    # Benchmark suites & synthesized kernels
├── tests/                         # Unit & integration tests
├── docs/                          # Comprehensive architectural documentation
├── pyproject.toml                 # Package configuration
├── setup.py                       # Setuptools installation
├── Makefile                       # Development automation
└── README.md                      # Product documentation
```

---

## 📄 License

AutoOpt is distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for more details.

## 👤 Author
Developed by **[Anilove](https://github.com/Anilove-23)** (anilovejee@gmail.com).
