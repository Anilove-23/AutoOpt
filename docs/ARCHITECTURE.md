# AutoOpt Architecture & LLVM Analysis Mapping

AutoOpt is a machine-learning-powered compiler optimization engine that replaces static, one-size-fits-all heuristics (`-O2`, `-O3`, `-Os`) with learned, program-aware optimization sequences tailored to individual source codes.

---

## 1. The Core Compiler Problem

Modern compilers (LLVM, GCC) include hundreds of discrete optimization passes (inlining, vectorization, loop unrolling, dead code elimination, memory promotion). However, compilers bundle them into fixed presets:

$$\text{Fixed Presets: } -O0, -O1, -O2, -O3, -Os$$

This fixed bundling is a fundamental compromise:
- **Numerical & Linear Algebra Kernels**: Benefit from aggressive loop unrolling, fast-math reassociation, and SIMD vectorization.
- **Branch-Heavy & Graph Traversal Codes**: Hurt by unrolling (code bloat, instruction-cache thrashing, branch mispredictions); benefit more from `-Os` or non-inlined recursion.
- **Pointer-Chasing & Irregular Memory**: Suffer latency stalls where scalar vectorization peel loops introduce unnecessary overhead.

**AutoOpt bridges this gap**: It extracts static structural features directly correlated with compiler pass profitability and predicts the sequence that minimizes execution time.

---

## 2. Theoretical Grounding: LLVM Analysis Pass Mapping

AutoOpt's feature extractor (`autoopt/analyzer/`) directly proxies the primary LLVM Middle-End Analysis Passes:

### A. Memory Access & Alias Analysis (LLVM `aa-eval`, `basic-aa`, `scev-aa`, `memdep`, `da`)
- **`aa-eval` (Exhaustive Alias Analysis Precision Evaluator)**: AutoOpt computes the pointer multiplicity and aliasing ambiguity score across scope-local pointer arguments.
- **`basic-aa` & `globals-aa`**: Distinguishes local stack allocations (`alloca`) from global array references.
- **`da` (Dependence Analysis)**: Classifies loop-carried dependencies (RAW, WAR, WAW) that restrict vectorizer safety.
- **`memdep` (Memory Dependence Analysis)**: Measures redundant load/store patterns (candidates for Global Value Numbering / Partial Redundancy Elimination).

### B. Control Flow Graph Topology & Dominance (LLVM `domtree`, `domfrontier`, `dot-cfg`)
- **`domtree` & `domfrontier`**: AutoOpt computes basic block counts, dominance tree depths, and dominance frontiers.
- **`jump-threading` & `simplifycfg`**: Detects correlated conditional predicates across basic blocks and branch ladders.
- **Cyclomatic Complexity**: Measures graph edges $E$, nodes $N$, and connected components $P$:
  $$M = E - N + 2P$$

### C. Loop Geometry & Scalar Evolution (LLVM `loops`, `scalar-evolution`, `indvars`, `licm`)
- **`loops` (Natural Loop Info)**: Loop counts, nested loop ratios, and maximum loop nesting depth.
- **`scalar-evolution` (SCEV)**: Classifies induction variables into:
  - Canonical unit step ($i \leftarrow i + 1$)
  - Non-unit strided step ($i \leftarrow i + \text{stride}$)
  - Geometric / exponential step ($i \leftarrow i \times 2$)
- **`licm` (Loop Invariant Code Motion)**: Measures density of loop-invariant expressions eligible for hoisting into the loop preheader.

### D. Instruction Counting (LLVM `instcount`)
- Counts opcode distributions: Arithmetic, Floating-Point, Bitwise, Calls, and Jumps.
- Computes normalized ratios:
  $$\text{Float Ratio} = \frac{\text{Float Ops}}{\text{Total Ops}}, \quad \text{Memory Intensity} = \frac{\text{Loads} + \text{Stores}}{\text{Total Ops}}$$

---

## 3. Combinatorial Procedural Kernel Synthesizer

Rather than training on a small set of fixed hand-crafted algorithms, AutoOpt's Synthesizer (`autoopt/synthesizer/`) generates up to **1,000,000+ unique, compilable C++ kernels** by combinatorially crossing:

1. **Memory Patterns**:
   - `SequentialUnitStride` ($A[i]$)
   - `StridedNonUnit` ($A[i \times s]$)
   - `IndirectGatherScatter` ($A[idx[i]]$)
   - `PointerChasing` (Linked structures)
   - `LoopCarriedRAW` (True data dependencies)
   - `RedundantLoadsStores` (GVN / DSE targets)
   - `MultiDimStencil` (2D/3D neighborhood stencils)
   - `TriangularMatrix` (Non-rectangular iteration spaces)

2. **Loop Shapes**:
   - Canonical countable loops
   - Geometric / logarithmic induction loops
   - Multi-exit loops with early break predicates
   - 3D nested loop tensors
   - Invariant-heavy hoisting candidates
   - Branch-dominated loops (loop-unswitch targets)

3. **Control Flow Topologies**:
   - Diamond symmetric if-else
   - Deep cascade ladders (3-10 branch stages)
   - Multi-case switch jump-tables
   - Correlated jump-threading targets
   - Recursive call hierarchies

$$\text{Search Space: } 8 \text{ memory patterns} \times 6 \text{ loop shapes} \times 5 \text{ CFG topologies} \times 5 \text{ datatypes} \times 12 \text{ sizes} \dots > 10^6 \text{ variants}$$

---

## 4. Optimization Sequence Search Space

AutoOpt evaluates against 12 compiler pass configurations:

| ID | Preset Name | Compiler Flags | Primary Target |
| :--- | :--- | :--- | :--- |
| `0` | `O0_baseline` | `-O0` | Baseline reference |
| `2` | `O2_standard` | `-O2` | Standard production default |
| `3` | `O3_aggressive` | `-O3` | LLVM / GCC default aggressive preset |
| `4` | `Os_size` | `-Os` | Instruction cache locality (pointer chasing, branches) |
| `5` | `O2_unroll_vec` | `-O2 -funroll-loops -ftree-vectorize` | Clean countable loops |
| `8` | `O2_no_inline` | `-O2 -fno-inline` | Deep recursion preventing stack frame bloat |
| `10` | `O3_combine` | `-O3 -funroll-loops -finline-functions -ftree-vectorize` | Compute kernels with function call chains |
| `12` | `O3_no_vec` | `-O3 -fno-tree-vectorize` | Control-heavy loops avoiding vector peel overhead |
| `13` | `O2_fast_math` | `-O2 -ffast-math` | Numerical floating-point matrix and stencil math |
| `17` | `O1_unroll` | `-O1 -funroll-loops` | Tight loops avoiding register spilling |
| `18` | `O2_gcse_pre` | `-O2 -fgcse` | Redundant memory load elimination |
| `19` | `full_aggressive` | `-O3 -ffast-math -funroll-loops -finline-functions -ftree-vectorize` | Pure numerical compute throughput |

---

## 5. Machine Learning Ensemble Architecture

AutoOpt uses a calibrated ensemble of:
- **Random Forest (500 Estimators)**: Handles high-dimensional non-linear feature interactions and provides robust Gini impurity feature importance.
- **XGBoost (Optuna Hyperparameter Tuned)**: Gradient boosted trees with multi-class softmax loss and regularized tree depth.
- **Confidence Calibration**: Outputs class probability distributions across all candidate optimization sequences.
