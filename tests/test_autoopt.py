"""
tests/test_autoopt.py
=====================
Comprehensive Unit Test Suite for AutoOpt.
Tests analyzers, synthesizer, engine, and predictor.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from autoopt.analyzer.cfg_analyzer import CFGAnalyzer
from autoopt.analyzer.loop_analyzer import LoopAnalyzer
from autoopt.analyzer.memory_analyzer import MemoryAnalyzer
from autoopt.analyzer.inst_counter import InstructionCounter
from autoopt.engine.sequences import OPTIMIZATION_SEQUENCES, get_sequence_by_id
from autoopt.synthesizer.generator import KernelSynthesizer
from autoopt.models.predictor import AutoOptPredictor


def test_cfg_analyzer():
    code = """
    int foo(int x) {
        if (x > 10) {
            return x * 2;
        } else if (x < 0) {
            return -x;
        } else {
            return x + 1;
        }
    }
    """
    analyzer = CFGAnalyzer()
    metrics = analyzer.analyze(code)
    assert metrics.branch_count >= 2
    assert metrics.cyclomatic_complexity >= 2


def test_loop_analyzer():
    code = """
    void loop_test(int* arr, int n) {
        for(int i=0; i<n; ++i) {
            for(int j=0; j<n; ++j) {
                arr[i] += arr[j];
            }
        }
    }
    """
    analyzer = LoopAnalyzer()
    metrics = analyzer.analyze(code)
    assert metrics.total_loops == 2
    assert metrics.max_nesting_depth == 2
    assert metrics.canonical_loops >= 1


def test_memory_analyzer():
    code = """
    void mem_test(int* a, int* b, int n) {
        for(int i=0; i<n; ++i) {
            a[i] = b[i * 4] + 1;
        }
    }
    """
    analyzer = MemoryAnalyzer()
    metrics = analyzer.analyze(code)
    assert metrics.pointer_arguments == 2
    assert metrics.aliasing_risk_score > 0.0


def test_inst_counter():
    code = """
    float compute(float x, float y) {
        float z = x * 2.5f + y * 1.5f;
        return z;
    }
    """
    counter = InstructionCounter()
    metrics = counter.analyze(code)
    assert metrics.float_ratio >= 0.0


def test_kernel_synthesizer():
    syn = KernelSynthesizer(seed=123)
    p = syn.synthesize_one(1)
    assert "int main()" in p.source_code
    assert p.dimension in ("memory", "loop", "control_flow")
    assert len(p.source_code) > 100


def test_optimization_sequences():
    assert len(OPTIMIZATION_SEQUENCES) >= 8
    seq = get_sequence_by_id(3)
    assert seq.name == "O3_aggressive"
    assert "-O3" in seq.gcc_flags


def test_predictor_fallback():
    predictor = AutoOptPredictor(model_path=Path("nonexistent_model.pkl"))
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as tf:
        tf.write(b"int main() { return 0; }")
        t_path = Path(tf.name)
    try:
        res = predictor.predict(t_path)
        assert res.recommended_flags != ""
        assert res.sequence_id in [0, 2, 3, 4, 5, 8, 10, 13, 19]
    finally:
        try: t_path.unlink()
        except: pass
