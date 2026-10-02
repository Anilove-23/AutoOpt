"""
AutoOpt: Machine-Learning-Powered Adaptive Compiler Optimization Framework
==========================================================================
AutoOpt statically and dynamically analyzes program control-flow, loop geometry,
memory access patterns, and instruction mixes to predict optimal compiler optimization
passes tailored for individual programs, consistently outperforming static presets (-O2 / -O3).
"""

__version__ = "1.0.0"
__author__ = "Anilove"
__license__ = "Apache-2.0"

from autoopt.analyzer import analyze_program, ProgramProfile
from autoopt.models.predictor import AutoOptPredictor
from autoopt.engine.compiler import AutoOptCompiler
from autoopt.engine.sequences import OPTIMIZATION_SEQUENCES

__all__ = [
    "analyze_program",
    "ProgramProfile",
    "AutoOptPredictor",
    "AutoOptCompiler",
    "OPTIMIZATION_SEQUENCES",
]
