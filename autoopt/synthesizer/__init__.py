"""
autoopt.synthesizer
===================
Combinatorial Procedural Kernel Synthesizer package.
Provides infinite unique compilable C++ benchmark kernels spanning
memory access patterns, loop structures, and control flow topologies.
"""

from autoopt.synthesizer.generator import KernelSynthesizer, SynthesizedProgram
from autoopt.synthesizer.memory_patterns import MemoryPatternGenerator
from autoopt.synthesizer.loop_shapes import LoopShapeGenerator
from autoopt.synthesizer.control_flows import ControlFlowGenerator

__all__ = [
    "KernelSynthesizer",
    "SynthesizedProgram",
    "MemoryPatternGenerator",
    "LoopShapeGenerator",
    "ControlFlowGenerator",
]
