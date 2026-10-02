"""
autoopt.analyzer
================
Unified Static & Structural Program Analysis Engine.
Integrates CFG, Loop, Memory Access, and Instruction Opcode Analyzers into
a canonical ProgramProfile used by AutoOpt ML predictors.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from autoopt.analyzer.cfg_analyzer import CFGAnalyzer, CFGMetrics
from autoopt.analyzer.loop_analyzer import LoopAnalyzer, LoopMetrics
from autoopt.analyzer.memory_analyzer import MemoryAnalyzer, MemoryMetrics
from autoopt.analyzer.inst_counter import InstructionCounter, InstructionMetrics


@dataclass
class ProgramProfile:
    source_file: str
    lines_of_code: int
    functions_count: int
    cfg: CFGMetrics
    loop: LoopMetrics
    memory: MemoryMetrics
    inst: InstructionMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Flattens all analysis metrics into a dictionary representation."""
        res: Dict[str, Any] = {
            "source_file": self.source_file,
            "lines_of_code": self.lines_of_code,
            "functions_count": self.functions_count,
        }
        for k, v in asdict(self.cfg).items():
            res[f"cfg_{k}"] = v
        for k, v in asdict(self.loop).items():
            res[f"loop_{k}"] = v
        for k, v in asdict(self.memory).items():
            res[f"mem_{k}"] = v
        for k, v in asdict(self.inst).items():
            res[f"inst_{k}"] = float(v) if isinstance(v, bool) else v
        return res

    def to_feature_vector(self, feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """Converts profile into numeric vector for machine learning model inference."""
        d = self.to_dict()
        if feature_columns is None:
            # Drop string columns
            cols = [k for k in d.keys() if k != "source_file"]
            return np.array([float(d[c]) for c in cols], dtype=np.float32)
        return np.array([float(d.get(c, 0.0)) for c in feature_columns], dtype=np.float32)


def analyze_program(source_file: Path, gxx: str = "g++") -> ProgramProfile:
    """
    Analyzes a C or C++ source file using the full AutoOpt analysis pipeline.
    Emits an unoptimized assembly file to extract ground-truth low-level metrics.
    """
    source_file = Path(source_file)
    src_text = source_file.read_text(encoding="utf-8", errors="replace")

    loc = len([l for l in src_text.splitlines() if l.strip()])
    func_count = max(1, len([l for l in src_text.splitlines() if "{" in l and "(" in l and not l.strip().startswith("//")]))

    # Generate unoptimized assembly for low-level analysis
    asm_text = ""
    with tempfile.NamedTemporaryFile(suffix=".s", delete=False) as tf:
        asm_path = tf.name
        tf.close()  # Close handle so compiler can write

    from autoopt.engine.compiler import AutoOptCompiler
    compiler = AutoOptCompiler()
    success, _ = compiler.emit_assembly(source_file, Path(asm_path), optimization="-O0")
    if success and Path(asm_path).exists():
        try:
            asm_text = Path(asm_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            asm_text = ""
        finally:
            try: Path(asm_path).unlink()
            except Exception: pass

    cfg_analyzer = CFGAnalyzer()
    loop_analyzer = LoopAnalyzer()
    memory_analyzer = MemoryAnalyzer()
    inst_counter = InstructionCounter()

    cfg_res = cfg_analyzer.analyze(src_text, asm_text)
    loop_res = loop_analyzer.analyze(src_text)
    mem_res = memory_analyzer.analyze(src_text, asm_text)
    inst_res = inst_counter.analyze(src_text, asm_text)

    return ProgramProfile(
        source_file=str(source_file),
        lines_of_code=loc,
        functions_count=func_count,
        cfg=cfg_res,
        loop=loop_res,
        memory=mem_res,
        inst=inst_res,
    )
