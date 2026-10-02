"""
autoopt.engine.compiler
=======================
Cross-platform compiler interface supporting GCC, G++, Clang, and Clang++.
Handles executable generation, binary size analysis, and assembly generation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class AutoOptCompiler:
    """Universal compiler driver for C and C++ programs."""

    def __init__(self, c_compiler: str = "gcc", cxx_compiler: str = "g++"):
        self.c_compiler = shutil.which(c_compiler) or c_compiler
        self.cxx_compiler = shutil.which(cxx_compiler) or cxx_compiler

    def get_compiler_for_file(self, source_path: Path) -> str:
        ext = source_path.suffix.lower()
        if ext in (".cpp", ".cxx", ".cc", ".cp", ".c++"):
            return self.cxx_compiler
        return self.c_compiler

    def compile(
        self,
        source_file: Path,
        output_binary: Path,
        flags: str = "-O2",
        extra_args: Optional[List[str]] = None,
        timeout: int = 60,
    ) -> Tuple[bool, str, int]:
        """
        Compiles source_file into output_binary with specified optimization flags.
        Returns: (success: bool, stderr: str, return_code: int)
        """
        compiler = self.get_compiler_for_file(source_file)
        # Ensure Windows .exe extension
        if sys.platform == "win32" and output_binary.suffix.lower() != ".exe":
            output_binary = output_binary.with_suffix(".exe")

        cmd = [compiler]
        if self.is_cpp(source_file):
            cmd.extend(["-std=c++17"])
        else:
            cmd.extend(["-std=c11"])

        if flags.strip():
            cmd.extend(flags.split())

        if extra_args:
            cmd.extend(extra_args)

        cmd.extend([str(source_file), "-o", str(output_binary), "-lm"])

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            success = (res.returncode == 0) and output_binary.exists()
            return success, res.stderr, res.returncode
        except subprocess.TimeoutExpired:
            return False, f"Compilation timed out after {timeout}s", -1
        except Exception as exc:
            return False, str(exc), -1

    def emit_assembly(
        self,
        source_file: Path,
        output_asm: Path,
        optimization: str = "-O0",
        timeout: int = 30,
    ) -> Tuple[bool, str]:
        """Compiles source code to assembly (.s) for structural analysis."""
        compiler = self.get_compiler_for_file(source_file)
        cmd = [compiler]
        if self.is_cpp(source_file):
            cmd.extend(["-std=c++17"])
        else:
            cmd.extend(["-std=c11"])

        cmd.extend([optimization, "-S", str(source_file), "-o", str(output_asm)])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return (res.returncode == 0) and output_asm.exists(), res.stderr
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def is_cpp(source_file: Path) -> bool:
        return source_file.suffix.lower() in (".cpp", ".cxx", ".cc", ".cp", ".c++")

    @staticmethod
    def get_binary_size(binary_path: Path) -> int:
        if sys.platform == "win32" and binary_path.suffix.lower() != ".exe":
            binary_path = binary_path.with_suffix(".exe")
        return binary_path.stat().st_size if binary_path.exists() else 0
