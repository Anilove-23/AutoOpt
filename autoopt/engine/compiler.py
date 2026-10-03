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


def _is_mingw(compiler_path: str) -> bool:
    """Detect whether the compiler is MinGW GCC on Windows."""
    if sys.platform != "win32":
        return False
    try:
        r = subprocess.run(
            [compiler_path, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return "mingw" in r.stdout.lower() or "mingw" in r.stderr.lower()
    except Exception:
        return False


class AutoOptCompiler:
    """Universal compiler driver for C and C++ programs."""

    def __init__(self, c_compiler: str = "gcc", cxx_compiler: str = "g++"):
        self.c_compiler = shutil.which(c_compiler) or c_compiler
        self.cxx_compiler = shutil.which(cxx_compiler) or cxx_compiler
        # Detect MinGW once at construction time
        self._is_mingw = _is_mingw(self.c_compiler)

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

        On Windows + MinGW, automatically adds ``-mconsole`` to prevent the
        linker from looking for ``WinMain`` instead of ``main``.

        Returns: (success: bool, stderr: str, return_code: int)
        """
        source_file = Path(source_file)
        output_binary = Path(output_binary)
        compiler = self.get_compiler_for_file(source_file)

        # Guard: empty source file produces a confusing WinMain linker error.
        try:
            if source_file.stat().st_size == 0:
                return (
                    False,
                    f"Source file is empty: {source_file.name}\n"
                    "Add your C/C++ code to the file and try again.",
                    1,
                )
        except OSError:
            pass

        # Ensure Windows .exe extension
        if sys.platform == "win32" and output_binary.suffix.lower() != ".exe":
            output_binary = output_binary.with_suffix(".exe")

        cmd = [compiler]

        # Language standard
        if self.is_cpp(source_file):
            cmd.extend(["-std=c++17"])
        else:
            cmd.extend(["-std=c11"])

        # Optimization / user flags
        if flags.strip():
            cmd.extend(flags.split())

        # On MinGW/Windows: force console subsystem so the linker finds
        # main() instead of WinMain@16.  This is harmless for all normal
        # command-line programs and fixes the "undefined reference to WinMain"
        # error that occurs without it.
        if self._is_mingw:
            cmd.append("-mconsole")

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
            stderr = res.stderr
            if "undefined reference to `WinMain@16'" in stderr or "undefined reference to `WinMain" in stderr:
                stderr = (
                    "Error: The linker is looking for 'WinMain' but couldn't find it.\n"
                    "This usually happens because:\n"
                    "  1. Your file is missing a 'main()' function.\n"
                    "  2. Your 'main()' function has a typo (e.g., 'Main()' or 'mian()').\n"
                    "Please ensure you have a standard 'int main()' defined in your source code.\n\n"
                    f"Original linker output:\n{res.stderr.strip()}"
                )
            return success, stderr, res.returncode
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
        source_file = Path(source_file)
        compiler = self.get_compiler_for_file(source_file)
        cmd = [compiler]
        if self.is_cpp(source_file):
            cmd.extend(["-std=c++17"])
        else:
            cmd.extend(["-std=c11"])

        cmd.extend([optimization, "-S", str(source_file), "-o", str(output_asm)])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return (res.returncode == 0) and Path(output_asm).exists(), res.stderr
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def is_cpp(source_file: Path) -> bool:
        return source_file.suffix.lower() in (".cpp", ".cxx", ".cc", ".cp", ".c++")

    @staticmethod
    def get_binary_size(binary_path: Path) -> int:
        binary_path = Path(binary_path)
        if sys.platform == "win32" and binary_path.suffix.lower() != ".exe":
            binary_path = binary_path.with_suffix(".exe")
        return binary_path.stat().st_size if binary_path.exists() else 0
