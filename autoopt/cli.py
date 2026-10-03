"""
autoopt.cli
===========
Production Command-Line Interface for AutoOpt.

Usage (installed):
  autoopt <command> [options]

Usage (without install):
  python -m autoopt <command> [options]
  python autoopt/cli.py <command> [options]

Commands:
  recommend   Predict optimal compiler flags for any C/C++ source file
  compile     Analyze + compile with AI-recommended flags (like gcc but smarter)
  run         Compile & immediately run a program with AI-optimized flags
  analyze     Emit LLVM-inspired structural analysis (CFG, loops, memory, ISA)
  benchmark   Empirically sweep all optimization sequences and compare speeds
  synthesize  Procedurally generate N unique benchmark kernel programs
  sequences   List all supported optimization sequences with flags
  init        Scaffold a new AutoOpt-ready C/C++ project
  version     Display version, compiler toolchain, and environment info

Quick start (just like using gcc):
  # Instead of:   gcc -O3 my_prog.c -o my_prog
  # Use:          autoopt compile my_prog.c -o my_prog
  #
  # Instead of:   gcc -O3 my_prog.c -o my_prog && ./my_prog
  # Use:          autoopt run my_prog.c
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------
# Ensure project root is on sys.path when running as script
# ----------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoopt import __version__
from autoopt.analyzer import analyze_program
from autoopt.engine.compiler import AutoOptCompiler
from autoopt.engine.runner import AutoOptRunner
from autoopt.engine.sequences import OPTIMIZATION_SEQUENCES, get_sequence_by_id
from autoopt.models.predictor import AutoOptPredictor

# ----------------------------------------------------------
# Terminal color helpers (no external deps)
# ----------------------------------------------------------

def _supports_color() -> bool:
    """Return True if the terminal likely supports ANSI color codes."""
    if sys.platform == "win32":
        # Windows Terminal, VSCode, and modern cmd support ANSI
        return os.environ.get("WT_SESSION") is not None or \
               os.environ.get("TERM_PROGRAM") is not None or \
               "ANSICON" in os.environ or \
               os.environ.get("COLORTERM") in ("truecolor", "24bit")
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_COLOR = _supports_color()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text

def green(s: str)  -> str: return _c("32;1", s)
def yellow(s: str) -> str: return _c("33;1", s)
def red(s: str)    -> str: return _c("31;1", s)
def cyan(s: str)   -> str: return _c("36;1", s)
def bold(s: str)   -> str: return _c("1", s)
def dim(s: str)    -> str: return _c("2", s)


# ----------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------

def _resolve_source(path_str: str, check_content: bool = True) -> Path:
    p = Path(path_str).resolve()
    if not p.exists():
        print(red(f"Error: file not found: {p}"), file=sys.stderr)
        sys.exit(1)

    if check_content:
        content = p.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            print(red(f"Error: source file is empty: {p.name}"), file=sys.stderr)
            print(dim("       Add your C/C++ code to the file and try again."), file=sys.stderr)
            sys.exit(1)
        # Warn if no main() found (compile/run would fail at link time)
        if "main" not in content:
            print(yellow(f"Warning: no 'main' function found in {p.name}"), file=sys.stderr)
            print(dim("         Compilation may fail at the linker step (undefined reference to WinMain)."), file=sys.stderr)

    return p


def _make_predictor(model_arg: Optional[str]) -> AutoOptPredictor:
    model_path = Path(model_arg) if model_arg else None
    return AutoOptPredictor(model_path=model_path)


def _resolve_output(source: Path, output_arg: Optional[str]) -> Path:
    if output_arg:
        out = Path(output_arg)
    else:
        out = source.with_suffix("")
    if sys.platform == "win32" and out.suffix.lower() != ".exe":
        out = out.with_suffix(".exe")
    return out


def _print_sep(char: str = "-", width: int = 64):
    try:
        print(dim(char * width))
    except UnicodeEncodeError:
        print("-" * width)


# ----------------------------------------------------------
# recommend
# ----------------------------------------------------------

def cmd_recommend(args):
    """Predict the best compiler flags for a C/C++ source file."""
    source = _resolve_source(args.source)
    predictor = _make_predictor(args.model)

    print(f"Analyzing {cyan(source.name)} ...")
    profile = analyze_program(source, gxx=args.compiler or "g++")
    res = predictor.predict_from_profile(profile)

    compiler_bin = AutoOptCompiler().get_compiler_for_file(source)
    binary_name = source.stem + (".exe" if sys.platform == "win32" else "")
    comp_cmd = f"{compiler_bin} {res.recommended_flags} {source.name} -o {binary_name}"

    if args.as_json:
        out = {
            "source_file": str(source),
            "recommended_flags": res.recommended_flags,
            "sequence_id": res.sequence_id,
            "sequence_name": res.sequence_name,
            "confidence": round(res.confidence, 4),
            "compile_command": comp_cmd,
            "model": res.model_name,
            "metrics": profile.to_dict(),
        }
        print(json.dumps(out, indent=2))
    else:
        _print_sep("=")
        print(bold("  AutoOpt: AI Compiler Optimization Recommendation"))
        _print_sep("=")
        print(f"  Source File  : {cyan(source.name)}")
        print(f"  Sequence     : {green(res.sequence_name)}  (ID: {res.sequence_id})")
        print(f"  Confidence   : {green(f'{res.confidence * 100:.1f}%')}")
        print(f"  Model        : {dim(res.model_name)}")
        _print_sep()
        print(bold("  Recommended compile command:"))
        print(f"    {green(comp_cmd)}")
        _print_sep("=")
        print()


# ----------------------------------------------------------
# compile
# ----------------------------------------------------------

def cmd_compile(args):
    """
    Drop-in GCC replacement: analyze -> choose flags -> compile.

    Examples:
      autoopt compile my_prog.c                   # auto flags
      autoopt compile my_prog.c -o my_prog        # custom output name
      autoopt compile my_prog.c --compare-o3      # benchmark against -O3
      autoopt compile my_prog.c --flags '-O2'     # override with manual flags
    """
    source = _resolve_source(args.source)
    out_binary = _resolve_output(source, args.output)
    compiler = AutoOptCompiler(
        c_compiler=args.compiler or "gcc",
        cxx_compiler=args.cxx_compiler or "g++",
    )

    if args.flags:
        # Manual flag override -- behave exactly like gcc
        flags = args.flags
        seq_name = "manual"
        confidence = 1.0
    else:
        # AI-powered flag prediction
        print(f"  {dim('Analyzing')} {cyan(source.name)} ...")
        predictor = _make_predictor(args.model)
        profile = analyze_program(source, gxx=args.cxx_compiler or "g++")
        res = predictor.predict_from_profile(profile)
        flags = res.recommended_flags
        seq_name = res.sequence_name
        confidence = res.confidence
        print(f"  {dim('Predicted')}  : {green(seq_name)}  ({confidence*100:.0f}% confidence)")

    print(f"  {dim('Compiling')} -> {cyan(str(out_binary))}")
    print(f"  {dim('Flags')}     : {yellow(flags)}")

    t0 = time.perf_counter()
    success, stderr, code = compiler.compile(source, out_binary, flags=flags,
                                              extra_args=args.extra or [])
    comp_time = time.perf_counter() - t0

    if not success:
        print(red(f"\nCompilation FAILED (exit {code}):"), file=sys.stderr)
        print(stderr, file=sys.stderr)
        sys.exit(code)

    sz = compiler.get_binary_size(out_binary)
    print(f"  {green('[OK] Built')} {out_binary.name}  "
          f"{dim(f'({sz:,} bytes, {comp_time:.2f}s)')}")

    if args.compare_o3 or args.run:
        runner = AutoOptRunner(runs=args.runs)

        if args.run:
            print(f"\n{bold('Running')} {out_binary.name} ...\n")
            t_run, _, stdout = runner.benchmark(out_binary)
            if stdout:
                print(stdout, end="")
            if t_run:
                print(f"\n{dim(f'  Exit OK  -- {t_run:.4f}s elapsed')}")

        if args.compare_o3:
            _benchmark_vs_o3(source, out_binary, compiler, runner, flags)

    print()


# ----------------------------------------------------------
# run
# ----------------------------------------------------------

def cmd_run(args):
    """
    Compile with AI-optimal flags and immediately run the program.

    Works just like:   gcc -O3 prog.c -o prog && ./prog
    But smarter:       autoopt run prog.c
    """
    source = _resolve_source(args.source)
    compiler = AutoOptCompiler(
        c_compiler=args.compiler or "gcc",
        cxx_compiler=args.cxx_compiler or "g++",
    )

    if args.flags:
        flags = args.flags
        seq_name = "manual"
    else:
        predictor = _make_predictor(args.model)
        profile = analyze_program(source, gxx=args.cxx_compiler or "g++")
        res = predictor.predict_from_profile(profile)
        flags = res.recommended_flags
        seq_name = res.sequence_name
        print(f"{dim('AutoOpt ->')} {green(seq_name)}  {dim(f'({flags})')}")

    # Compile to a temp directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out_binary = Path(tmpdir) / (source.stem + (".exe" if sys.platform == "win32" else ""))
        success, stderr, code = compiler.compile(source, out_binary, flags=flags)
        if not success:
            print(red(f"Compilation failed (exit {code}):"), file=sys.stderr)
            print(stderr, file=sys.stderr)
            sys.exit(code)

        # Run program -- pass any extra positional args to it
        run_args = args.prog_args or []
        cmd = [str(out_binary)] + run_args
        try:
            result = subprocess.run(cmd, timeout=args.timeout)
            sys.exit(result.returncode)
        except subprocess.TimeoutExpired:
            print(red(f"\nProgram timed out after {args.timeout}s"), file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            sys.exit(130)


# ----------------------------------------------------------
# analyze
# ----------------------------------------------------------

def cmd_analyze(args):
    """Emit a deep LLVM-inspired structural analysis report."""
    source = _resolve_source(args.source)
    profile = analyze_program(source, gxx=args.compiler or "g++")
    d = profile.to_dict()

    if args.as_json:
        print(json.dumps(d, indent=2))
        return

    W = 68
    _print_sep("=", W)
    print(bold(f"  AutoOpt Analysis: {source.name}"))
    _print_sep("=", W)
    print(f"  Lines of Code   : {profile.lines_of_code}")
    print(f"  Functions       : {profile.functions_count}")
    _print_sep("-", W)
    print(bold("  Control Flow Graph & Dominance"))
    print(f"    Basic Blocks      : {profile.cfg.basic_blocks}")
    print(f"    Cyclomatic Cmplx  : {yellow(str(profile.cfg.cyclomatic_complexity))}")
    print(f"    Branches          : {profile.cfg.branch_count}  "
          f"{dim(f'(density {profile.cfg.branch_density:.2f})')}")
    print(f"    Conditional Jumps : {profile.cfg.conditional_jumps}")
    print(f"    Switch Statements : {profile.cfg.switch_statements}")
    print(f"    Diamond Branches  : {profile.cfg.diamond_branches}")
    _print_sep("-", W)
    print(bold("  Loop Structure & Induction Variables (SCEV)"))
    print(f"    Total Loops       : {yellow(str(profile.loop.total_loops))}")
    print(f"    Max Nesting Depth : {profile.loop.max_nesting_depth}")
    print(f"    Canonical Loops   : {profile.loop.canonical_loops}")
    print(f"    Strided Loops     : {profile.loop.strided_loops}")
    print(f"    Geometric Loops   : {profile.loop.geometric_loops}")
    _print_sep("-", W)
    print(bold("  Memory Access & Alias Analysis (aa-eval)"))
    print(f"    Memory Ops        : {profile.memory.total_memory_ops}"
          f"  {dim(f'(loads={profile.memory.load_operations}, stores={profile.memory.store_operations})')}")
    print(f"    Memory Intensity  : {yellow(f'{profile.memory.memory_intensity:.1%}')}")
    print(f"    Pointer Deref     : {profile.memory.pointer_dereferences}")
    print(f"    Alias Risk Score  : {profile.memory.aliasing_risk_score:.2f}")
    print(f"    Contiguous Ratio  : {profile.memory.contiguous_access_ratio:.1%}")
    _print_sep("-", W)
    print(bold("  Instruction Mix (instcount)"))
    print(f"    Total Instructions: {profile.inst.total_instructions}")
    print(f"    Arithmetic        : {profile.inst.arithmetic_instructions}"
          f"  {dim(f'({profile.inst.arithmetic_mix_ratio:.1%})')}")
    print(f"    Floating Point    : {profile.inst.floating_point_instructions}"
          f"  {dim(f'({profile.inst.float_ratio:.1%})')}")
    print(f"    Calls             : {profile.inst.call_instructions}"
          f"  {dim(f'({profile.inst.call_density:.1%})')}")
    _print_sep("=", W)
    print()


# ----------------------------------------------------------
# benchmark
# ----------------------------------------------------------

def cmd_benchmark(args):
    """Sweep all optimization sequences and compare empirical runtimes."""
    source = _resolve_source(args.source)
    compiler = AutoOptCompiler(
        c_compiler=args.compiler or "gcc",
        cxx_compiler=args.cxx_compiler or "g++",
    )
    runner = AutoOptRunner(runs=args.runs)

    print(f"\nBenchmarking {cyan(source.name)} across all optimization sequences ...\n")

    # Baseline: -O3
    o3_seq = get_sequence_by_id(3)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_o3 = Path(tmpdir) / ("baseline_O3" + (".exe" if sys.platform == "win32" else ""))
        ok, err, _ = compiler.compile(source, tmp_o3, flags=o3_seq.gcc_flags)
        t_o3 = None
        if ok:
            t_o3, _, _ = runner.benchmark(tmp_o3)

        col_w = [22, 46, 12, 12, 10]
        header = (
            f"{'Sequence':<{col_w[0]}} {'Flags':<{col_w[1]}} "
            f"{'Size(B)':>{col_w[2]}} {'Time(s)':>{col_w[3]}} {'vs -O3':>{col_w[4]}}"
        )
        _print_sep("-", sum(col_w) + 4)
        print(bold(header))
        _print_sep("-", sum(col_w) + 4)

        results = []
        for seq in OPTIMIZATION_SEQUENCES:
            tmp_exe = Path(tmpdir) / (f"seq_{seq.id}" + (".exe" if sys.platform == "win32" else ""))
            ok, _, _ = compiler.compile(source, tmp_exe, flags=seq.gcc_flags)
            if not ok:
                print(f"{seq.name:<{col_w[0]}} {seq.gcc_flags:<{col_w[1]}} "
                      f"{'N/A':>{col_w[2]}} {'FAIL':>{col_w[3]}} {'--':>{col_w[4]}}")
                continue

            sz = compiler.get_binary_size(tmp_exe)
            tm, _, _ = runner.benchmark(tmp_exe)

            if tm is not None and t_o3 is not None:
                sp = t_o3 / tm
                sp_str = f"{sp:.3f}x"
                color_fn = green if sp > 1.005 else (red if sp < 0.995 else dim)
            else:
                sp = None
                sp_str = "N/A"
                color_fn = dim

            tm_str = f"{tm:.6f}" if tm is not None else "FAIL"
            flags_short = seq.gcc_flags if len(seq.gcc_flags) <= col_w[1] - 2 else seq.gcc_flags[:col_w[1]-5] + "..."
            line = (
                f"{seq.name:<{col_w[0]}} {flags_short:<{col_w[1]}} "
                f"{sz:>{col_w[2]},} {tm_str:>{col_w[3]}} {sp_str:>{col_w[4]}}"
            )
            print(color_fn(line) if sp and sp != 1.0 else line)
            results.append((seq.name, sz, tm, sp))

        _print_sep("-", sum(col_w) + 4)

        # Summary
        winners = [(n, s, t, sp) for n, s, t, sp in results if sp and sp > 1.005]
        if winners:
            best = max(winners, key=lambda x: x[3])
            print(f"\n  {green('Best sequence:')} {best[0]}  "
                  f"{green(f'{best[3]:.3f}x')} faster than -O3")
        print()


# ----------------------------------------------------------
# synthesize
# ----------------------------------------------------------

def cmd_synthesize(args):
    """Procedurally generate unique C++ benchmark kernel programs."""
    from autoopt.synthesizer.generator import KernelSynthesizer
    out_dir = Path(args.out)
    count = args.count
    print(f"Synthesizing {cyan(f'{count:,}')} programs -> {out_dir} ...")
    syn = KernelSynthesizer(seed=args.seed)
    t0 = time.perf_counter()
    progs = syn.synthesize_batch(count=count, output_dir=out_dir)
    elapsed = time.perf_counter() - t0
    rate = len(progs) / elapsed if elapsed > 0 else 0
    print(green(f"[OK] Generated {len(progs):,} programs in {elapsed:.2f}s  ({rate:.0f} prog/s)"))
    print(f"  Output: {out_dir.resolve()}")


# ----------------------------------------------------------
# sequences
# ----------------------------------------------------------

def cmd_sequences(args):
    """List all supported compiler optimization sequences."""
    if args.as_json:
        data = [
            {
                "id": s.id,
                "name": s.name,
                "gcc_flags": s.gcc_flags,
                "clang_flags": s.clang_flags,
                "description": s.description,
                "primary_benefit": s.primary_benefit,
            }
            for s in OPTIMIZATION_SEQUENCES
        ]
        print(json.dumps(data, indent=2))
        return

    _print_sep("=")
    print(bold("  AutoOpt Optimization Sequences"))
    _print_sep("=")
    for seq in OPTIMIZATION_SEQUENCES:
        print(f"  {cyan(f'[{seq.id:2d}]')} {green(seq.name)}")
        print(f"       GCC   : {yellow(seq.gcc_flags)}")
        print(f"       Clang : {dim(seq.clang_flags)}")
        print(f"       Use   : {seq.primary_benefit}")
        _print_sep("-")
    print()


# ----------------------------------------------------------
# init
# ----------------------------------------------------------

_MAIN_C_TEMPLATE = '''\
#include <stdio.h>
#include <stdlib.h>

/*
 * {name}: AutoOpt-ready C project
 * Build: autoopt compile src/main.c -o {name}
 * Run:   autoopt run src/main.c
 */

int main(int argc, char *argv[]) {{
    printf("Hello from {name}!\\n");
    return 0;
}}
'''

_MAIN_CPP_TEMPLATE = '''\
#include <iostream>
#include <vector>
#include <string>

/*
 * {name}: AutoOpt-ready C++ project
 * Build: autoopt compile src/main.cpp -o {name}
 * Run:   autoopt run src/main.cpp
 */

int main(int argc, char* argv[]) {{
    std::cout << "Hello from {name}!" << std::endl;
    return 0;
}}
'''

_MAKEFILE_TEMPLATE = '''\
# AutoOpt Makefile -- generated by `autoopt init`
NAME    := {name}
SRC     := src/main.{ext}

.PHONY: all run analyze benchmark clean

all:
\tautoopt compile $(SRC) -o $(NAME)

run:
\tautoopt run $(SRC)

analyze:
\tautoopt analyze $(SRC)

benchmark:
\tautoopt benchmark $(SRC)

recommend:
\tautoopt recommend $(SRC)

clean:
\trm -f $(NAME) $(NAME).exe
'''

_GITIGNORE_TEMPLATE = '''\
# AutoOpt build artifacts
*.exe
*.o
*.s
__pycache__/
*.pyc
.autoopt_cache/
'''

def cmd_init(args):
    """Scaffold a new AutoOpt-ready C/C++ project directory."""
    name = args.name
    lang = args.lang  # "c" or "cpp"
    target = Path(name)

    if target.exists() and not args.force:
        print(red(f"Directory '{name}' already exists. Use --force to overwrite."), file=sys.stderr)
        sys.exit(1)

    ext = "cpp" if lang == "cpp" else "c"
    template = _MAIN_CPP_TEMPLATE if lang == "cpp" else _MAIN_C_TEMPLATE

    dirs = [target / "src", target / "include", target / "tests"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    (target / "src" / f"main.{ext}").write_text(
        template.format(name=name), encoding="utf-8"
    )
    (target / "Makefile").write_text(
        _MAKEFILE_TEMPLATE.format(name=name, ext=ext), encoding="utf-8"
    )
    (target / ".gitignore").write_text(_GITIGNORE_TEMPLATE, encoding="utf-8")
    (target / "README.md").write_text(
        f"# {name}\n\nBuilt with [AutoOpt](https://github.com/Anilove-23/AutoOpt) AI compiler optimization.\n\n"
        f"```bash\n# Build with AI-optimal flags\nautoopt compile src/main.{ext} -o {name}\n\n"
        f"# Run directly\nautoopt run src/main.{ext}\n\n"
        f"# See what flags AutoOpt chose\nautoopt recommend src/main.{ext}\n```\n",
        encoding="utf-8",
    )

    print(green(f"[OK] Created project: {target.resolve()}"))
    print(f"  {dim('Files created:')}")
    print(f"    src/main.{ext}  -- main source file")
    print(f"    Makefile       -- build automation")
    print(f"    .gitignore     -- git ignore rules")
    print(f"    README.md      -- project readme")
    print()
    print(f"  {bold('Next steps:')}")
    print(f"    {cyan(f'cd {name}')}")
    print(f"    {cyan(f'autoopt run src/main.{ext}')}")
    print()


# ----------------------------------------------------------
# version
# ----------------------------------------------------------

def cmd_version(args):
    """Print version and toolchain environment info."""
    compiler = AutoOptCompiler()
    print(f"{bold('AutoOpt')} v{__version__} -- ML-Powered Adaptive Compiler")

    def _tool_version(cmd: str) -> str:
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
            return r.stdout.split("\n")[0].strip() if r.returncode == 0 else dim("not found")
        except Exception:
            return dim("not found")

    print(f"  GCC    : {_tool_version(compiler.c_compiler)}")
    print(f"  G++    : {_tool_version(compiler.cxx_compiler)}")
    clang = shutil.which("clang")
    if clang:
        print(f"  Clang  : {_tool_version('clang')}")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  OS     : {platform.system()} {platform.release()} ({platform.machine()})")

    # Show available model files
    models_dir = _ROOT / "models"
    model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
    if model_files:
        print(f"  Models : {', '.join(m.name for m in model_files)}")
    else:
        print(f"  Models : {dim('none found -- run ml/train.py to train')}")


# ----------------------------------------------------------
# Internal helper: compare to -O3
# ----------------------------------------------------------

def _benchmark_vs_o3(source: Path, out_binary: Path,
                     compiler: AutoOptCompiler, runner: AutoOptRunner,
                     flags: str) -> None:
    import tempfile
    print(f"\n  {bold('Benchmarking')} ({runner.runs} runs each) ...")
    t_rec, _, _ = runner.benchmark(out_binary)

    with tempfile.NamedTemporaryFile(
        suffix=".exe" if sys.platform == "win32" else "",
        delete=False,
    ) as tf:
        tmp_o3 = Path(tf.name)
    try:
        compiler.compile(source, tmp_o3, flags="-O3")
        t_o3, _, _ = runner.benchmark(tmp_o3)
    finally:
        try:
            tmp_o3.unlink()
        except Exception:
            pass

    if t_rec and t_o3:
        sp = t_o3 / t_rec
        pct = (sp - 1.0) * 100.0
        status_fn = green if sp > 1.005 else (red if sp < 0.995 else dim)
        status = "FASTER [OK]" if sp > 1.005 else ("SLOWER [X]" if sp < 0.995 else "EQUAL  ~")
        print(f"  AutoOpt ({flags[:30]}): {t_rec:.6f}s")
        print(f"  -O3 baseline              : {t_o3:.6f}s")
        print(f"  {status_fn(f'Speedup: {sp:.4f}x  ({pct:+.2f}%)  -> {status}')}")
    else:
        print(dim("  (timing unavailable -- program may not produce measurable runtime)"))


# ----------------------------------------------------------
# Argument parser
# ----------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoopt",
        description=bold("AutoOpt: ML-Powered Adaptive Compiler Optimization"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples -- use AutoOpt like you use gcc:
  autoopt compile myfile.c                 # AI-optimal flags, auto output name
  autoopt compile myfile.c -o myapp        # specify output binary
  autoopt run myfile.c                     # compile & run immediately
  autoopt run myfile.cpp -- --input data   # pass args to your program
  autoopt recommend myfile.c               # see the recommended flags only
  autoopt analyze myfile.c                 # deep structural analysis
  autoopt benchmark myfile.c               # sweep all 12 flag-sets
  autoopt sequences                        # list all flag-sets
  autoopt init myproject --lang cpp        # scaffold a new C++ project
  autoopt version                          # toolchain info
""",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"autoopt {__version__}",
    )

    sub = parser.add_subparsers(dest="command", metavar="command")

    # -- recommend ------------------------------------------
    p_rec = sub.add_parser(
        "recommend",
        help="Predict the best compiler flags for a C/C++ source file",
        description="Analyze source file structure and recommend optimal GCC/Clang flags.",
    )
    p_rec.add_argument("source", help="Path to C or C++ source file")
    p_rec.add_argument("--model", metavar="PATH", help="Custom model .pkl file path")
    p_rec.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    p_rec.add_argument("--compiler", metavar="BIN", help="Compiler binary (default: g++)")

    # -- compile --------------------------------------------
    p_comp = sub.add_parser(
        "compile",
        help="Compile a program with AI-recommended flags (drop-in gcc replacement)",
        description="Like running gcc/g++ but AutoOpt picks the best flags automatically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autoopt compile hello.c                        # -> ./hello
  autoopt compile hello.c -o hello               # -> ./hello
  autoopt compile hello.cpp -o app               # -> ./app (C++ auto-detected)
  autoopt compile loop.c --compare-o3            # compile + benchmark vs -O3
  autoopt compile loop.c --flags '-O2 -march=native'  # manual flags override
""",
    )
    p_comp.add_argument("source", help="Path to C or C++ source file")
    p_comp.add_argument("-o", "--output", metavar="FILE", help="Output binary path")
    p_comp.add_argument("--model", metavar="PATH", help="Custom model .pkl file path")
    p_comp.add_argument("--flags", metavar="FLAGS", help="Override flags instead of using AI prediction")
    p_comp.add_argument("--compiler", metavar="BIN", help="C compiler binary (default: gcc)")
    p_comp.add_argument("--cxx-compiler", metavar="BIN", dest="cxx_compiler",
                        help="C++ compiler binary (default: g++)")
    p_comp.add_argument("--compare-o3", action="store_true",
                        help="After building, benchmark runtime against -O3")
    p_comp.add_argument("--run", action="store_true", help="Execute the binary after building")
    p_comp.add_argument("--runs", type=int, default=3, metavar="N",
                        help="Number of benchmark timing runs (default: 3)")
    p_comp.add_argument("extra", nargs="*", metavar="ARG",
                        help="Extra compiler arguments (e.g. -lm -lpthread)")

    # -- run ------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Compile with AI-optimal flags and immediately run the program",
        description="One-step compile & run. Like `gcc -O3 prog.c -o prog && ./prog`, but smarter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autoopt run hello.c                     # compile & run with AI flags
  autoopt run fib.cpp                     # C++ auto-detected
  autoopt run solver.c -- 100 200         # pass args to your program
  autoopt run bench.c --flags '-O3'       # manual flags override
""",
    )
    p_run.add_argument("source", help="Path to C or C++ source file")
    p_run.add_argument("--model", metavar="PATH", help="Custom model .pkl file path")
    p_run.add_argument("--flags", metavar="FLAGS", help="Override with manual flags")
    p_run.add_argument("--compiler", metavar="BIN", help="C compiler binary (default: gcc)")
    p_run.add_argument("--cxx-compiler", metavar="BIN", dest="cxx_compiler",
                       help="C++ compiler binary (default: g++)")
    p_run.add_argument("--timeout", type=int, default=60, metavar="SEC",
                       help="Program execution timeout in seconds (default: 60)")
    p_run.add_argument("prog_args", nargs="*", metavar="ARG",
                       help="Arguments to pass to the compiled program (use -- to separate)")

    # -- analyze --------------------------------------------
    p_ana = sub.add_parser(
        "analyze",
        help="Show a deep structural analysis (CFG, loops, memory, instruction mix)",
    )
    p_ana.add_argument("source", help="Path to C or C++ source file")
    p_ana.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    p_ana.add_argument("--compiler", metavar="BIN", help="Compiler to use for assembly emission (default: g++)")

    # -- benchmark ------------------------------------------
    p_bm = sub.add_parser(
        "benchmark",
        help="Empirically test all optimization sequences and rank by speed",
    )
    p_bm.add_argument("source", help="Path to C or C++ source file")
    p_bm.add_argument("--runs", type=int, default=3, metavar="N",
                      help="Timing runs per sequence (default: 3)")
    p_bm.add_argument("--compiler", metavar="BIN", help="C compiler binary (default: gcc)")
    p_bm.add_argument("--cxx-compiler", metavar="BIN", dest="cxx_compiler",
                      help="C++ compiler binary (default: g++)")

    # -- synthesize -----------------------------------------
    p_syn = sub.add_parser(
        "synthesize",
        help="Procedurally generate unique C++ benchmark kernel programs",
    )
    p_syn.add_argument("-n", "--count", type=int, default=100, metavar="N",
                       help="Number of programs to generate (default: 100)")
    p_syn.add_argument("-o", "--out", default="benchmarks/synthesized", metavar="DIR",
                       help="Output directory (default: benchmarks/synthesized)")
    p_syn.add_argument("--seed", type=int, default=42, metavar="SEED",
                       help="Random seed for reproducible generation (default: 42)")

    # -- sequences ------------------------------------------
    p_seqs = sub.add_parser(
        "sequences",
        help="List all supported optimization sequences with flags and descriptions",
        aliases=["list-sequences", "ls"],
    )
    p_seqs.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # -- init -----------------------------------------------
    p_init = sub.add_parser(
        "init",
        help="Scaffold a new AutoOpt-ready C/C++ project",
    )
    p_init.add_argument("name", help="Project name / directory to create")
    p_init.add_argument("--lang", choices=["c", "cpp"], default="c",
                        help="Language for the project scaffold (default: c)")
    p_init.add_argument("--force", action="store_true",
                        help="Overwrite existing directory")

    # -- version --------------------------------------------
    sub.add_parser("version", help="Display version and toolchain info")

    return parser


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "recommend":      cmd_recommend,
        "compile":        cmd_compile,
        "run":            cmd_run,
        "analyze":        cmd_analyze,
        "benchmark":      cmd_benchmark,
        "synthesize":     cmd_synthesize,
        "sequences":      cmd_sequences,
        "list-sequences": cmd_sequences,
        "ls":             cmd_sequences,
        "init":           cmd_init,
        "version":        cmd_version,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print(f"\n{dim('Interrupted.')}")
        sys.exit(130)


if __name__ == "__main__":
    main()
