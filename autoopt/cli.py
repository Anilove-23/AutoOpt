"""
autoopt.cli
===========
Production Command Line Interface for AutoOpt.

Commands:
  autoopt recommend <file>    Predicts optimal compiler flags for a source file
  autoopt compile <file>      Compiles program with AI-recommended flags
  autoopt analyze <file>      Emits LLVM-inspired static & structural profile
  autoopt synthesize -n <N>   Procedurally generates N unique benchmark programs
  autoopt benchmark <file>    Empirically benchmarks program across all passes
  autoopt version             Displays version and toolchain environment
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoopt import __version__
from autoopt.analyzer import analyze_program
from autoopt.engine.compiler import AutoOptCompiler
from autoopt.engine.runner import AutoOptRunner
from autoopt.engine.sequences import OPTIMIZATION_SEQUENCES, get_sequence_by_id
from autoopt.models.predictor import AutoOptPredictor
from autoopt.synthesizer.generator import KernelSynthesizer


def cmd_recommend(args):
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    predictor = AutoOptPredictor(model_path=Path(args.model) if args.model else None)
    profile = analyze_program(source_path)
    res = predictor.predict_from_profile(profile)

    compiler = AutoOptCompiler()
    binary_name = source_path.stem
    comp_cmd = f"{compiler.get_compiler_for_file(source_path)} {res.recommended_flags} {source_path.name} -o {binary_name}"

    if args.as_json:
        out = {
            "source_file": str(source_path),
            "recommended_flags": res.recommended_flags,
            "sequence_id": res.sequence_id,
            "sequence_name": res.sequence_name,
            "confidence": round(res.confidence, 4),
            "command": comp_cmd,
            "metrics": profile.to_dict(),
        }
        print(json.dumps(out, indent=2))
    else:
        print("\n" + "=" * 60)
        print("  AutoOpt: AI Compiler Optimization Recommendation")
        print("=" * 60)
        print(f"  Source File:        {source_path.name}")
        print(f"  Predicted Sequence: {res.sequence_name} (ID: {res.sequence_id})")
        print(f"  Confidence:         {res.confidence * 100:.1f}%")
        print("-" * 60)
        print(f"  RECOMMENDED COMPILATION COMMAND:")
        print(f"    {comp_cmd}")
        print("=" * 60 + "\n")


def cmd_compile(args):
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    out_binary = Path(args.output) if args.output else source_path.with_suffix(".exe" if sys.platform == "win32" else "")

    print(f"Analyzing {source_path.name} ...")
    predictor = AutoOptPredictor(model_path=Path(args.model) if args.model else None)
    profile = analyze_program(source_path)
    res = predictor.predict_from_profile(profile)

    compiler = AutoOptCompiler()
    print(f"Applying recommended flags: {res.recommended_flags}")
    t0 = time.perf_counter()
    success, stderr, code = compiler.compile(source_path, out_binary, flags=res.recommended_flags)
    comp_time = time.perf_counter() - t0

    if not success:
        print(f"Compilation failed (code {code}):\n{stderr}", file=sys.stderr)
        sys.exit(code)

    sz = compiler.get_binary_size(out_binary)
    print(f"[OK] Successfully built {out_binary.name} ({sz:,} bytes in {comp_time:.2f}s)")

    if args.compare_o3 or args.run:
        runner = AutoOptRunner(runs=args.runs)
        print(f"\nBenchmarking execution ({args.runs} runs) ...")
        t_rec, _, _ = runner.benchmark(out_binary)

        if args.compare_o3:
            tmp_o3 = source_path.with_name(f"{source_path.stem}_o3_baseline.exe")
            compiler.compile(source_path, tmp_o3, flags="-O3")
            t_o3, _, _ = runner.benchmark(tmp_o3)
            try: tmp_o3.unlink()
            except: pass

            if t_rec and t_o3:
                sp = t_o3 / t_rec
                pct = (sp - 1.0) * 100.0
                status = "FASTER" if sp > 1.0 else ("EQUAL" if abs(sp-1.0) < 0.005 else "SLOWER")
                print(f"  AutoOpt Time:   {t_rec:.6f}s")
                print(f"  -O3 Default:    {t_o3:.6f}s")
                print(f"  Speedup Factor: {sp:.4f}x ({pct:+.2f}%) -> {status}")
            else:
                print("Benchmark timing failed.")
        else:
            print(f"  Execution Time: {t_rec:.6f}s")


def cmd_analyze(args):
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    profile = analyze_program(source_path)
    d = profile.to_dict()

    if args.as_json:
        print(json.dumps(d, indent=2))
    else:
        print("\n" + "=" * 65)
        print(f"  AutoOpt Structural & LLVM Analysis Report: {source_path.name}")
        print("=" * 65)
        print(f"  Lines of Code:          {profile.lines_of_code}")
        print(f"  Functions:              {profile.functions_count}")
        print("-" * 65)
        print("  Control Flow Graph (CFG) & Dominance:")
        print(f"    Basic Blocks:         {profile.cfg.basic_blocks}")
        print(f"    Cyclomatic Complexity:{profile.cfg.cyclomatic_complexity}")
        print(f"    Branches:             {profile.cfg.branch_count} (density {profile.cfg.branch_density:.2f})")
        print(f"    Conditional Jumps:    {profile.cfg.conditional_jumps}")
        print(f"    Switch Statements:    {profile.cfg.switch_statements}")
        print(f"    Diamond Branches:     {profile.cfg.diamond_branches}")
        print("-" * 65)
        print("  Loop Structure & Induction Variables (SCEV):")
        print(f"    Total Loops:          {profile.loop.total_loops}")
        print(f"    Max Nesting Depth:    {profile.loop.max_nesting_depth}")
        print(f"    Canonical Loops:      {profile.loop.canonical_loops}")
        print(f"    Strided Loops:        {profile.loop.strided_loops}")
        print(f"    Geometric Loops:      {profile.loop.geometric_loops}")
        print("-" * 65)
        print("  Memory Access & Alias Analysis (aa-eval):")
        print(f"    Memory Operations:    {profile.memory.total_memory_ops} (loads: {profile.memory.load_operations}, stores: {profile.memory.store_operations})")
        print(f"    Memory Intensity:     {profile.memory.memory_intensity:.1%}")
        print(f"    Pointer Dereferences: {profile.memory.pointer_dereferences}")
        print(f"    Alias Risk Score:     {profile.memory.aliasing_risk_score:.2f}")
        print(f"    Contiguous Ratio:     {profile.memory.contiguous_access_ratio:.1%}")
        print("-" * 65)
        print("  Instruction Mix (instcount):")
        print(f"    Total Instructions:   {profile.inst.total_instructions}")
        print(f"    Arithmetic Ops:       {profile.inst.arithmetic_instructions} ({profile.inst.arithmetic_mix_ratio:.1%})")
        print(f"    Floating Point Ops:   {profile.inst.floating_point_instructions} ({profile.inst.float_ratio:.1%})")
        print(f"    Call Instructions:    {profile.inst.call_instructions} ({profile.inst.call_density:.1%})")
        print("=" * 65 + "\n")


def cmd_synthesize(args):
    out_dir = Path(args.out)
    count = args.count
    print(f"Procedurally synthesizing {count:,} unique C++ programs into {out_dir} ...")
    syn = KernelSynthesizer(seed=args.seed)
    t0 = time.perf_counter()
    progs = syn.synthesize_batch(count=count, output_dir=out_dir)
    elapsed = time.perf_counter() - t0
    print(f"[OK] Generated {len(progs):,} distinct C++ programs in {elapsed:.2f}s ({len(progs)/elapsed:.0f} prog/s)")


def cmd_benchmark(args):
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking {source_path.name} across all optimization presets ...\n")
    compiler = AutoOptCompiler()
    runner = AutoOptRunner(runs=args.runs)

    print(f"{'Sequence':<20} {'Flags':<45} {'Size (B)':>10} {'Time (s)':>12} {'vs -O3':>10}")
    print("-" * 102)

    o3_seq = get_sequence_by_id(3)
    tmp_o3 = source_path.with_name(f"tmp_seq3.exe")
    compiler.compile(source_path, tmp_o3, flags=o3_seq.gcc_flags)
    t_o3, _, _ = runner.benchmark(tmp_o3)
    try: tmp_o3.unlink()
    except: pass

    for seq in OPTIMIZATION_SEQUENCES:
        tmp_exe = source_path.with_name(f"tmp_seq_{seq.id}.exe")
        ok, _, _ = compiler.compile(source_path, tmp_exe, flags=seq.gcc_flags)
        if not ok:
            continue
        sz = compiler.get_binary_size(tmp_exe)
        tm, _, _ = runner.benchmark(tmp_exe)
        try: tmp_exe.unlink()
        except: pass

        if tm and t_o3:
            sp = t_o3 / tm
            sp_str = f"{sp:.3f}x"
        else:
            sp_str = "N/A"

        tm_str = f"{tm:.6f}" if tm else "FAIL"
        flags_short = seq.gcc_flags if len(seq.gcc_flags) <= 42 else seq.gcc_flags[:39] + "..."
        print(f"{seq.name:<20} {flags_short:<45} {sz:>10,} {tm_str:>12} {sp_str:>10}")
    print("-" * 102)


def cmd_version(args):
    compiler = AutoOptCompiler()
    print(f"AutoOpt v{__version__} - Machine-Learning-Powered Adaptive Compiler")
    print(f"C Compiler:   {compiler.c_compiler}")
    print(f"C++ Compiler: {compiler.cxx_compiler}")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Platform:     {sys.platform}")


def main():
    parser = argparse.ArgumentParser(
        prog="autoopt",
        description="AutoOpt: ML-Powered Adaptive Compiler Optimization Framework",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # recommend
    p_rec = sub.add_parser("recommend", help="Recommend optimal compiler flags for a source file")
    p_rec.add_argument("source", help="Path to C/C++ source file")
    p_rec.add_argument("--model", help="Path to custom model weights .pkl")
    p_rec.add_argument("--json", action="store_true", dest="as_json", help="Output JSON")

    # compile
    p_comp = sub.add_parser("compile", help="Compile source with AI-recommended flags")
    p_comp.add_argument("source", help="Path to C/C++ source file")
    p_comp.add_argument("-o", "--output", help="Output binary name")
    p_comp.add_argument("--model", help="Path to custom model weights")
    p_comp.add_argument("--compare-o3", action="store_true", help="Benchmark against -O3 after build")
    p_comp.add_argument("--run", action="store_true", help="Execute binary after build")
    p_comp.add_argument("--runs", type=int, default=3, help="Benchmark run count")

    # analyze
    p_ana = sub.add_parser("analyze", help="Analyze program structure (CFG, Loop, Memory, Inst)")
    p_ana.add_argument("source", help="Path to C/C++ source file")
    p_ana.add_argument("--json", action="store_true", dest="as_json", help="Output JSON")

    # synthesize
    p_syn = sub.add_parser("synthesize", help="Procedurally synthesize benchmark kernels")
    p_syn.add_argument("-n", "--count", type=int, default=100, help="Number of programs to generate")
    p_syn.add_argument("-o", "--out", default="benchmarks/synthesized", help="Output directory")
    p_syn.add_argument("--seed", type=int, default=42, help="Random seed")

    # benchmark
    p_bm = sub.add_parser("benchmark", help="Empirically test all passes on a program")
    p_bm.add_argument("source", help="Path to C/C++ source file")
    p_bm.add_argument("--runs", type=int, default=3, help="Runs per pass")

    # version
    sub.add_parser("version", help="Print version info")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "recommend": cmd_recommend,
        "compile": cmd_compile,
        "analyze": cmd_analyze,
        "synthesize": cmd_synthesize,
        "benchmark": cmd_benchmark,
        "version": cmd_version,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
