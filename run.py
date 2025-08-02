"""
Tritonbench benchmark runner.

Note: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
      has been installed. This script intentionally does not automate or enforce setup steps.
"""

import argparse
import os
import shlex
import sys
from typing import List, Tuple

from tritonbench.operator_loader import get_op_loader_bench_cls_by_name, is_loader_op

from tritonbench.operators import load_opbench_by_name
from tritonbench.operators_collection import list_operators_by_collection
from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.gpu_utils import gpu_lockdown
from tritonbench.utils.list_operator_details import list_operator_details
from tritonbench.utils.parser import get_parser
from tritonbench.utils.run_utils import run_config, run_in_task

from tritonbench.utils.triton_op import BenchmarkOperatorResult
from tritonbench.utils.tritonparse_utils import tritonparse_init, tritonparse_parse

try:
    if is_fbcode():
        from .fb.utils import usage_report_logger  # @manual
    else:
        usage_report_logger = lambda *args, **kwargs: None
except ImportError:
    usage_report_logger = lambda *args, **kwargs: None


def parse_ab_config(config_str: str) -> List[str]:
    """Parse A/B configuration string into list of arguments."""
    if not config_str:
        return []
    
    # Use shlex to properly handle quoted arguments
    try:
        return shlex.split(config_str)
    except ValueError as e:
        raise ValueError(f"Invalid configuration string: {config_str}. Error: {e}")


def separate_global_and_op_args(config_args: List[str]) -> Tuple[List[str], List[str]]:
    """Separate global tritonbench args from operator-specific args."""
    if not config_args:
        return [], []
    
    # Create a temporary parser with only global arguments to identify which args are global
    temp_parser = get_parser()
    
    # Parse the config args to separate global from operator-specific
    try:
        # Use parse_known_args to get global args and remaining (operator) args
        global_args, op_args = temp_parser.parse_known_args(config_args)
        
        # Simple approach: just return the input config_args that were recognized as global
        # and the remaining op_args
        global_arg_list = []
        i = 0
        while i < len(config_args):
            arg = config_args[i]
            if arg.startswith('--'):
                # Check if this arg was consumed by the global parser
                arg_name = arg[2:].replace('-', '_')
                if hasattr(global_args, arg_name) and arg_name not in ['side_a', 'side_b']:
                    global_arg_list.append(arg)
                    # If it's not a flag and has a value, include the value too
                    if i + 1 < len(config_args) and not config_args[i + 1].startswith('-'):
                        i += 1
                        global_arg_list.append(config_args[i])
            i += 1
        
        return global_arg_list, op_args
        
    except SystemExit:
        # If parsing fails, treat all args as operator-specific
        return [], config_args


def update_args_with_global(base_args: argparse.Namespace, global_args: List[str]) -> argparse.Namespace:
    """Update base args with global arguments from A/B config."""
    if not global_args:
        return argparse.Namespace(**vars(base_args))
    
    # Create a copy of base args
    updated_args = argparse.Namespace(**vars(base_args))
    
    # Parse global args and update the namespace
    temp_parser = get_parser()
    try:
        parsed_globals, _ = temp_parser.parse_known_args(global_args)
        
        # Update the namespace with new global values
        for key, value in vars(parsed_globals).items():
            if value is not None and key not in ['side_a', 'side_b']:
                setattr(updated_args, key, value)
                
    except SystemExit:
        # If parsing fails, keep original args
        pass
    
    return updated_args


def run_ab_test(base_args: argparse.Namespace, base_extra_args: List[str]) -> Tuple[BenchmarkOperatorResult, BenchmarkOperatorResult]:
    """Run A/B test with two configurations and return both results."""
    
    # Parse A and B configurations
    config_a_args = parse_ab_config(base_args.side_a)
    config_b_args = parse_ab_config(base_args.side_b)
    
    print(f"[A/B Test] Configuration A: {' '.join(config_a_args)}")
    print(f"[A/B Test] Configuration B: {' '.join(config_b_args)}")
    
    # Separate global and operator-specific arguments
    global_a_args, op_a_args = separate_global_and_op_args(config_a_args)
    global_b_args, op_b_args = separate_global_and_op_args(config_b_args)
    
    if global_a_args:
        print(f"[A/B Test] Global args A: {' '.join(global_a_args)}")
    if op_a_args:
        print(f"[A/B Test] Operator args A: {' '.join(op_a_args)}")
    if global_b_args:
        print(f"[A/B Test] Global args B: {' '.join(global_b_args)}")
    if op_b_args:
        print(f"[A/B Test] Operator args B: {' '.join(op_b_args)}")
    print()
    
    # Update args with global parameters
    args_a = update_args_with_global(base_args, global_a_args)
    args_b = update_args_with_global(base_args, global_b_args)
    
    # Combine extra_args with operator-specific args only
    extra_args_a = base_extra_args + op_a_args
    extra_args_b = base_extra_args + op_b_args
    
    print("=" * 60)
    print(f"Running Configuration A: {' '.join(config_a_args)}")
    if global_a_args:
        print(f"  Global args: {' '.join(global_a_args)}")
    if op_a_args:
        print(f"  Operator args: {' '.join(op_a_args)}")
    print("=" * 60)
    result_a = _run(args_a, extra_args_a)
    
    print("\n" + "=" * 60)
    print(f"Running Configuration B: {' '.join(config_b_args)}")
    if global_b_args:
        print(f"  Global args: {' '.join(global_b_args)}")
    if op_b_args:
        print(f"  Operator args: {' '.join(op_b_args)}")
    print("=" * 60)
    result_b = _run(args_b, extra_args_b)
    
    return result_a, result_b


def _run(args: argparse.Namespace, extra_args: List[str]) -> BenchmarkOperatorResult:
    if is_loader_op(args.op):
        Opbench = get_op_loader_bench_cls_by_name(args.op)
    else:
        Opbench = load_opbench_by_name(args.op)
    opbench = Opbench(
        tb_args=args,
        extra_args=extra_args,
    )
    try:
        opbench.run(args.warmup, args.iter)
    finally:
        metrics = opbench.output
        if not args.skip_print:
            if args.csv:
                metrics.write_csv_to_file(sys.stdout)
            else:
                print(metrics)
        if is_fbcode() and args.log_scuba:
            from .fb.utils import log_benchmark  # @manual

            kwargs = {
                "metrics": metrics,
                "benchmark_name": args.op,
                "device": args.device,
                "logging_group": args.logging_group or args.op,
                "precision": args.precision,
            }
            if args.production_shapes:
                from tritonbench.utils.fb.durin_data import productionDataLoader

                kwargs["weights_loader"] = productionDataLoader

            if "hardware" in args:
                kwargs["hardware"] = args.hardware
            if "triton_type" in args:
                kwargs["triton_type"] = args.triton_type
            log_benchmark(**kwargs)

        if args.plot:
            try:
                opbench.plot()
            except NotImplementedError:
                print(f"Plotting is not implemented for {args.op}")

        if args.output:
            with open(args.output, "w") as f:
                metrics.write_csv_to_file(f)
            print(f"[tritonbench] Output result csv to {args.output}")
        if args.output_json:
            with open(args.output_json, "w") as f:
                metrics.write_json_to_file(f)
        if args.output_dir:
            if args.csv:
                output_file = os.path.join(args.output_dir, f"{args.op}.csv")
                with open(output_file, "w") as f:
                    metrics.write_csv_to_file(f)
            else:
                output_file = os.path.join(args.output_dir, f"{args.op}.json")
                with open(output_file, "w") as f:
                    metrics.write_json_to_file(f)
        return metrics


def run(args: List[str] = []):
    if args == []:
        args = sys.argv[1:]
    if config := os.environ.get("TRITONBENCH_RUN_CONFIG", None):
        run_config(config)
        return

    # Log the tool usage
    usage_report_logger(benchmark_name="tritonbench")
    parser = get_parser()
    args, extra_args = parser.parse_known_args(args)

    tritonparse_init(args.tritonparse)
    if args.op:
        ops = args.op.split(",")
    else:
        ops = list_operators_by_collection(args.op_collection)

    # Handle --list-metrics and --list-backends after determining operators list
    if args.list_metrics or args.list_backends:
        print(
            list_operator_details(
                operators=ops if ops else None,
                show_metrics=args.list_metrics,
                show_backends=args.list_backends,
            )
        )
        return

    # Check if A/B testing mode is enabled
    if args.side_a is not None and args.side_b is not None:
        # A/B testing mode - only support single operator
        assert len(ops) == 1, "A/B testing validation should have caught multiple operators"
        op = ops[0]
        args.op = op
        
        print("[A/B Testing Mode Enabled]")
        print(f"Operator: {op}")
        print()
        
        with gpu_lockdown(args.gpu_lockdown):
            try:
                result_a, result_b = run_ab_test(args, extra_args)
                # TODO: Phase 3 - Implement A/B comparison output
                print("\n[A/B Test Results]")
                print("Configuration A result:", result_a.benchmark_name if result_a else "Failed")
                print("Configuration B result:", result_b.benchmark_name if result_b else "Failed")
            except Exception as e:
                print(f"A/B test failed: {e}")
                if not args.bypass_fail:
                    raise
    else:
        # Normal mode
        # Force isolation in subprocess if testing more than one op.
        if len(ops) >= 2:
            args.isolate = True

        with gpu_lockdown(args.gpu_lockdown):
            for op in ops:
                args.op = op
                if args.isolate:
                    run_in_task(op)
                else:
                    _run(args, extra_args)
                    
    tritonparse_parse(args.tritonparse)


if __name__ == "__main__":
    run()
