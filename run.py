"""
Tritonbench benchmark runner.

Note: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
      has been installed. This script intentionally does not automate or enforce setup steps.
"""

import argparse
import os
import shlex
import sys
from typing import List, Tuple, Optional

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
from tritonbench.utils.ab_test import run_ab_test, compare_ab_results

try:
    if is_fbcode():
        from .fb.utils import usage_report_logger  # @manual
    else:
        usage_report_logger = lambda *args, **kwargs: None
except ImportError:
    usage_report_logger = lambda *args, **kwargs: None




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


from triton.knobs import JITHook, LaunchHook
class JITHookImpl(JITHook):
    """
    JIT Hook implementation that overrides or sets the launch_metadata function for Triton kernels.

    This hook is essential for capturing detailed kernel launch information beyond the basic
    metadata (like kernel name) that Triton provides by default. Without setting a custom
    launch_metadata function, only minimal launch information is available as shown in:
    https://github.com/triton-lang/triton/blob/7ce287dc24b43476cdeb30529089ac361564505d/python/triton/compiler/compiler.py#L504

    By intercepting the JIT compilation process and setting a custom launch_metadata function,
    we can capture comprehensive runtime information including grid parameters, kernel metadata,
    and argument dictionaries for detailed analysis and logging.
    """

    def __call__(
        self,
        *,
        key: str,
        repr: str,
        fn,
        compile,
        is_manual_warmup: bool,
        already_compiled: bool,
    ) -> Optional[bool]:
        """
        Override or set the launch_metadata function for the JIT-compiled kernel.

        This method is called during the JIT compilation process and allows us to
        inject our custom launch_metadata function that will be used to collect
        detailed kernel launch information.

        Args:
            key: Unique identifier for the kernel
            repr: String representation of the kernel
            fn: The JIT function object
            compile: Compilation function
            is_manual_warmup: Whether this is a manual warmup call
            already_compiled: Whether the kernel is already compiled

        Returns:
            True to continue with compilation, None/False to skip
        """
        # Check kernel allowlist early to avoid unnecessary work

        # Get the current launch_metadata function if it exists
        current_launch_metadata = getattr(fn.jit_function, "launch_metadata", None)
        if current_launch_metadata is not None:
            print(
                f"fn {fn} launch_metadata is not None: {current_launch_metadata}. It will be overridden by tritonparse."
            )
        fn.jit_function.launch_metadata = None
        return True



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
    if args.no_listener:
        from triton import knobs
        jit_hook = JITHookImpl()
        knobs.runtime.jit_post_compile_hook = jit_hook

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
                result_a, result_b = run_ab_test(args, extra_args, _run)
                
                from tritonbench.utils.ab_test import parse_ab_config
                config_a_args = parse_ab_config(args.side_a)
                config_b_args = parse_ab_config(args.side_b)
                compare_ab_results(result_a, result_b, config_a_args, config_b_args)
                
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
