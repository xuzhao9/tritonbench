"""A/B testing utilities for tritonbench."""

import argparse
import shlex
from typing import List, Tuple, Optional, Dict, Any

from .triton_op import BenchmarkOperatorResult, BenchmarkOperatorMetrics, REGISTERED_X_VALS
from .parser import get_parser


def parse_ab_config(config_str: str) -> List[str]:
    """Parse A/B configuration string into argument list."""
    if not config_str:
        return []
    try:
        return shlex.split(config_str)
    except ValueError as e:
        raise ValueError(f"Failed to parse configuration string '{config_str}': {e}")


def separate_global_and_op_args(config_args: List[str]) -> Tuple[List[str], List[str]]:
    """Separate global tritonbench args from operator-specific args."""
    if not config_args:
        return [], []
    
    # Use a temporary parser to identify known global arguments
    temp_parser = get_parser()
    
    global_args = []
    op_args = []
    i = 0
    
    while i < len(config_args):
        arg = config_args[i]
        
        if arg.startswith('--'):
            # Check if this is a known global argument
            arg_name = arg.split('=')[0]  # Handle --arg=value format
            
            # Check if the argument is known to the parser
            is_known = False
            for action in temp_parser._actions:
                if arg_name in action.option_strings:
                    is_known = True
                    break
            
            if is_known:
                # This is a global argument
                if '=' in arg:
                    # --arg=value format
                    global_args.append(arg)
                    i += 1
                else:
                    # --arg value format (might need next argument)
                    global_args.append(arg)
                    # Check if next argument is a value (not starting with -)
                    if i + 1 < len(config_args) and not config_args[i + 1].startswith('-'):
                        global_args.append(config_args[i + 1])
                        i += 2
                    else:
                        i += 1
            else:
                # This is an operator-specific argument
                if '=' in arg:
                    # --arg=value format
                    op_args.append(arg)
                    i += 1
                else:
                    # --arg value format (might need next argument)
                    op_args.append(arg)
                    # Check if next argument is a value (not starting with -)
                    if i + 1 < len(config_args) and not config_args[i + 1].startswith('-'):
                        op_args.append(config_args[i + 1])
                        i += 2
                    else:
                        i += 1
        else:
            # Positional argument or value - add to operator args
            op_args.append(arg)
            i += 1
    
    return global_args, op_args


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


def compare_ab_results(result_a: BenchmarkOperatorResult, result_b: BenchmarkOperatorResult, 
                      config_a_args: List[str], config_b_args: List[str]):
    """Compare A/B test results and display formatted comparison."""
    if not result_a or not result_b:
        print("\n[A/B Comparison] ERROR: One or both results are invalid")
        return
    
    print("\n" + "=" * 80)
    print(f"[A/B Test Results Comparison] - {result_a.op_name}")
    print("=" * 80)
    print(f"Configuration A: {' '.join(config_a_args)}")
    print(f"Configuration B: {' '.join(config_b_args)}")
    print()
    
    # Check if both results have data
    if not result_a.result or not result_b.result:
        print("ERROR: No benchmark data available for comparison")
        return
    
    # Get all x_vals (input shapes) that are common to both results
    x_vals_a = {x_val for x_val, _ in result_a.result}
    x_vals_b = {x_val for x_val, _ in result_b.result}
    common_x_vals = sorted(x_vals_a.intersection(x_vals_b))
    
    if not common_x_vals:
        print("ERROR: No common input shapes found between configurations")
        return
    
    # Create result dictionaries for easier lookup
    result_dict_a = {x_val: metrics_dict for x_val, metrics_dict in result_a.result}
    result_dict_b = {x_val: metrics_dict for x_val, metrics_dict in result_b.result}
    
    # Get all backends that are common to both results
    all_backends_a = set()
    all_backends_b = set()
    for x_val in common_x_vals:
        all_backends_a.update(result_dict_a[x_val].keys())
        all_backends_b.update(result_dict_b[x_val].keys())
    common_backends = sorted(all_backends_a.intersection(all_backends_b))
    
    if not common_backends:
        print("ERROR: No common backends found between configurations")
        return
    
    print(f"Comparing {len(common_x_vals)} input shapes across {len(common_backends)} backends")
    print(f"Metrics: {', '.join(result_a.metrics)}")
    print()
    
    # Create comparison table
    x_val_name = REGISTERED_X_VALS.get(result_a.op_name, "x_val")
    
    for backend in common_backends:
        print(f"Backend: {backend}")
        print("-" * 60)
        
        # Create table headers
        headers = [x_val_name]
        for metric in result_a.metrics:
            headers.extend([f"{metric}_A", f"{metric}_B", f"{metric}_diff%"])
        
        # Print headers
        print("{:<15} ".format(headers[0]), end="")
        for i in range(1, len(headers)):
            print("{:<12} ".format(headers[i]), end="")
        print()
        print("-" * (15 + 12 * (len(headers) - 1)))
        
        # Print data rows
        for x_val in common_x_vals:
            if backend not in result_dict_a[x_val] or backend not in result_dict_b[x_val]:
                continue
                
            metrics_a = result_dict_a[x_val][backend]
            metrics_b = result_dict_b[x_val][backend]
            
            # Print x_val
            print("{:<15} ".format(str(x_val)), end="")
            
            # Print metrics comparisons
            for metric in result_a.metrics:
                val_a = getattr(metrics_a, metric, None)
                val_b = getattr(metrics_b, metric, None)
                
                if val_a is not None and val_b is not None:
                    # Handle latency objects
                    if hasattr(val_a, 'p50'):
                        val_a_num = val_a.p50
                    else:
                        val_a_num = val_a
                    
                    if hasattr(val_b, 'p50'):
                        val_b_num = val_b.p50
                    else:
                        val_b_num = val_b
                    
                    # Calculate percentage difference
                    if val_a_num != 0:
                        diff_pct = ((val_b_num - val_a_num) / val_a_num) * 100
                    else:
                        diff_pct = 0
                    
                    # Format values
                    if isinstance(val_a_num, float):
                        val_a_str = f"{val_a_num:.3f}"
                        val_b_str = f"{val_b_num:.3f}"
                    else:
                        val_a_str = str(val_a_num)
                        val_b_str = str(val_b_num)
                    
                    print("{:<12} {:<12} {:<12} ".format(
                        val_a_str, val_b_str, f"{diff_pct:+.1f}%"
                    ), end="")
                else:
                    print("{:<12} {:<12} {:<12} ".format("N/A", "N/A", "N/A"), end="")
            print()
        print()


def run_ab_test(base_args: argparse.Namespace, base_extra_args: List[str], _run_func) -> Tuple[BenchmarkOperatorResult, BenchmarkOperatorResult]:
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
    result_a = _run_func(args_a, extra_args_a)
    
    print("\n" + "=" * 60)
    print(f"Running Configuration B: {' '.join(config_b_args)}")
    if global_b_args:
        print(f"  Global args: {' '.join(global_b_args)}")
    if op_b_args:
        print(f"  Operator args: {' '.join(op_b_args)}")
    print("=" * 60)
    result_b = _run_func(args_b, extra_args_b)
    
    return result_a, result_b