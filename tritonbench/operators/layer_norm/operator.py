import argparse
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
import triton

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
    register_x_val,
)

from . import tutorial

try:
    from .quack_layernorm_bias import layernorm as quack_layernorm_fn
    HAS_QUACK = True
except ImportError:
    quack_layernorm_fn = None
    HAS_QUACK = False


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--M",
        type=int,
        default=4096,
        help="[Optional] Size of dimension 0 in input shape (integer), default: 4096",
    )
    parser.add_argument(
        "--N",
        type=int,
        help="[Optional] Size of dimension 1 in input shape (integer)",
    )
    parser.add_argument(
        "--input-dtype",
        type=str,
        default=None,
        choices=["fp16", "bf16", "fp32"],
        help="[Optional] Data type for input tensor (fp16, bf16, fp32). If not specified, uses the default dtype from tb_args",
    )
    parser.add_argument(
        "--weight-dtype",
        type=str,
        default=None,
        choices=["fp16", "bf16", "fp32"],
        help="[Optional] Data type for weight tensor (fp16, bf16, fp32). If not specified, uses the same as input dtype",
    )
    parser.add_argument(
        "--bias-dtype",
        type=str,
        default=None,
        choices=["fp16", "bf16", "fp32"],
        help="[Optional] Data type for bias tensor (fp16, bf16, fp32). If not specified, uses the same as weight dtype",
    )
    return parser.parse_args(args)

try:
    from liger_kernel.ops.layer_norm import LigerLayerNormFunction

    HAS_LIGER_KERNEL = True
except ModuleNotFoundError:
    LigerLayerNormFunction = None
    HAS_LIGER_KERNEL = False


class Operator(BenchmarkOperator):
    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.M = args.M
        self.N = args.N
        
        # Parse dtype arguments
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        
        # Set input dtype (default to tb_args.dtype if not specified)
        self.input_dtype = dtype_map.get(args.input_dtype) if args.input_dtype else self.dtype
        
        # Set weight dtype (default to input_dtype if not specified)
        self.weight_dtype = dtype_map.get(args.weight_dtype) if args.weight_dtype else self.input_dtype
        
        # Set bias dtype (default to weight_dtype if not specified)
        self.bias_dtype = dtype_map.get(args.bias_dtype) if args.bias_dtype else self.weight_dtype
    @register_benchmark()
    def triton_layer_norm(self, *args):
        (x, w_shape, weight, bias, eps) = args
        # Triton's layer_norm also expects matching dtypes
        weight_matched = weight.to(x.dtype) if weight.dtype != x.dtype else weight
        bias_matched = bias.to(x.dtype) if bias is not None and bias.dtype != x.dtype else bias
        return lambda: tutorial.layer_norm(x, w_shape, weight_matched, bias_matched, eps)

    @register_benchmark(baseline=True)
    def torch_layer_norm(self, *args):
        (x, w_shape, weight, bias, eps) = args
        # PyTorch layer_norm requires all tensors to have the same dtype
        # Convert weight and bias to match x's dtype if necessary
        weight_matched = weight.to(x.dtype) if weight.dtype != x.dtype else weight
        bias_matched = bias.to(x.dtype) if bias is not None and bias.dtype != x.dtype else bias
        return lambda: F.layer_norm(x, w_shape, weight_matched, bias_matched, eps)

    @register_benchmark()
    def torch_compile_layer_norm(self, *args):
        # TODO: remove this once we have a better way to handle backward benchmarking
        # We need to run backward multiple times for proper benchmarking
        # so donated buffer have to be disabled
        if self.mode == Mode.BWD or self.mode == Mode.FWD_BWD:
            from torch._functorch import config as functorch_config

            functorch_config.donated_buffer = False
        import torch
        
        (x, w_shape, weight, bias, eps) = args
        # PyTorch layer_norm requires all tensors to have the same dtype
        weight_matched = weight.to(x.dtype) if weight.dtype != x.dtype else weight
        bias_matched = bias.to(x.dtype) if bias is not None and bias.dtype != x.dtype else bias

        @torch.compile(mode="max-autotune-no-cudagraphs")
        def inner(x, w_shape, weight, bias, eps):
            return F.layer_norm(x, w_shape, weight, bias, eps)

        return lambda: inner(x, w_shape, weight_matched, bias_matched, eps)

    @register_benchmark(enabled=HAS_LIGER_KERNEL)
    def liger_layer_norm(self, *args):
        (x, w_shape, weight, bias, eps) = args
        # Liger kernel might also expect matching dtypes
        weight_matched = weight.to(x.dtype) if weight.dtype != x.dtype else weight
        bias_matched = bias.to(x.dtype) if bias is not None and bias.dtype != x.dtype else bias
        return lambda: LigerLayerNormFunction.apply(x, weight_matched, bias_matched, eps)
    
    @register_benchmark(enabled=HAS_QUACK)
    def quack_layer_norm(self, *args):
        (x, w_shape, weight, bias, eps) = args
        # Quack implementation requires weight and bias to be float32
        # Only convert if they're not already float32
        weight_f32 = weight if weight.dtype == torch.float32 else weight.float()
        bias_f32 = (bias if bias.dtype == torch.float32 else bias.float()) if bias is not None else None
        return lambda: quack_layernorm_fn(x, weight_f32, eps, bias=bias_f32)

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        """Override accuracy to use looser tolerance for mixed precision cases."""
        output = fn()
        baseline_output = baseline_fn()
        
        # Use looser tolerance when mixed precision is detected
        # (when input dtype differs from weight dtype)
        if self.input_dtype != self.weight_dtype:
            rtol = 1e-2
            atol = 1e-2
        else:
            # Use default tolerances for same dtype
            rtol = None
            atol = None
        
        try:
            if rtol is not None and atol is not None:
                torch.testing.assert_close(output, baseline_output, rtol=rtol, atol=atol)
            else:
                torch.testing.assert_close(output, baseline_output)
            return True
        except Exception:
            return False
    
    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        y = fwd_fn()
        dy = 0.1 * torch.randn_like(y)
        return lambda: y.backward(dy, retain_graph=True)

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        x = args[0]
        return [x]

    def get_input_iter(self):
        eps = 1e-5
        
        # If N is provided, use only that value; otherwise use the default range
        if self.N is not None:
            N_values = [self.N]
        else:
            N_values = [512 * i for i in range(2, 32)]
        
        for N in N_values:
            x_shape = (self.M, N)
            w_shape = (x_shape[-1],)
            x = -2.3 + 0.5 * torch.randn(
                x_shape,
                dtype=self.input_dtype,
                device=self.device,
            )
            x.requires_grad_()
            weight = torch.rand(
                w_shape, dtype=self.weight_dtype, device=self.device, requires_grad=True
            )
            bias = torch.rand(
                w_shape, dtype=self.bias_dtype, device=self.device, requires_grad=True
            )
            yield (x, w_shape, weight, bias, eps)

    @register_x_val(label="(M, N)")
    def get_x_val(self, args):
        M, N = args[0].shape
        return (M, N)

    @register_metric()
    def gbps(self, fn, args, metrics: BenchmarkOperatorMetrics) -> float:
        x = args[0]
        base = x.numel() * x.element_size() / metrics.latency * 1e-6
        return {
            Mode.FWD: 2 * base,
            Mode.BWD: 3 * base,
            Mode.FWD_BWD: 5 * base,
        }[self.mode]

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["N"],
                x_vals=self.output.x_vals,
                line_arg="provider",
                line_vals=[
                    "triton_layer_norm",
                    "torch_layer_norm",
                ],
                line_names=[
                    "triton_layer_norm",
                    "torch_layer_norm",
                ],
                styles=[("blue", "-"), ("green", "-")],
                ylabel="GB/s",
                plot_name="layer-norm-fwd",
                args={"M": self.M},
            )
        )
        def _plot(M, N, provider):
            gbps, max_gbps, min_gbps = self.output.get_y_vals(N, provider, "gbps")
            return gbps, max_gbps, min_gbps

        _plot.run(show_plots=True, print_data=True, save_path="/tmp/test_layer_norm")
