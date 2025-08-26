import argparse

from typing import Any, Callable, List, Optional

import torch

from tritonbench.utils.env_utils import get_nvidia_gpu_model, is_cuda, is_fbcode

from tritonbench.utils.input import input_filter
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
)

from .hstu import get_test_inputs, triton_hstu_mha

HAS_CUDA = False
try:
    HAS_CUDA = is_fbcode() and is_cuda() and get_nvidia_gpu_model() != "NVIDIA B200"
except (FileNotFoundError, AttributeError):
    HAS_CUDA = False

if HAS_CUDA:
    from .fb.hstu import cuda_hstu_mha


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--min-seq-len-log2", type=int, default=8)
    parser.add_argument("--max-seq-len-log2", type=int, default=10)
    parser.add_argument("--seq-sparsity", type=float, default=1.0)
    parser.add_argument("--has-delta-q", type=bool, default=False)
    parser.add_argument("--delta-size", type=int, default=256)
    parser.add_argument("--target-size", type=int, default=20)
    parser.add_argument("--max-attn-len", type=int, default=0)
    # set to 0 to use hstu_mha
    parser.add_argument("--min-full-attn-seq-len", type=int, default=0)
    parser.add_argument("--contextual-seq-len", type=int, default=0)
    parser.add_argument("--sampling-alpha", type=float, default=1.7)
    parser.add_argument("--causal", action="store_true")
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        self.batch_size = args.batch_size
        self.num_heads = args.heads
        self.attn_dim = args.attn_dim
        self.hidden_dim = args.hidden_dim
        self.min_seq_len_log2 = args.min_seq_len_log2
        self.max_seq_len_log2 = args.max_seq_len_log2
        self.sparsity = args.seq_sparsity
        self.has_delta_q = args.has_delta_q
        self.delta_size = args.delta_size
        self.target_size = args.target_size
        self.max_attn_len = args.max_attn_len
        self.min_full_attn_seq_len = args.min_full_attn_seq_len
        self.contextual_seq_len = args.contextual_seq_len
        self.sampling_alpha = args.sampling_alpha
        self.causal = args.causal
        self.alpha = 1.0 / self.attn_dim
        self.requires_grad = not (self.mode == Mode.FWD_NO_GRAD)

    @register_benchmark(baseline=True)
    def hstu(self, q, k, v, seq_offsets, num_targets, max_seq_len):
        return lambda: triton_hstu_mha(
            max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=self.max_attn_len,
            contextual_seq_len=self.contextual_seq_len,
            sort_by_length=True,
            enable_tma=True,
        )

    # TODO: remove B200 hacks like these.
    @register_benchmark(enabled=(HAS_CUDA))
    def hstu_cuda(self, q, k, v, seq_offsets, num_targets, max_seq_len):
        return lambda: cuda_hstu_mha(
            max_seq_len,
            alpha=self.alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=self.causal,
            num_targets=num_targets,
            max_attn_len=self.max_attn_len,
            min_full_attn_seq_len=self.min_full_attn_seq_len,
            contextual_seq_len=self.contextual_seq_len,
            sort_by_length=True,
        )

    def get_x_val(self, example_inputs):
        seq_len = example_inputs[-1]
        return (
            self.batch_size,
            self.num_heads,
            seq_len,
            self.attn_dim,
            self.hidden_dim,
            self.sparsity,
            self.target_size,
            self.max_attn_len,
        )

    def get_input_iter(self):
        for seq_len in [
            2**i for i in range(self.min_seq_len_log2, self.max_seq_len_log2 + 1)
        ]:
            yield get_test_inputs(
                self.batch_size,
                self.num_heads,
                seq_len,
                self.attn_dim,
                self.hidden_dim,
                self.sparsity,
                self.has_delta_q,
                self.delta_size,
                self.target_size,
                self.max_attn_len,
                self.dtype,
                requires_grad=self.requires_grad,
            )

    def get_bwd_fn(self, fwd_fn: Callable[..., Any]) -> Callable[..., Any]:
        o = fwd_fn()
        o_tensor = input_filter(
            lambda x: isinstance(x, torch.Tensor),
            o,
        )
        do = torch.rand_like(o_tensor)
        fn = lambda: o_tensor.backward(do, retain_graph=True)
        return fn

    def _flops(
        self,
        batch_size: int,
        max_seqlen: int,
        attn_dim: int,
        hidden_dim: int,
        nheads: int,
        seq_offsets: torch.Tensor,
        mode: str = "fwd",
    ) -> float:
        assert mode in ["fwd", "bwd", "fwd_bwd"]
        ratio = 2.0  # triangular masking
        f1 = 0.0
        f2 = 0.0
        for i in range(batch_size):
            seq_len = int((seq_offsets[i + 1] - seq_offsets[i]).item())
            # (QK^T), dQ = d(QK^T)K, dK^T = Q^Td(QK^T)
            f1 += 2 * nheads * attn_dim * seq_len**2 // ratio
            # (QK^T)V, d(QK^T) = dOV^T, dV = (QK^T)^TdO,
            f2 += 2 * nheads * hidden_dim * seq_len**2 // ratio
        if mode == "fwd":
            return f1 + f2  # computes (QK^T) and (QK^T)V
        elif mode == "bwd":
            return 3 * f1 + 2 * f2  # computes (QK^T), dQ, dK, dV, d(QK^T)
        else:
            return 4 * f1 + 3 * f2

    @register_metric()
    def flops(
        self, fn_name, example_inputs, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q, k, v, seq_offsets, num_targets, max_seq_len = example_inputs
        flops = self._flops(
            self.batch_size,
            max_seq_len,
            self.attn_dim,
            self.hidden_dim,
            self.num_heads,
            seq_offsets,
            mode=self.mode.value,
        )
        return flops
