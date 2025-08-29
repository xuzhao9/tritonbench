# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TRTLLM FMHA utility functions for handling tensor conversion and kernel preparation.
"""

import torch


def trtllm_decode_fmha_func(q, k_cache, v_cache, cache_seqlens):
    """
    TRTLLM FMHA decode function that converts standard tensors to paged format
    and calls the TRTLLM FMHA kernel via PyBind extension.
    """
    
    device = q.device
    # Convert input tensors to paged format for TRTLLM FMHA
    batch_size, seq_len_q, num_qo_heads, head_dim = q.shape
    _, max_seq_len_kv, num_kv_heads, _ = k_cache.shape
    
    # Use page size of 16 for TRTLLM FMHA
    page_size = 16
    max_num_blocks_per_seq = (max_seq_len_kv + page_size - 1) // page_size
    total_pages = batch_size * max_num_blocks_per_seq
    
    # Reshape k_cache and v_cache to paged format [total_pages, num_kv_heads, page_size, head_dim]
    k_cache_paged = k_cache.view(batch_size, max_num_blocks_per_seq, page_size, num_kv_heads, head_dim)
    k_cache_paged = k_cache_paged.permute(0, 1, 3, 2, 4).contiguous()
    k_cache_paged = k_cache_paged.view(total_pages, num_kv_heads, page_size, head_dim)
    
    v_cache_paged = v_cache.view(batch_size, max_num_blocks_per_seq, page_size, num_kv_heads, head_dim)
    v_cache_paged = v_cache_paged.permute(0, 1, 3, 2, 4).contiguous()
    v_cache_paged = v_cache_paged.view(total_pages, num_kv_heads, page_size, head_dim)
    
    # Create block tables
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), 
        dtype=torch.int32, 
        device=device
    )
    for i in range(batch_size):
        for j in range(max_num_blocks_per_seq):
            block_tables[i, j] = i * max_num_blocks_per_seq + j
    
    # Create output tensor
    out = torch.zeros_like(q)
    
    # Create workspace buffer
    workspace_size = 128 * 1024 * 1024  # 128MB
    workspace_buffer = torch.zeros(workspace_size, dtype=torch.uint8, device=device)
    
    # Attention parameters
    max_seq_len = cache_seqlens.max().item()
    bmm1_scale = 1.0 / (head_dim ** 0.5)
    bmm2_scale = 1.0
    window_left = -1  # No sliding window
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    
    args =(
        out, q, k_cache_paged, v_cache_paged, workspace_buffer,
        block_tables, cache_seqlens, max_seq_len,
        bmm1_scale, bmm2_scale, window_left, sm_count
    )
    return args
