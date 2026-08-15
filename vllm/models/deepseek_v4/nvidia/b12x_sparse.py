# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 SM120 attention with the b12x compressed-MLA decode kernel.

Port of the dspark-recipe overlay's ``sm120.py`` b12x decode branch onto
vLLM 0.26's :class:`DeepseekV4FlashInferSM120Attention`. Only ``_forward_decode``
is overridden: decode goes through ``b12x`` (page-size agnostic, CUDA-graph
capture-safe by contract); prefill/verify stay on the inherited FlashInfer
path — the exact split the docker image ran in production for 4 days.

Enable via ``VLLM_DSV4_B12X_COMPRESSED_MLA=1`` (selection happens in
``nvidia/model.py:_select_dsv4_attn_cls``).
"""

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)

_DECODE_MAX_TOKENS = 64
_DECODE_SPLIT_TILE = 64
_C128A_TOPK_ALIGNMENT = 128


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _decode_num_splits(topk: int, extra_topk: int = 0) -> int:
    return _cdiv(topk, _DECODE_SPLIT_TILE) + _cdiv(extra_topk, _DECODE_SPLIT_TILE)


def _max_decode_workspace_tokens(max_num_batched_tokens: int) -> int:
    return min(int(max_num_batched_tokens), _DECODE_MAX_TOKENS)


def _c128a_max_compressed(max_model_len: int, compress_ratio: int) -> int:
    return (
        _cdiv(_cdiv(max_model_len, compress_ratio), _C128A_TOPK_ALIGNMENT)
        * _C128A_TOPK_ALIGNMENT
    )


def use_b12x_compressed_mla() -> bool:
    value = os.getenv("VLLM_DSV4_B12X_COMPRESSED_MLA", "0").strip().lower()
    return value not in ("0", "false", "no", "off", "")


def _import_b12x_compressed_decode():
    """Resolve b12x compressed-MLA entry points across package layouts.

    b12x <=0.15.x: b12x.attention.mla.compressed_api / b12x.attention.workspace
    b12x >=1.x:    b12x.attention.compressed_mla.run /
                   b12x.attention._shared.workspace (plan/bind grammar)
    """
    try:  # <= 0.15.x (the production-validated layout)
        from b12x.attention.mla.compressed_api import compressed_mla_decode_forward
        from b12x.attention.workspace import B12XAttentionWorkspace

        return compressed_mla_decode_forward, B12XAttentionWorkspace, "0.15"
    except ImportError:
        from b12x.attention._shared.mla.compressed_api import (
            compressed_mla_decode_forward,
        )
        from b12x.attention._shared.workspace import B12XAttentionWorkspace

        return compressed_mla_decode_forward, B12XAttentionWorkspace, "1.x"


def _get_decode_scratch(
    num_tokens: int,
    num_heads: int,
    d_v: int,
    topk: int,
    extra_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_splits = _decode_num_splits(topk, extra_topk)
    mid_out, mid_lse = current_workspace_manager().get_simultaneous(
        ((num_tokens, num_heads, num_splits, d_v), torch.bfloat16),
        ((num_tokens, num_heads, num_splits), torch.float32),
    )
    return mid_out, mid_lse


def _b12x_index_matrix(indices: torch.Tensor | None) -> torch.Tensor | None:
    if indices is None:
        return None
    if indices.ndim == 3:
        assert indices.shape[1] == 1
        return indices.squeeze(1)
    return indices


class DeepseekV4B12XSM120Attention(DeepseekV4FlashInferSM120Attention):
    """FlashInfer SM120 attention with b12x taking over the decode path."""

    def _extra_topk_capacity(self) -> int:
        if self.compress_ratio <= 1:
            return 0
        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            return int(self.topk_indices_buffer.shape[-1])
        if self.compress_ratio == 128:
            return _c128a_max_compressed(self.max_model_len, self.compress_ratio)
        raise ValueError(
            f"Unsupported compress_ratio={self.compress_ratio}; expected 1, 4, 128."
        )

    def _get_b12x_decode_workspace(self, *, extra_topk: int):
        _, B12XAttentionWorkspace, _ = _import_b12x_compressed_decode()

        total_topk = int(self.window_size) + int(extra_topk)
        max_rows = _max_decode_workspace_tokens(self.max_num_batched_tokens)
        max_chunks = _decode_num_splits(self.window_size, extra_topk)

        workspace = getattr(self, "_b12x_compressed_mla_workspace", None)
        if (
            workspace is None
            or int(workspace.topk) < total_topk
            or int(workspace.max_total_q) < max_rows
            or int(workspace.max_chunks_per_row) < max_chunks
            or int(workspace.num_q_heads) != int(self.padded_heads)
        ):
            device = self.attn_sink.device
            if device.type != "cuda":
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            workspace = B12XAttentionWorkspace(
                mode="decode",
                device=device,
                dtype=torch.bfloat16,
                kv_dtype=torch.uint8,
                num_q_heads=int(self.padded_heads),
                head_dim=512,
                v_head_dim=512,
                topk=total_topk,
                max_total_q=max_rows,
                max_batch=max_rows,
                max_page_table_width=total_topk,
                max_paged_q_rows=max_rows,
                page_size=int(self.swa_cache_layer.block_size),
                padded_heads=int(self.padded_heads),
                max_chunks_per_row=max_chunks,
            )
            workspace.kv_chunk_size_ptr = torch.empty(
                (1,), dtype=torch.int32, device=device
            )
            workspace.num_chunks_ptr = torch.empty(
                (1,), dtype=torch.int32, device=device
            )
            self._b12x_compressed_mla_workspace = workspace
            logger.info_once(
                "DeepSeek V4 SM120 b12x compressed MLA decode enabled "
                "(topk=%d, max_rows=%d, max_chunks=%d).",
                total_topk,
                max_rows,
                max_chunks,
            )
        return workspace

    def _reserve_b12x_decode_workspace(self) -> None:
        """Pre-touch all decode scratch during profiling/dummy runs so CUDA
        graph capture never allocates (overlay sm120.py:276-286 semantics)."""
        extra_topk = self._extra_topk_capacity()
        _get_decode_scratch(
            _max_decode_workspace_tokens(self.max_num_batched_tokens),
            self.padded_heads,
            512,
            self.window_size,
            extra_topk,
        )
        self._get_b12x_decode_workspace(extra_topk=extra_topk)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        compressed_mla_decode_forward, _, api_gen = _import_b12x_compressed_decode()

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # --- index prep: byte-for-byte the parent class's logic -------------
        extra_sparse_indices = None
        extra_sparse_lengths = None
        block_size = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            block_size = attn_metadata.block_size // self.compress_ratio
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_query(q, output)

        # --- b12x decode ------------------------------------------------------
        extra_topk = (
            extra_sparse_indices.shape[-1] if extra_sparse_indices is not None else 0
        )
        mid_out, mid_lse = _get_decode_scratch(
            num_decode_tokens,
            q.shape[1],
            output.shape[-1],
            swa_indices.shape[-1],
            extra_topk,
        )
        workspace = self._get_b12x_decode_workspace(extra_topk=extra_topk)
        workspace.tmp_output = mid_out
        workspace.tmp_lse = mid_lse
        workspace.output_buffer = output

        # Pass caches RAW, exactly as the overlay did: b12x performs its own
        # byte-view internally (_compressed_mla_cache_byte_view). The parent
        # class's _as_sparse_cache() adds an unsqueeze(-2) for flashinfer's
        # 4-D layout, which silently mis-strides b12x's gather (garbage
        # output, no exception).
        swa_cache = self.swa_cache_layer.kv_cache
        extra_cache = kv_cache
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        if api_gen != "0.15":
            raise NotImplementedError(
                "b12x >=1.x binding-form call not wired yet (Stage 4); "
                "install b12x 0.15.x."
            )
        result = compressed_mla_decode_forward(
            q_all=q,
            swa_k_cache=swa_cache,
            swa_indices=_b12x_index_matrix(swa_indices),
            swa_topk_lengths=swa_lens,
            workspace=workspace,
            sm_scale=self.scale,
            swa_page_size=swa_metadata.block_size,
            indexed_k_cache=extra_cache,
            indexed_indices=_b12x_index_matrix(extra_sparse_indices),
            indexed_topk_lengths=extra_sparse_lengths,
            indexed_page_size=block_size if extra_cache is not None else None,
            attn_sink=self.attn_sink,
            expected_num_q_heads=q.shape[1],
            backend="sm120_unified",
        )
        if result.data_ptr() != output.data_ptr():
            output.copy_(result)
