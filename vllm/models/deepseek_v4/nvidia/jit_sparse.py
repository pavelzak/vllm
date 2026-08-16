# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 SM120 attention on flashinfer's JIT warp-spec sparse-MLA kernels.

Port of the dspark-recipe image's proven serving path — its overlay ``sm120.py``
driving ``BatchSparseMLAPagedAttentionWrapper`` (an NVIDIA backport into
flashinfer 0.6.12) — onto stock flashinfer 0.6.17, whose
``flashinfer.mla._sparse_mla_sm120`` functional API exposes the same kernels
(the DSV4 decode ``.cu`` is byte-identical between the two). Both decode and
prefill route through ``_sparse_mla_sm120_paged_attention``, which
auto-dispatches decode (``num_tokens <= 64``) vs prefill internally — replacing
the parent class's trtllm-gen cubin calls, the family implicated in the
cross-node CUDA-graph replay deadlock and the spec-decode page-size-64
constraint. The docker image ran CUDA graphs + DSpark k=5 at block-size 256 on
exactly these kernels.

Contract (flashinfer 0.6.17, private API — pin the version):
  q [num_tokens, num_heads, 512] bf16 (d_qk=512 selects the DSV4
  footer-scale cache traits); caches passed as raw 3-D byte pages
  [num_blocks, page_block_size, bytes]; indices [num_tokens, (1,) topk] int32
  with -1 invalid; out_lse [num_tokens, num_heads] fp32, mutated in place;
  mid_out/mid_lse split-K scratch required on decode-sized calls.

Enable via ``VLLM_DSV4_SM120_JIT_ATTN=1`` (selection in ``nvidia/model.py``).
"""

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.b12x_sparse import (
    _DECODE_MAX_TOKENS,
    _get_decode_scratch,
)
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)


def use_sm120_jit_attn() -> bool:
    value = os.getenv("VLLM_DSV4_SM120_JIT_ATTN", "0").strip().lower()
    return value not in ("0", "false", "no", "off", "")


def _paged_attention():
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

    return _sparse_mla_sm120_paged_attention



def _check_decode_dispatch(
    num_tokens: int,
    num_heads: int,
    topk: int,
    kv_cache: torch.Tensor,
    extra_topk: int,
    where: str,
) -> None:
    """Fail with named shapes when a <=64-token call cannot dispatch.

    Non-dispatchable small calls fall into the JIT prefill orchestrator,
    which dies with an unhelpful `num_tokens > 64` check failure.
    """
    from flashinfer.mla._sparse_mla_sm120 import _decode_dsv4_dispatchable

    page_block_size = int(kv_cache.shape[1])
    if not _decode_dsv4_dispatchable(
        num_tokens, num_heads, topk, 512, page_block_size, extra_topk
    ):
        raise RuntimeError(
            f"SM120 JIT sparse-MLA {where} call is not decode-dispatchable: "
            f"num_tokens={num_tokens} num_heads={num_heads} topk={topk} "
            f"page_block_size={page_block_size} extra_topk={extra_topk}. "
            "Requirements: page_block_size == 64 and (num_heads, topk) in "
            "flashinfer.mla._sparse_mla_sm120._DECODE_DSV4_DISPATCH "
            "(topk in {128, 512, 1024}; indices must be padded to a "
            "supported width, cf. noncausal_index_width in sparse_swa.py)."
        )


class DeepseekV4JITSparseSM120Attention(DeepseekV4FlashInferSM120Attention):
    """SM120 attention with decode+prefill on the JIT warp-spec kernel family."""

    def _get_out_lse(self, num_tokens: int, num_heads: int) -> torch.Tensor:
        buf = getattr(self, "_jit_out_lse", None)
        if buf is None or buf.shape[0] < num_tokens or buf.shape[1] < num_heads:
            buf = torch.empty(
                (num_tokens, num_heads),
                dtype=torch.float32,
                device=self.attn_sink.device,
            )
            self._jit_out_lse = buf
        return buf[:num_tokens, :num_heads]

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Index prep: byte-for-byte the parent class's logic.
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
        if num_decode_tokens <= _DECODE_MAX_TOKENS:
            _check_decode_dispatch(
                num_decode_tokens,
                q.shape[1],
                swa_indices.shape[-1],
                self.swa_cache_layer.kv_cache,
                extra_topk,
                "decode",
            )
        _paged_attention()(
            q=q,
            kv_cache=self.swa_cache_layer.kv_cache,
            indices=swa_indices,
            output=output,
            out_lse=self._get_out_lse(num_decode_tokens, q.shape[1]),
            sm_scale=self.scale,
            topk_length=swa_lens,
            attn_sink=self.attn_sink,
            extra_kv_cache=kv_cache if not swa_only else None,
            extra_indices=extra_sparse_indices,
            extra_topk_length=extra_sparse_lengths,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio <= 1

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        local_topk_indices: torch.Tensor | None
        if swa_only:
            local_topk_indices = None
        elif self.compress_ratio == 4:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "C4A prefill requires top-k indices from the indexer."
                )
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            if attn_metadata is None:
                raise RuntimeError("C128A prefill metadata is missing.")
            local_topk_indices = attn_metadata.c128a_prefill_topk_indices

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        if local_topk_indices is not None:
            if attn_metadata is None:
                raise RuntimeError("C4A prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError("C4A prefill request mapping is missing.")
            if swa_metadata.is_valid_token is None:
                raise RuntimeError("C4A prefill validity metadata is missing.")
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            block_size = attn_metadata.block_size // self.compress_ratio
            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        q = self._prepare_query(q, output)
        if not swa_only and compressed_k_cache is None:
            raise RuntimeError(
                "Compressed sparse MLA layers require their compressed KV cache."
            )

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            chunk_tokens = int(query_end - query_start)
            if chunk_tokens == 0:
                continue

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )

            mid_out = None
            mid_lse = None
            if chunk_tokens <= _DECODE_MAX_TOKENS:
                # The functional API dispatches <=64-token calls to the decode
                # kernels, which require caller-provided split-K scratch. The
                # decode kernels treat rows independently, so request
                # boundaries inside the chunk need no special handling.
                extra_topk = (
                    extra_sparse_indices_chunk.shape[-1]
                    if extra_sparse_indices_chunk is not None
                    else 0
                )
                mid_out, mid_lse = _get_decode_scratch(
                    chunk_tokens,
                    q.shape[1],
                    output.shape[-1],
                    swa_metadata.prefill_swa_indices.shape[-1],
                    extra_topk,
                )
                _check_decode_dispatch(
                    chunk_tokens,
                    q.shape[1],
                    swa_metadata.prefill_swa_indices.shape[-1],
                    swa_k_cache,
                    extra_topk,
                    "prefill-chunk",
                )

            _paged_attention()(
                q=q[query_start:query_end],
                kv_cache=swa_k_cache,
                indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                output=output[query_start:query_end],
                out_lse=self._get_out_lse(chunk_tokens, q.shape[1]),
                sm_scale=self.scale,
                topk_length=swa_metadata.prefill_swa_lens[query_start:query_end],
                attn_sink=self.attn_sink,
                extra_kv_cache=compressed_k_cache if not swa_only else None,
                extra_indices=extra_sparse_indices_chunk,
                extra_topk_length=extra_sparse_lengths_chunk,
                mid_out=mid_out,
                mid_lse=mid_lse,
            )
