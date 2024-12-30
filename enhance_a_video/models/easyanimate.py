from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention import Attention
from einops import rearrange
from torch import nn

from enhance_a_video.enhance import enhance_score
from enhance_a_video.globals import get_num_frames, is_enhance_enabled, set_num_frames


def inject_enhance_for_easyanimate(model: nn.Module) -> None:
    """
    Inject enhance score for EasyAnimate model.
    1. register hook to update num frames
    2. replace attention processor with enhance processor to weight the attention scores
    """
    # register hook to update num frames
    model.register_forward_pre_hook(num_frames_hook, with_kwargs=True)
    # replace attention with enhanceAvideo
    for name, module in model.named_modules():
        if "attn" in name and isinstance(module, Attention):
            module.set_processor(EnhanceEasyAnimateAttnProcessor2_0())


def num_frames_hook(_, args, kwargs):
    """
    Hook to update the number of frames automatically.
    """
    if "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
    else:
        hidden_states = args[0]
    num_frames = hidden_states.shape[1]
    set_num_frames(num_frames)
    return args, kwargs


class EnhanceEasyAnimateAttnProcessor2_0:
    def __init__(self):
        pass

    def _get_enhance_scores(self, attn, query, key):

        num_frames = get_num_frames()
        _, num_heads, ST, head_dim = query.shape
        spatial_dim = ST / num_frames
        spatial_dim = int(spatial_dim)

        query_image = rearrange(
            query, "B N (T S) C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
        )
        key_image = rearrange(key, "B N (T S) C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim)

        return enhance_score(query_image, key_image, head_dim, num_frames)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attn2: Attention = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn2 is None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ========== Enhance-A-Video ==========
        if is_enhance_enabled():
            enhance_scores = self._get_enhance_scores(attn, query, key)
        # ========== Enhance-A-Video ==========
        
        if attn2 is not None:
            query_txt = attn2.to_q(encoder_hidden_states)
            key_txt = attn2.to_k(encoder_hidden_states)
            value_txt = attn2.to_v(encoder_hidden_states)
            
            inner_dim = key_txt.shape[-1]
            head_dim = inner_dim // attn.heads

            query_txt = query_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_txt = key_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_txt = value_txt.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn2.norm_q is not None:
                query_txt = attn2.norm_q(query_txt)
            if attn2.norm_k is not None:
                key_txt = attn2.norm_k(key_txt)

            query = torch.cat([query_txt, query], dim=2)
            key = torch.cat([key_txt, key], dim=2)
            value = torch.cat([value_txt, value], dim=2)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        if attn2 is None:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
        else:
            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            encoder_hidden_states = attn2.to_out[0](encoder_hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn2.to_out[1](encoder_hidden_states)

        # ========== Enhance-A-Video ==========
        if is_enhance_enabled():
            hidden_states = hidden_states * enhance_scores
        # ========== Enhance-A-Video ==========

        return hidden_states, encoder_hidden_states