"""
Masked Diffusion Model for T5Gemma-TTS

Implements a non-autoregressive masked diffusion approach for audio token generation.
Instead of generating tokens autoregressively, this model:
1. Starts with all tokens masked
2. Iteratively unmasks and predicts tokens following a diffusion schedule
3. Gradually denoises the sequence to generate high-quality audio tokens

This approach is inspired by Masked Diffusion LMs and MaskGIT.
"""

import logging
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.t5gemma.modeling_t5gemma import (
    ALL_ATTENTION_FUNCTIONS,
    EncoderDecoderCache,
    T5GemmaCrossAttention,
    T5GemmaDecoderLayer,
    T5GemmaRotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)

from .utils import make_pad_mask, topk_sampling


class PMCrossAttention(T5GemmaCrossAttention):
    """T5Gemma cross-attention augmented with Progress-Monitoring RoPE."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        # Independent rotary embeddings for decoder queries and encoder keys.
        self.decoder_rotary_emb = T5GemmaRotaryEmbedding(config=config)
        self.encoder_rotary_emb = T5GemmaRotaryEmbedding(config=config)

    @staticmethod
    def _apply_rotary_with_progress(
        projected_states: torch.Tensor,
        base_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        rotary_module: T5GemmaRotaryEmbedding,
    ) -> torch.Tensor:
        if position_ids is None:
            return projected_states
        cos, sin = rotary_module(base_states, position_ids)
        # Broadcast cos/sin to match [B, num_heads, seq, head_dim]
        cos = cos.unsqueeze(1).to(
            dtype=projected_states.dtype, device=projected_states.device
        )
        sin = sin.unsqueeze(1).to(
            dtype=projected_states.dtype, device=projected_states.device
        )
        return (projected_states * cos) + (rotate_half(projected_states) * sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        pm_decoder_position_ids: Optional[torch.Tensor] = None,
        pm_encoder_position_ids: Optional[torch.Tensor] = None,
        **kwargs: FlashAttentionKwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if encoder_hidden_states is None:
            raise ValueError("Encoder hidden state is required for cross attention.")

        # Protect downstream flash attention wrappers from unexpected kwargs.
        pm_decoder_position_ids = kwargs.pop(
            "pm_decoder_position_ids", pm_decoder_position_ids
        )
        pm_encoder_position_ids = kwargs.pop(
            "pm_encoder_position_ids", pm_encoder_position_ids
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if pm_decoder_position_ids is not None:
            query_states = self._apply_rotary_with_progress(
                query_states,
                hidden_states,
                pm_decoder_position_ids,
                self.decoder_rotary_emb,
            )

        if past_key_values is not None:
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            curr_past_key_values = past_key_values.cross_attention_cache

        if past_key_values is None or not is_updated:
            encoder_input_shape = encoder_hidden_states.shape[:-1]
            encoder_hidden_shape = (*encoder_input_shape, -1, self.head_dim)
            key_states = (
                self.k_proj(encoder_hidden_states)
                .view(encoder_hidden_shape)
                .transpose(1, 2)
            )
            if pm_encoder_position_ids is not None:
                key_states = self._apply_rotary_with_progress(
                    key_states,
                    encoder_hidden_states,
                    pm_encoder_position_ids,
                    self.encoder_rotary_emb,
                )
            value_states = (
                self.v_proj(encoder_hidden_states)
                .view(encoder_hidden_shape)
                .transpose(1, 2)
            )

            if past_key_values is not None:
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx
                )
                past_key_values.is_updated[self.layer_idx] = True
        else:
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=None,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PMDecoderLayer(T5GemmaDecoderLayer):
    """Decoder layer variant with PM-RoPE cross-attention built in."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace the existing cross-attention with a PM-enabled version.
        self.cross_attn = PMCrossAttention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        pm_decoder_position_ids: Optional[torch.Tensor] = None,
        pm_encoder_position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        pm_decoder_position_ids = kwargs.pop(
            "pm_decoder_position_ids", pm_decoder_position_ids
        )
        pm_encoder_position_ids = kwargs.pop(
            "pm_encoder_position_ids", pm_encoder_position_ids
        )

        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=(
                past_key_values.self_attention_cache
                if past_key_values is not None
                else None
            ),
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_cross_attn_layernorm(hidden_states)
        hidden_states, _ = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            pm_decoder_position_ids=pm_decoder_position_ids,
            pm_encoder_position_ids=pm_encoder_position_ids,
            **kwargs,
        )
        hidden_states = self.post_cross_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


def _require_transformers():
    try:
        from transformers import AutoModelForSeq2SeqLM
    except ImportError as exc:
        raise ImportError(
            "transformers library not found. Run `pip install transformers`."
        ) from exc
    return AutoModelForSeq2SeqLM


def _require_peft():
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required when use_lora=1. Run `pip install peft`."
        ) from exc
    return LoraConfig, get_peft_model


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    Cosine masking schedule from MaskGIT.
    
    Args:
        t: Timestep values in [0, 1]
    
    Returns:
        Masking ratio in [0, 1]
    """
    return torch.cos(t * math.pi * 0.5)


class MaskedDiffusionModel(nn.Module):
    """
    Masked Diffusion model for discrete audio token generation.
    
    Uses an iterative masking/unmasking process instead of autoregressive generation.
    At each diffusion step:
    1. Model predicts tokens at masked positions
    2. Some masked tokens are unmasked based on confidence
    3. Process repeats until all tokens are unmasked
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        if getattr(self.args, "n_codebooks", 1) != 1:
            logging.info("Resetting n_codebooks to 1 for XCodec2 backend.")
            self.args.n_codebooks = 1
        
        AutoModelForSeq2SeqLM = _require_transformers()

        logging.info(f"Loading T5Gemma backbone: {self.args.t5gemma_model_name}")
        precision = getattr(self.args, "precision", "float32")
        if precision == "float16":
            dtype = torch.float16
        elif precision == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        self.backbone = AutoModelForSeq2SeqLM.from_pretrained(
            self.args.t5gemma_model_name,
            attn_implementation=getattr(self.args, "attn_implementation", "eager"),
            torch_dtype=dtype,
        )
        
        # Prune text modules if requested
        prune_text_modules = getattr(self.args, "prune_text_modules", None)
        legacy_drop_lm_head = getattr(self.args, "drop_lm_head", 0)
        if prune_text_modules is None:
            prune_text_modules = 1 if legacy_drop_lm_head else 0
        drop_lm_head = prune_text_modules >= 1
        drop_decoder_embed = prune_text_modules >= 2

        if drop_lm_head and hasattr(self.backbone, "lm_head"):
            del self.backbone.lm_head
            self.backbone.lm_head = nn.Identity()
            if hasattr(self.backbone.config, "tie_word_embeddings"):
                self.backbone.config.tie_word_embeddings = False
            logging.info("lm_head removed (prune_text_modules=%d)", prune_text_modules)

        if drop_decoder_embed:
            decoder = getattr(self.backbone, "model", getattr(self.backbone, "decoder", None))
            decoder = getattr(decoder, "decoder", decoder)
            if decoder is not None and hasattr(decoder, "embed_tokens"):
                del decoder.embed_tokens
                decoder.embed_tokens = nn.Identity()
                if hasattr(self.backbone.config, "tie_word_embeddings"):
                    self.backbone.config.tie_word_embeddings = False
                logging.info("decoder.embed_tokens removed (prune_text_modules=%d)", prune_text_modules)
        
        self._gc_enabled = bool(getattr(self.args, "t5_gradient_checkpointing", 0))
        if self._gc_enabled:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.backbone.config.use_cache = False
        else:
            self.backbone.config.use_cache = True
        
        if hasattr(self.backbone, "model"):
            self.encoder_module = self.backbone.model.encoder
            self.decoder_module = self.backbone.model.decoder
        else:
            self.encoder_module = getattr(self.backbone, "encoder", None)
            self.decoder_module = getattr(self.backbone, "decoder", None)
        
        if self.encoder_module is None or self.decoder_module is None:
            raise AttributeError(
                "Failed to locate encoder/decoder modules on T5Gemma backbone."
            )
        
        config = self.backbone.config
        hidden_size = getattr(config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "encoder", None)
            if hidden_size is not None:
                hidden_size = getattr(hidden_size, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("T5Gemma config does not expose d_model/hidden_size.")
        self.hidden_size = hidden_size
        self.args.audio_embedding_dim = getattr(
            self.args, "audio_embedding_dim", self.hidden_size
        )

        self._enable_pm_rope_cross_attention()
        self._enable_lora()

        if getattr(self.args, "freeze_t5gemma", 0):
            if getattr(self.args, "use_lora", 0):
                logging.warning(
                    "freeze_t5gemma is ignored when use_lora=1 because LoRA freezes the base model automatically."
                )
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                logging.info("Backbone parameters frozen (freeze_t5gemma=1)")

        self.text_input_type = getattr(self.args, "text_input_type", "text")
        if self.text_input_type == "text":
            self.text_embedding = None
        else:
            text_vocab = self.args.text_vocab_size + 1
            self.text_embedding = nn.Embedding(text_vocab, self.hidden_size)
        self.text_dropout = nn.Dropout(
            getattr(self.args, "text_embedding_dropout", 0.0)
        )

        if isinstance(self.args.audio_vocab_size, list):
            audio_vocab_sizes = [
                size + self.args.n_special for size in self.args.audio_vocab_size
            ]
        else:
            audio_vocab_sizes = [
                self.args.audio_vocab_size + self.args.n_special
            ] * self.args.n_codebooks
        self.n_audio_tokens = audio_vocab_sizes

        self.audio_embedding = nn.ModuleList(
            [
                nn.Embedding(audio_vocab_sizes[k], self.hidden_size)
                for k in range(self.args.n_codebooks)
            ]
        )
        self.audio_dropout = nn.Dropout(
            getattr(self.args, "audio_embedding_dropout", 0.0)
        )

        self.predict_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hidden_size, audio_vocab_sizes[k]),
                )
                for k in range(self.args.n_codebooks)
            ]
        )

        # Diffusion-specific parameters
        self.diffusion_steps = getattr(self.args, "diffusion_steps", 10)
        self.mask_schedule = getattr(self.args, "mask_schedule", "cosine")
        
        # Keep metric lightweight
        self.topk_eval = 10

        class_weight = torch.ones(audio_vocab_sizes[0])
        if getattr(self.args, "eog_weight", 1.0) != 1.0:
            class_weight[self.args.eog] = self.args.eog_weight
        self.register_buffer("class_weight", class_weight, persistent=False)
        self.progress_scale = getattr(self.args, "progress_scale", 2000.0)

        self._freeze_to_lora_only()

        logging.info(
            "MaskedDiffusionModel initialized with %d diffusion steps and %s schedule",
            self.diffusion_steps,
            self.mask_schedule,
        )

    def _enable_pm_rope_cross_attention(self) -> None:
        if getattr(self, "_pm_rope_enabled", False):
            return
        if not getattr(self.args, "use_pm_rope", 1):
            logging.info("PM-RoPE cross-attention disabled by config.")
            return
        decoder_layers = getattr(self.decoder_module, "layers", None)
        if decoder_layers is None:
            logging.warning(
                "Decoder module does not expose layers attribute; skipping PM-RoPE injection."
            )
            return

        new_layers = nn.ModuleList()
        for layer in decoder_layers:
            pm_layer = PMDecoderLayer(layer.config, layer.layer_idx)
            pm_layer.load_state_dict(layer.state_dict(), strict=False)
            pm_layer.gradient_checkpointing = getattr(
                layer, "gradient_checkpointing", False
            )
            if hasattr(layer, "_gradient_checkpointing_func"):
                pm_layer._gradient_checkpointing_func = layer._gradient_checkpointing_func
            new_layers.append(pm_layer)
        self.decoder_module.layers = new_layers
        self._pm_rope_enabled = True
        logging.info(
            "PM-RoPE cross-attention enabled for %d decoder layers.", len(new_layers)
        )

    def _freeze_to_lora_only(self) -> None:
        if not getattr(self.args, "use_lora", 0):
            return
        for _, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def _enable_lora(self) -> None:
        if getattr(self, "_lora_enabled", False):
            return
        if not getattr(self.args, "use_lora", 0):
            return
        LoraConfig, get_peft_model = _require_peft()
        targets = getattr(self.args, "lora_target_modules", None)
        if isinstance(targets, str):
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        if not targets:
            targets = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        lora_config = LoraConfig(
            r=getattr(self.args, "lora_r", 16),
            lora_alpha=getattr(self.args, "lora_alpha", 32),
            lora_dropout=getattr(self.args, "lora_dropout", 0.05),
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=targets,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        for _, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        self._lora_enabled = True
        try:
            self.backbone.print_trainable_parameters()
        except Exception:
            pass
        logging.info(
            "LoRA enabled: targets=%s, r=%d, alpha=%d, dropout=%.3f",
            targets,
            lora_config.r,
            lora_config.lora_alpha,
            lora_config.lora_dropout,
        )

    def _progress_positions_single(self, length: int, device) -> torch.Tensor:
        if length <= 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        if length == 1:
            return torch.zeros(1, device=device, dtype=torch.float32)
        base = torch.arange(length, device=device, dtype=torch.float32)
        return base / (length - 1) * self.progress_scale

    def _build_position_ids(
        self, lengths: torch.Tensor, max_len: int, device
    ) -> torch.Tensor:
        lengths = lengths.to(device=device)
        pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
        denom = (lengths.clamp(min=2).to(torch.float32) - 1.0)[:, None]
        position_ids = pos / denom * self.progress_scale
        mask = pos < lengths[:, None]
        return position_ids.masked_fill(~mask, 0.0)

    def get_mask_ratio(self, step: int) -> float:
        """
        Get the masking ratio for a given diffusion step.
        
        Args:
            step: Current diffusion step (0 = fully masked, diffusion_steps = fully unmasked)
        
        Returns:
            Masking ratio in [0, 1]
        """
        t = step / self.diffusion_steps
        if self.mask_schedule == "cosine":
            return float(cosine_schedule(torch.tensor(t)).item())
        elif self.mask_schedule == "linear":
            return 1.0 - t
        else:
            raise ValueError(f"Unknown mask schedule: {self.mask_schedule}")

    def apply_mask(
        self, y: torch.Tensor, mask_ratio: float, y_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking to audio tokens.
        
        Args:
            y: Audio tokens [B, K, T]
            mask_ratio: Ratio of tokens to mask
            y_lens: Length of each sequence [B]
        
        Returns:
            masked_y: Masked audio tokens
            mask: Binary mask (1 = masked, 0 = unmasked)
        """
        device = y.device
        batch_size, n_codebooks, seq_len = y.shape
        
        # Create mask for each sequence based on its actual length
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            actual_len = int(y_lens[i].item())
            num_mask = int(actual_len * mask_ratio)
            
            if num_mask > 0:
                # Randomly select positions to mask
                indices = torch.randperm(actual_len, device=device)[:num_mask]
                mask[i, indices] = True
        
        # Expand mask to match codebook dimension
        mask_expanded = mask.unsqueeze(1).expand(-1, n_codebooks, -1)
        
        # Apply mask - replace masked tokens with mask_token (using torch.where for efficiency)
        masked_y = torch.where(mask_expanded, self.args.audio_mask_token, y)
        
        return masked_y, mask

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Forward pass for Masked Diffusion training.
        
        During training:
        1. Sample a random diffusion step
        2. Apply corresponding masking ratio
        3. Predict masked tokens
        4. Compute loss on masked positions
        """
        x, x_lens, y, y_lens = batch["x"], batch["x_lens"], batch["y"], batch["y_lens"]
        if len(x) == 0:
            return None

        x = x[:, : x_lens.max()]
        y = y[..., : y_lens.max()]
        
        batch_size = x.shape[0]
        device = x.device

        # Encode text
        x_padding_mask = make_pad_mask(x_lens).to(device)
        encoder_attention_mask = (~x_padding_mask).long()
        if getattr(self.args, "use_pm_rope", 1):
            encoder_position_ids = self._build_position_ids(x_lens, x.shape[1], device)
        else:
            encoder_position_ids = None

        if self.text_input_type == "text":
            encoder_outputs = self.encoder_module(
                input_ids=x,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
            )
        else:
            x_embeds = self.text_dropout(self.text_embedding(x))
            encoder_outputs = self.encoder_module(
                inputs_embeds=x_embeds,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
            )
        memory = encoder_outputs.last_hidden_state

        # Sample random diffusion steps for each batch element
        # Step 0 = fully masked, step diffusion_steps = no masking
        diffusion_steps = torch.randint(
            0, self.diffusion_steps + 1, (batch_size,), device=device
        )
        
        # Get masking ratios and apply masks
        masked_y_list = []
        mask_list = []
        for i in range(batch_size):
            step = int(diffusion_steps[i].item())
            mask_ratio = self.get_mask_ratio(step)
            masked_y_i, mask_i = self.apply_mask(
                y[i:i+1], mask_ratio, y_lens[i:i+1]
            )
            masked_y_list.append(masked_y_i)
            mask_list.append(mask_i)
        
        masked_y = torch.cat(masked_y_list, dim=0)
        mask = torch.cat(mask_list, dim=0)

        # Embed masked tokens (single codebook)
        # masked_y shape: [B, 1, T]
        cated_y = masked_y.transpose(2, 1).contiguous()  # [B, T, 1]
        embedded_y = self.audio_embedding[0](cated_y[:, :, 0])  # [B, T, H]
        embedded_y = self.audio_dropout(embedded_y)

        # Create attention masks
        y_padding_mask = make_pad_mask(y_lens).to(device)
        base_padding_mask = (~y_padding_mask).long()
        
        # For masked diffusion, we use bidirectional attention (no causal mask)
        seq_len = embedded_y.shape[1]
        decoder_attention_mask = base_padding_mask  # [B, T]
        
        pm_kwargs = {}
        if getattr(self.args, "use_pm_rope", 1):
            decoder_position_ids = self._build_position_ids(
                y_lens, embedded_y.shape[1], device
            )
            pm_kwargs["position_ids"] = decoder_position_ids
            pm_kwargs["pm_decoder_position_ids"] = decoder_position_ids
            pm_kwargs["pm_encoder_position_ids"] = encoder_position_ids
        else:
            pm_kwargs["position_ids"] = None

        # Decode
        decoder_outputs = self.decoder_module(
            inputs_embeds=embedded_y,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            **pm_kwargs,
        )
        decoder_hidden = decoder_outputs.last_hidden_state  # [B, T, H]

        # Predict tokens
        logits = self.predict_layer[0](decoder_hidden)  # [B, T, V]

        # Compute loss only on masked positions
        losses = []
        ntokens = []
        top10acc = []
        
        for i in range(batch_size):
            actual_len = int(y_lens[i].item())
            logit_i = logits[i, :actual_len]  # [T, V]
            target_i = y[i, 0, :actual_len]  # [T]
            mask_i = mask[i, :actual_len]  # [T]
            
            # Only compute loss on masked positions
            if mask_i.any():
                masked_logit = logit_i[mask_i]  # [M, V]
                masked_target = target_i[mask_i]  # [M]
                
                loss_i = F.cross_entropy(
                    masked_logit,
                    masked_target,
                    reduction="mean",
                    weight=(
                        self.class_weight.data if self.args.eog_weight != 1 else None
                    ),
                )
                losses.append(loss_i)
                ntokens.append(masked_target.numel())
                
                # Compute accuracy
                with torch.no_grad():
                    k_val = min(self.topk_eval, masked_logit.shape[-1])
                    topk_idx = masked_logit.topk(k_val, dim=-1).indices
                    correct = (topk_idx == masked_target.unsqueeze(-1)).any(dim=-1)
                    top10acc.append(correct.sum())
            else:
                # No masked tokens in this sample (happens at final diffusion step)
                losses.append(torch.tensor(0.0, device=device))
                ntokens.append(0)
                top10acc.append(torch.tensor(0, device=device))

        if sum(ntokens) == 0:
            # Edge case: no tokens to predict
            return {
                "loss": torch.tensor(0.0, device=device),
                "perplexity_by_codebook": [torch.tensor(1.0, device=device)],
                "top10acc": torch.tensor(0, device=device),
                "top10acc_by_codebook": [torch.tensor(0, device=device)],
                "effective_ntoken": torch.tensor(1, device=device),
            }

        # Aggregate loss
        total_tokens = sum(ntokens)
        loss = sum(l * nt for l, nt in zip(losses, ntokens)) / total_tokens if total_tokens > 0 else losses[0]
        perplexity = torch.exp(loss).detach()
        top10acc_total = sum(top10acc)
        effective_ntokens = torch.tensor(total_tokens).to(device)

        return {
            "loss": loss,
            "perplexity_by_codebook": [perplexity],
            "top10acc": top10acc_total,
            "top10acc_by_codebook": [top10acc_total],
            "effective_ntoken": effective_ntokens,
        }

    @torch.inference_mode()
    def inference_tts(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        tgt_y_lens: torch.Tensor,
        top_k: Union[int, List[int]] = -100,
        top_p: float = 1.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
        stop_repetition: int = 3,
        silence_tokens: List[int] = None,
        multi_trial: List[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference using iterative masked diffusion.
        
        Args:
            x: Text tokens [B, T_text]
            x_lens: Text lengths [B]
            y: Prompt audio tokens [B, K, T_prompt]
            tgt_y_lens: Target audio length [B]
        
        Returns:
            Generated audio tokens and the newly generated part
        """
        if getattr(self.args, "n_codebooks", 1) != 1:
            raise ValueError("XCodec2 inference expects n_codebooks=1.")

        device = x.device
        batch_size = x.shape[0]
        assert batch_size == 1, "Current implementation only supports batch size 1."

        # Encode text
        x_padding_mask = make_pad_mask(x_lens).to(device)
        encoder_attention_mask = (~x_padding_mask).long()
        if getattr(self.args, "use_pm_rope", 1):
            encoder_position_ids = self._build_position_ids(x_lens, x.shape[1], device)
        else:
            encoder_position_ids = None

        if self.text_input_type == "text":
            encoder_outputs = self.encoder_module(
                input_ids=x,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
            )
        else:
            x_embeds = self.text_dropout(self.text_embedding(x))
            encoder_outputs = self.encoder_module(
                inputs_embeds=x_embeds,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
            )
        memory = encoder_outputs.last_hidden_state

        # Prepare initial sequence
        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2, 1).contiguous()  # [B, 1, T]
        prompt_len = y.shape[-1]
        
        # Determine target length
        if tgt_y_lens is not None:
            target_len = int(tgt_y_lens[0].item())
        else:
            # Default: double the prompt length
            target_len = prompt_len * 2

        # Initialize with prompt + masked tokens
        mask_token_tensor = torch.full(
            (batch_size, 1, target_len - prompt_len),
            self.args.audio_mask_token,
            dtype=torch.long,
            device=device,
        )
        current_y = torch.cat([y, mask_token_tensor], dim=-1)  # [B, 1, T_total]
        
        # Track which positions are still masked
        is_masked = torch.zeros((batch_size, target_len), dtype=torch.bool, device=device)
        is_masked[:, prompt_len:] = True  # Only new tokens are masked

        # Iterative demasking
        for step in range(self.diffusion_steps, 0, -1):
            # Embed current sequence
            embedded_y = self.audio_embedding[0](current_y[:, 0])  # [B, T, H]
            embedded_y = self.audio_dropout(embedded_y)

            # Prepare attention mask
            y_lens_current = torch.tensor([target_len], device=device, dtype=torch.long)
            decoder_attention_mask = torch.ones(
                (batch_size, target_len), dtype=torch.long, device=device
            )

            pm_kwargs = {}
            if getattr(self.args, "use_pm_rope", 1):
                decoder_position_ids = self._build_position_ids(
                    y_lens_current, target_len, device
                )
                pm_kwargs["position_ids"] = decoder_position_ids
                pm_kwargs["pm_decoder_position_ids"] = decoder_position_ids
                pm_kwargs["pm_encoder_position_ids"] = encoder_position_ids
            else:
                pm_kwargs["position_ids"] = None

            # Decode
            decoder_outputs = self.decoder_module(
                inputs_embeds=embedded_y,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=memory,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                **pm_kwargs,
            )
            decoder_hidden = decoder_outputs.last_hidden_state

            # Predict tokens at masked positions
            logits = self.predict_layer[0](decoder_hidden)  # [B, T, V]

            # Determine how many tokens to unmask at this step
            next_mask_ratio = self.get_mask_ratio(step - 1)
            current_mask_ratio = self.get_mask_ratio(step)
            unmask_ratio = current_mask_ratio - next_mask_ratio
            
            # Count currently masked tokens
            num_masked = is_masked[0].sum().item()
            if num_masked == 0:
                break
            
            # Prevent division by zero
            safe_current_mask_ratio = max(current_mask_ratio, 1e-8)
            num_to_unmask = max(1, int(num_masked * unmask_ratio / safe_current_mask_ratio))

            # Get confidence scores for masked positions
            masked_positions = torch.where(is_masked[0])[0]
            masked_logits = logits[0, masked_positions]  # [M, V]
            
            # Sample tokens for masked positions
            sampled_tokens = topk_sampling(
                masked_logits,
                top_k=top_k if isinstance(top_k, int) else top_k[0],
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
            )  # [M]
            
            # Compute confidence (max probability)
            probs = F.softmax(masked_logits, dim=-1)
            max_probs = probs.max(dim=-1).values  # [M]
            
            # Select most confident tokens to unmask
            if num_to_unmask < len(masked_positions):
                _, confident_indices = torch.topk(max_probs, num_to_unmask)
                positions_to_unmask = masked_positions[confident_indices]
                tokens_to_assign = sampled_tokens[confident_indices]
            else:
                positions_to_unmask = masked_positions
                tokens_to_assign = sampled_tokens
            
            # Update current_y and mask
            current_y[0, 0, positions_to_unmask] = tokens_to_assign
            is_masked[0, positions_to_unmask] = False

        # Extract generated part (excluding prompt)
        generated_tensor = current_y[:, :, prompt_len:]  # [B, 1, T_gen]
        res = current_y  # [B, 1, T_total]

        if self.args.special_first:
            res = res - int(self.args.n_special)
            generated_tensor = generated_tensor - int(self.args.n_special)
        
        return res, generated_tensor

    def carefully_load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load weights while handling potential incompatibilities."""
        try:
            target_dtype = next(self.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32

        def cast_fp(t: torch.Tensor):
            return t.to(dtype=target_dtype) if torch.is_floating_point(t) else t

        prune = getattr(self.args, "prune_text_modules", 0)
        drop_lm_head = prune >= 1 or getattr(self.args, "drop_lm_head", 0)
        drop_dec_embed = prune >= 2

        has_lm_head = any(k.startswith("backbone.lm_head.") for k in state_dict)
        has_dec_embed = any(k.startswith("backbone.model.decoder.embed_tokens.") for k in state_dict)

        removed_keys = []
        if drop_lm_head:
            removed_keys += [k for k in list(state_dict.keys()) if k.startswith("backbone.lm_head.")]
        if drop_dec_embed:
            removed_keys += [k for k in list(state_dict.keys()) if k.startswith("backbone.model.decoder.embed_tokens.")]
        for k in removed_keys:
            state_dict.pop(k)

        if removed_keys:
            strict = False
            logging.info("Dropped %d text-related keys while loading", len(removed_keys))
        elif (not has_lm_head and not drop_lm_head) or (not has_dec_embed and not drop_dec_embed and prune >= 2):
            strict = False
            logging.warning("Checkpoint missing text modules; loading with strict=False")
        else:
            strict = True

        if getattr(self.args, "use_lora", 0):
            targets = getattr(self.args, "lora_target_modules", None)
            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(",") if t.strip()]
            targets = targets or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            remapped = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith("backbone."):
                    new_key = "backbone.base_model.model." + k[len("backbone.") :]
                for tgt in targets:
                    suffix = f".{tgt}.weight"
                    if new_key.endswith(suffix) and "lora_" not in new_key and ".base_layer." not in new_key:
                        new_key = new_key.replace(f".{tgt}.weight", f".{tgt}.base_layer.weight")
                        break
                remapped[new_key] = cast_fp(v)
            current_sd = self.state_dict()
            for k, v in current_sd.items():
                if "lora_" in k and k not in remapped:
                    remapped[k] = v
            state_dict = remapped
            strict = False
        else:
            state_dict = {k: cast_fp(v) for k, v in state_dict.items()}

        result = self.load_state_dict(state_dict, strict=strict)
        missing = getattr(result, "missing_keys", [])
        unexpected = getattr(result, "unexpected_keys", [])
        if missing:
            logging.info("Missing keys: %s", missing)
        if unexpected:
            logging.info("Unexpected keys: %s", unexpected)

        try:
            target_dtype = next(self.parameters()).dtype
            self.to(dtype=target_dtype)
        except StopIteration:
            pass
        return result

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load weights while skipping legacy entries."""
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("accuracy_metrics")}
        return super().load_state_dict(filtered, strict=strict)
