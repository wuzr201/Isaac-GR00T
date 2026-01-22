from typing import Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from diffusers.models.attention import Attention
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree
from typing import Optional


class CrossAttentionBetweenHeads(nn.Module):
    """
    Cross-attention module that allows different action heads to attend to each other.
    This enables coordination between different action parts (e.g., left and right arms).
    
    Each head's features can attend to features from all other heads, allowing
    them to coordinate their actions.
    """
    
    def __init__(
        self,
        num_action_heads: int,  # Number of action heads (e.g., 2 for left/right arms)
        num_attention_heads: int,  # Number of attention heads in the attention mechanism
        num_layers: int,
        hidden_dim: int,
        head_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_action_heads = num_action_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Create multiple layers of cross-attention
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Cross-attention layer: each head attends to all heads
            cross_attn = Attention(
                query_dim=hidden_dim,
                heads=num_attention_heads,
                dim_head=head_dim,
                dropout=dropout,
                bias=True,
                cross_attention_dim=hidden_dim,  # Cross-attend to other heads
                upcast_attention=False,
                out_bias=True,
            )
            # Layer norms
            norm1 = nn.LayerNorm(hidden_dim)
            norm2 = nn.LayerNorm(hidden_dim)
            # Feed-forward
            ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            )
            self.layers.append(nn.ModuleDict({
                'cross_attn': cross_attn,
                'norm1': norm1,
                'norm2': norm2,
                'ff': ff,
            }))
    
    def forward(
        self,
        head_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Apply cross-attention between different action head features.
        
        Args:
            head_features: List of tensors, each of shape [B, T, hidden_dim]
                          representing features for each action head
        
        Returns:
            List of updated features with cross-attention applied
        """
        num_heads = len(head_features)
        assert num_heads == self.num_action_heads, (
            f"Expected {self.num_action_heads} heads, got {num_heads}"
        )
        
        # Process through cross-attention layers
        for layer in self.layers:
            # Concatenate all head features along sequence dimension for cross-attention
            # Shape: [B, num_heads*T, hidden_dim]
            all_heads_concat = torch.cat(head_features, dim=1)
            
            # For each head, apply cross-attention to attend to all heads
            updated_features = []
            for i, head_feat in enumerate(head_features):
                # Query from current head, keys/values from all heads
                normed_query = layer['norm1'](head_feat)
                cross_attn_output = layer['cross_attn'](
                    normed_query,
                    encoder_hidden_states=all_heads_concat,  # Cross-attend to all heads
                    attention_mask=None,
                )
                # Residual connection
                updated_feat = head_feat + cross_attn_output
                
                # Feed-forward
                updated_feat = updated_feat + layer['ff'](layer['norm2'](updated_feat))
                updated_features.append(updated_feat)
            
            head_features = updated_features
        
        return head_features


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        
        # Initialize action decoder(s) - support multi-head or single head
        self.use_multi_head = config.use_multi_head_action_decoder
        if self.use_multi_head and config.action_head_dims is not None:
            # Create multiple decoder heads for different action parts
            self.action_head_dims = list(config.action_head_dims)
            head_sum = sum(self.action_head_dims)
            if head_sum < self.action_dim:
                remainder = self.action_dim - head_sum
                self.action_head_dims.append(remainder)
                print(
                    f"[Multi-head action decoder] action_head_dims sum ({head_sum}) < "
                    f"max_action_dim ({self.action_dim}); appended dummy head of dim {remainder}."
                )
            elif head_sum > self.action_dim:
                raise ValueError(
                    f"Sum of action_head_dims {head_sum} must be <= max_action_dim {self.action_dim}"
                )
            self.action_decoders = nn.ModuleList([
                CategorySpecificMLP(
                    num_categories=config.max_num_embodiments,
                    input_dim=self.hidden_size,
                    hidden_dim=self.hidden_size,
                    output_dim=head_dim,
                )
                for head_dim in self.action_head_dims
            ])
            self.action_decoder = None  # Not used in multi-head mode

            # Store action head names for semantic loss naming
            if config.action_head_names is not None:
                self.action_head_names = list(config.action_head_names)
                # If a dummy head was added, append a default name for it
                if len(self.action_head_names) < len(self.action_head_dims):
                    self.action_head_names.extend([
                        f"head_{i}" for i in range(len(self.action_head_names), len(self.action_head_dims))
                    ])
            else:
                # If no names provided, use default numeric names
                self.action_head_names = [f"head_{i}" for i in range(len(self.action_decoders))]

            self._num_action_heads = len(self.action_decoders)
            self.head_role_embedding = nn.Embedding(self._num_action_heads, self.hidden_size)
            nn.init.normal_(self.head_role_embedding.weight, mean=0.0, std=0.02)
            
            # Initialize cross-attention between heads if enabled
            self.use_cross_attention = config.use_cross_attention_between_heads
            if self.use_cross_attention:
                self.cross_attention = CrossAttentionBetweenHeads(
                    num_action_heads=len(self.action_decoders),
                    num_attention_heads=config.cross_attention_num_heads,
                    num_layers=config.cross_attention_num_layers,
                    hidden_dim=self.hidden_size,
                    head_dim=config.cross_attention_head_dim,
                    dropout=config.attn_dropout,
                )
                print(f"Using cross-attention between {len(self.action_decoders)} action heads for coordination")
            else:
                self.cross_attention = None
            
            print(f"Using multi-head action decoder with {len(self.action_decoders)} heads: {self.action_head_dims}")
        else:
            # Single decoder head (default)
            self.action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )
            self.action_decoders = None
            self.action_head_dims = None
            self.use_cross_attention = False
            self.cross_attention = None

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            if self.use_multi_head and self.action_decoders is not None:
                for decoder in self.action_decoders:
                    decoder.requires_grad_(False)
            else:
                self.action_decoder.requires_grad_(False)
            if self.use_cross_attention and self.cross_attention is not None:
                self.cross_attention.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                if self.use_multi_head and self.action_decoders is not None:
                    for decoder in self.action_decoders:
                        decoder.eval()
                else:
                    self.action_decoder.eval()
                if self.use_cross_attention and self.cross_attention is not None:
                    self.cross_attention.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        # Decode action using single or multi-head decoder
        if self.use_multi_head and self.action_decoders is not None:
            pred = self._decode_action_multi_head(model_output, embodiment_id)
        else:
            pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        # If using multi-head decoder, compute per-head losses and weighted sum
        result = {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        
        if self.use_multi_head and self.action_decoders is not None:
            # Compute per-head losses
            head_losses = []
            head_weights = []
            start_idx = 0
            
            for i, head_dim in enumerate(self.action_head_dims):
                end_idx = start_idx + head_dim
                
                # Slice pred_actions, velocity, and action_mask for this head
                pred_head = pred_actions[:, :, start_idx:end_idx]
                velocity_head = velocity[:, :, start_idx:end_idx]
                mask_head = action_mask[:, :, start_idx:end_idx]
                
                # Compute loss for this head
                head_loss = F.mse_loss(pred_head, velocity_head, reduction="none") * mask_head
                head_loss_sum = head_loss.sum() / (mask_head.sum() + 1e-6)
                head_losses.append(head_loss_sum)
                
                # Weight by the number of valid masked dimensions for this head
                # This gives more weight to heads with more valid action dimensions
                head_weight = mask_head.sum().float() / (action_mask.sum().float() + 1e-6)
                head_weights.append(head_weight)
                
                # Store individual head loss in result (as scalar value) using semantic name
                head_name = self.action_head_names[i]
                result[f"{head_name}_loss"] = head_loss_sum.item()
                
                start_idx = end_idx
            
            # Compute weighted sum of head losses
            head_losses_tensor = torch.stack(head_losses)
            head_weights_tensor = torch.stack(head_weights)
            weighted_head_loss = (head_losses_tensor * head_weights_tensor).sum()
            
            result["loss"] = weighted_head_loss
        
        return result

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    def _decode_action_multi_head(
        self, model_output: torch.Tensor, embodiment_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode action using multiple decoder heads and concatenate outputs.
        Optionally applies cross-attention between heads for coordination.
        
        Args:
            model_output: [B, T, hidden_size] output from DiT model
            embodiment_id: [B] embodiment IDs
            
        Returns:
            [B, T, action_dim] concatenated action predictions
        """
        if not self.use_multi_head or self.action_decoders is None:
            raise ValueError("Multi-head decoder not initialized")
        
        # If cross-attention is enabled, apply it before decoding
        if self.use_cross_attention and self.cross_attention is not None:
            # Create head-specific features first (role-conditioned),
            # then allow heads to attend to each other.
            num_heads = len(self.action_decoders)
            role_ids = torch.arange(num_heads, device=model_output.device, dtype=torch.long)
            role_embs = self.head_role_embedding(role_ids)  # [H, hidden]
            head_features = [
                model_output + role_embs[i].view(1, 1, -1).to(dtype=model_output.dtype)
                for i in range(num_heads)
            ]  # List of [B, T, hidden_size]
            
            # Apply cross-attention between heads
            # This allows each head to attend to features from other heads
            head_features = self.cross_attention(head_features)  # List of [B, T, hidden_size]
            
            # Now decode each head with its cross-attended features
            head_outputs = []
            for decoder, features in zip(self.action_decoders, head_features):
                head_output = decoder(features, embodiment_id)  # [B, T, head_dim]
                head_outputs.append(head_output)
        else:
            # Standard multi-head decoding without cross-attention
            head_outputs = []
            for decoder in self.action_decoders:
                head_output = decoder(model_output, embodiment_id)  # [B, T, head_dim]
                head_outputs.append(head_output)
        
        # Concatenate along the last dimension
        return torch.cat(head_outputs, dim=-1)  # [B, T, action_dim]

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            # Decode action using single or multi-head decoder
            if self.use_multi_head and self.action_decoders is not None:
                pred = self._decode_action_multi_head(model_output, embodiment_id)
            else:
                pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
