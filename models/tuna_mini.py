"""TUNA-mini: reduced reproduction of the TUNA-2 architectural family.

Three input variants share the same transformer backbone and the same
classification head; they differ only in the front-end that maps a 224×224
grayscale chest radiograph to a sequence of D-dim tokens:

    pixel    — direct patch embedding (TUNA-2 style; no encoder)
    encoder  — frozen DINOv2 ViT-S/14 features projected to D (TUNA-R style)
    vae      — frozen SD VAE latents patchified (Tuna / SD style)

A second head (``diff_head``) operates on the same backbone output and
predicts a per-token rectified-flow velocity for masked positions, used
as an auxiliary self-supervised denoising signal.  In ``cls`` mode this
head is unused; in ``cls+diff`` mode both losses are summed.

All input transforms (RGB-replication, ImageNet norm, [-1,1] scaling)
live inside :meth:`TunaMini.encode` so the dataset can stay variant-agnostic
and emit a single grayscale tensor in [-1, 1].
"""
from __future__ import annotations
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


Variant = Literal["pixel", "encoder", "vae"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------- shared ViT pieces ----------


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding for t ∈ [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        ang = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def _patchify(x: torch.Tensor, ps: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, N, C*ps*ps] flattened patch tokens."""
    B, C, H, W = x.shape
    x = x.unfold(2, ps, ps).unfold(3, ps, ps)  # [B, C, H/ps, W/ps, ps, ps]
    return x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * ps * ps)


# ---------- main model ----------


class TunaMini(nn.Module):
    def __init__(
        self, variant: Variant,
        embed_dim: int = 384, depth: int = 12, n_heads: int = 6,
        mlp_ratio: float = 4.0, n_classes: int = 25,
        image_size: int = 224, patch_size: int = 14,
        vae_path: str = "stabilityai/sd-vae-ft-mse",
        dinov2_name: str = "dinov2_vits14",
    ):
        super().__init__()
        if variant not in ("pixel", "encoder", "vae"):
            raise ValueError(variant)
        self.variant = variant
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # ---- Front-end (sets self.token_dim and self.n_patches) ----
        if variant == "pixel":
            assert image_size % patch_size == 0
            self.token_dim = 1 * patch_size * patch_size
            self.n_patches = (image_size // patch_size) ** 2
            self._frozen = nn.ModuleList()  # nothing frozen
        elif variant == "encoder":
            self.frozen_encoder = torch.hub.load(
                "facebookresearch/dinov2", dinov2_name
            )
            for p in self.frozen_encoder.parameters():
                p.requires_grad = False
            self.frozen_encoder.eval()
            self.token_dim = self.frozen_encoder.embed_dim  # e.g., 384
            self.n_patches = (image_size // 14) ** 2
            self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("_std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        elif variant == "vae":
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(vae_path)
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.eval()
            self.vae_patch = 2
            self.vae_factor = 8
            self.token_dim = 4 * self.vae_patch * self.vae_patch
            self.n_patches = (image_size // self.vae_factor // self.vae_patch) ** 2

        # ---- shared transformer ----
        self.token_proj = nn.Linear(self.token_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.t_proj = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ---- heads ----
        self.cls_head = nn.Linear(embed_dim, n_classes)
        # diff_head : per-token velocity for mode 'cls+diff' (input-space denoising)
        self.diff_head = nn.Linear(embed_dim, self.token_dim)
        # mask projection + head : for mode 'cls+mask' the auxiliary diffusion
        # operates over the dataset's *soft segmentation masks* rather than over
        # input patches. n_mask_channels is the number of finding masks emitted
        # by data.render (see models.dataset.MASK_LABELS).
        self.n_mask_channels = 16
        self.mask_proj = nn.Linear(self.n_mask_channels, embed_dim)
        self.mask_head = nn.Linear(embed_dim, self.n_mask_channels)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    # ----- variant-specific input → flat native tokens [B, N, token_dim] -----

    def _native_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, H, W] in [-1, 1] grayscale. Returns flattened native tokens."""
        if self.variant == "pixel":
            return _patchify(x, self.patch_size)  # gradient-attached
        if self.variant == "encoder":
            x_rgb = x.repeat(1, 3, 1, 1)  # gray -> rgb replicate
            x_rgb = (x_rgb + 1) / 2
            x_rgb = (x_rgb - self._mean) / self._std
            with torch.no_grad():
                feats = self.frozen_encoder.forward_features(x_rgb)
            return feats["x_norm_patchtokens"].detach()  # [B, N, dino_dim]
        if self.variant == "vae":
            x_rgb = x.repeat(1, 3, 1, 1)  # SD VAE expects 3-channel in [-1, 1]
            with torch.no_grad():
                lat = self.vae.encode(x_rgb).latent_dist.sample()
                lat = lat * self.vae.config.scaling_factor
            return _patchify(lat, self.vae_patch).detach()  # [B, N, 4*ps*ps]
        raise RuntimeError

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------- forward ----------

    @torch.no_grad()
    def sample_mask(self, x_image: torch.Tensor,
                    n_steps: int = 20) -> torch.Tensor:
        """Generate predicted soft segmentation masks for ``x_image`` via
        Euler integration of the rectified-flow vector field defined by
        ``mask_head``.

        Trajectory: starts from pure noise at t=1 and integrates the
        predicted velocity ``noise − GT_mask`` backwards to t=0, where the
        result is the model's mask prediction. ``n_steps`` controls the
        Euler granularity; 20 is enough for reasonable visual quality.

        Returns ``[B, n_mask_channels, g, g]`` where ``g = sqrt(N)`` is
        the side of the token grid (so a ``320 × 320`` image at
        ``patch_size=14`` yields a ``16 × 16`` predicted mask grid).
        """
        was_training = self.training
        self.eval()
        native = self._native_tokens(x_image)
        B, N, _ = native.shape
        grid_side = int(round(N ** 0.5))
        image_tokens = self.token_proj(native)

        x = torch.randn(B, N, self.n_mask_channels,
                        device=x_image.device, dtype=image_tokens.dtype)
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=x_image.device)
        for i in range(n_steps):
            t_val = ts[i]
            dt = (ts[i] - ts[i + 1]).item()
            t_batch = torch.full((B,), float(t_val),
                                  device=x_image.device, dtype=image_tokens.dtype)
            mask_emb = self.mask_proj(x)
            t_e = self.t_proj(t_batch).unsqueeze(1)
            tokens = image_tokens + mask_emb + t_e
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, tokens], dim=1) + self.pos_embed
            for blk in self.blocks:
                seq = blk(seq)
            seq = self.norm(seq)
            v = self.mask_head(seq[:, 1:])  # velocity = noise − x_clean
            x = x - dt * v                  # integrate backwards in t

        if was_training:
            self.train()
        return x.permute(0, 2, 1).reshape(B, self.n_mask_channels,
                                           grid_side, grid_side)

    def forward(self, x: torch.Tensor, mode: str = "cls",
                mask_ratio: float = 0.5,
                mask_target: torch.Tensor | None = None) -> dict:
        """Forward pass.

        Modes:
            ``cls``      — classification head only, clean image input.
            ``cls+diff`` — classification + auxiliary rectified-flow head that
                           denoises *masked input patches* (each variant
                           denoises in its own native token space).
            ``cls+mask`` — classification + auxiliary rectified-flow head that
                           denoises the *dataset's soft segmentation masks*.
                           Requires ``mask_target`` of shape
                           ``[B, n_mask_channels, H, W]`` (typically the
                           original 320×320 grid; it is downsampled to the
                           token grid here).

        Returns:
            cls_logits : [B, n_classes]
            flow_loss  : scalar (only when mode != 'cls')
        """
        native = self._native_tokens(x)  # [B, N, token_dim]
        B, N, D_n = native.shape

        if mode == "cls":
            tokens = self.token_proj(native)  # [B, N, embed_dim]
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, tokens], dim=1) + self.pos_embed
            for blk in self.blocks:
                seq = blk(seq)
            seq = self.norm(seq)
            return {"cls_logits": self.cls_head(seq[:, 0])}

        if mode == "cls+diff":
            # Sample binary mask (n_mask of N per sample) and per-sample t ∈ [0,1]
            n_mask = max(1, int(round(mask_ratio * N)))
            scores = torch.rand(B, N, device=x.device)
            order = scores.argsort(dim=1)
            mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            mask.scatter_(1, order[:, :n_mask], True)
            t = torch.rand(B, device=x.device)
            noise = torch.randn_like(native)

            # Rectified-flow x_t = (1-t) x_0 + t ε  applied only at masked
            xt_native = (1.0 - t.view(B, 1, 1)) * native + t.view(B, 1, 1) * noise
            native_in = torch.where(mask.unsqueeze(-1), xt_native, native)
            tokens = self.token_proj(native_in)

            # Add t-embedding to masked positions to communicate the noise level
            t_e = self.t_proj(t).unsqueeze(1)        # [B, 1, embed_dim]
            tokens = tokens + mask.unsqueeze(-1).float() * t_e

            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, tokens], dim=1) + self.pos_embed
            for blk in self.blocks:
                seq = blk(seq)
            seq = self.norm(seq)
            cls_logits = self.cls_head(seq[:, 0])

            # Velocity prediction at every position; only masked positions count
            v_pred = self.diff_head(seq[:, 1:])      # [B, N, D_n]
            v_target = noise - native                # [B, N, D_n]
            sq_err = ((v_pred - v_target) ** 2).mean(dim=-1)  # [B, N]
            denom = mask.float().sum().clamp(min=1.0)
            flow_loss = (sq_err * mask.float()).sum() / denom
            return {"cls_logits": cls_logits, "flow_loss": flow_loss}

        if mode == "cls+mask":
            grid_side = int(round(N ** 0.5))
            assert grid_side * grid_side == N, \
                f"non-square token grid (N={N}) is unsupported by cls+mask"
            if mask_target is not None:
                # Training: rectified-flow target lives in the soft-mask space
                mask_grid = F.adaptive_avg_pool2d(
                    mask_target.float(), output_size=(grid_side, grid_side))
                mask_tokens = mask_grid.flatten(2).permute(0, 2, 1)  # [B, N, C]
                t = torch.rand(B, device=x.device)
                noise = torch.randn_like(mask_tokens)
                noisy_mask = (1.0 - t.view(B, 1, 1)) * mask_tokens + \
                              t.view(B, 1, 1) * noise
            else:
                # Inference: there is no GT mask to condition on, so we feed
                # the backbone "pure noise at t=1" — exactly the slice of the
                # training distribution that contains *zero mask information*.
                # This keeps the input distribution consistent between train
                # and eval (without it the eval forward sees image-only tokens
                # and the cls head outputs garbage).
                mask_tokens = None
                noise = torch.randn(B, N, self.n_mask_channels,
                                     device=x.device, dtype=native.dtype)
                t = torch.ones(B, device=x.device)
                noisy_mask = noise

            # Backbone sees clean image tokens + (noised mask projection) +
            # broadcast t embedding so it can condition on the noise level.
            image_tokens = self.token_proj(native)        # [B, N, D]
            mask_emb = self.mask_proj(noisy_mask)         # [B, N, D]
            t_e = self.t_proj(t).unsqueeze(1)             # [B, 1, D]
            tokens = image_tokens + mask_emb + t_e

            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, tokens], dim=1) + self.pos_embed
            for blk in self.blocks:
                seq = blk(seq)
            seq = self.norm(seq)
            cls_logits = self.cls_head(seq[:, 0])

            if mask_tokens is None:
                return {"cls_logits": cls_logits}

            v_pred = self.mask_head(seq[:, 1:])           # [B, N, n_mask_channels]
            v_target = noise - mask_tokens                # [B, N, n_mask_channels]
            flow_loss = ((v_pred - v_target) ** 2).mean()
            return {"cls_logits": cls_logits, "flow_loss": flow_loss}

        raise ValueError(f"unknown mode: {mode}")
