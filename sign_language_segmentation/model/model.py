import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from sign_language_segmentation.data.utils import BIO
from sign_language_segmentation.metrics import likeliest_probs_to_segments, segment_IoU, bio_labels_to_segments
from sign_language_segmentation.model.pose_encoder import ConvDef, PoseEncoderUNetBlock


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with Rotary Position Embedding (RoPE).

    RoPE rotates Q and K by position-dependent angles so that attention scores
    depend only on relative position, not absolute — generalises well across
    chunk boundaries during chunked inference.

    Timestamps are expected in seconds; they are scaled by reference_fps=50
    internally so that relative positions are expressed in "50fps frame units"
    (i.e. two frames 0.02s apart → relative position 1, same as consecutive
    frames at 50fps).
    """
    REFERENCE_FPS = 50.0

    def __init__(self, hidden_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % nhead == 0, f"hidden_dim {hidden_dim} must be divisible by nhead {nhead}"
        self.nhead = nhead
        self.head_dim = hidden_dim // nhead

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def _compute_rope(self, timestamps: torch.Tensor):
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0)
        # Scale seconds → 50fps-equivalent frame units before computing frequencies.
        freqs = (timestamps * self.REFERENCE_FPS).unsqueeze(-1).float() * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        if timestamps is None:
            timestamps = torch.arange(T, device=x.device, dtype=torch.float32)
        cos, sin = self._compute_rope(timestamps)

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x + self.out_proj(attn_out.transpose(1, 2).reshape(B, T, D))
        x = x + self.ffn(self.norm2(x))
        return x


class ClassifierHead(nn.Module):
    """Two-layer MLP classifier: hidden_dim → hidden_dim → num_classes.

    Decouples the sign and phrase heads so they don't share a linear map
    that would force a direct trade-off between their outputs.
    """
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoseTaggingModel(pl.LightningModule):
    """CNN-medium-attn model with RoPE for sign language segmentation.

    Architecture: two-stage UNet CNN spatial encoder + N-layer RoPE transformer.
    Trained jointly on sign (gloss) and phrase (sentence) BIO tagging.
    Validated by harmonic mean of sign and phrase IoU to prevent over-optimising
    one head at the expense of the other.
    """

    REFERENCE_FPS = RoPETransformerEncoderLayer.REFERENCE_FPS

    def __init__(self,
                 pose_dims: (int, int) = (178, 3),
                 hidden_dim: int = 384,
                 encoder_depth: int = 6,
                 num_classes: int = len(BIO),
                 learning_rate: float = 1e-3,
                 steps_per_epoch: int = 100,
                 max_epochs: int = 200,
                 # Loss
                 dice_loss_weight: float = 1.0,
                 # Attention
                 attn_nhead: int = 8,
                 attn_ff_mult: int = 2,
                 attn_dropout: float = 0.1,
                 # Optimizer
                 optimizer: str = "adamw-onecycle",
                 # Stored as hparams for evaluate.py (not used in forward)
                 fps_aug: bool = False,
                 frame_dropout: float = 0.0,
                 num_frames: int = 1024,
                 **kwargs):  # absorb deprecated hparams from old checkpoints
        super().__init__()
        self.save_hyperparameters(ignore=list(kwargs.keys()))

        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self._optimizer_name = optimizer

        self.sign_loss_fn = nn.NLLLoss(reduction='none')
        self.phrase_loss_fn = nn.NLLLoss(reduction='none')

        # Stage 1: spatial compression over joints (per-frame)
        # Stage 2: temporal context (across frames)
        self.frame_cnn = nn.Sequential(
            PoseEncoderUNetBlock(input_size=pose_dims[0], output_size=hidden_dim, convolutions=[
                ConvDef(in_channels=pose_dims[1], out_channels=16, kernel_size=5, stride=1),
                ConvDef(in_channels=16, out_channels=32, kernel_size=11, stride=1),
                ConvDef(in_channels=32, out_channels=64, kernel_size=21, stride=2),
            ]),
            Unsqueeze(dim=-1),
            PoseEncoderUNetBlock(input_size=hidden_dim, output_size=hidden_dim, convolutions=[
                ConvDef(in_channels=1, out_channels=16, kernel_size=5, stride=1),
                ConvDef(in_channels=16, out_channels=32, kernel_size=11, stride=2),
                ConvDef(in_channels=32, out_channels=64, kernel_size=21, stride=2),
                ConvDef(in_channels=64, out_channels=128, kernel_size=21, stride=2),
            ]),
        )
        self.input_norm = nn.RMSNorm(hidden_dim)
        self.encoder_attn = nn.ModuleList([
            RoPETransformerEncoderLayer(hidden_dim, attn_nhead, hidden_dim * attn_ff_mult, attn_dropout)
            for _ in range(encoder_depth)
        ])

        self.sign_bio_head = ClassifierHead(hidden_dim, num_classes)
        self.sentence_bio_head = ClassifierHead(hidden_dim, num_classes)

    def encode(self, pose_data: torch.Tensor, timestamps: torch.Tensor = None) -> torch.Tensor:
        x = self.input_norm(self.frame_cnn(pose_data))  # (B, T, hidden_dim)
        B, T, _ = x.shape

        if timestamps is None:
            # Assume 50fps when no timestamps provided (1/50s per frame → *50 → 1 unit/frame).
            ts = (torch.arange(T, device=x.device, dtype=torch.float32) / self.REFERENCE_FPS).unsqueeze(0).expand(B, -1)
        else:
            ts = timestamps.to(x.device)
            if ts.dim() == 1:
                ts = ts.unsqueeze(0).expand(B, -1)

        # Process in training-size chunks so eval context matches training distribution.
        # All chunks are stacked into a single batch and processed in one forward pass
        # through the transformer — much faster than sequential chunk processing for
        # long videos (e.g. 10 chunks → 1 batched call instead of 10 serial calls).
        chunk_size = self.hparams.num_frames
        if T <= chunk_size:
            for layer in self.encoder_attn:
                x = layer(x, ts)
            return x

        # Pad to a multiple of chunk_size, split, batch, process, unpad.
        n_chunks = (T + chunk_size - 1) // chunk_size
        pad_len = n_chunks * chunk_size - T
        x_pad = F.pad(x, (0, 0, 0, pad_len))          # (B, n_chunks*chunk_size, D)
        ts_pad = F.pad(ts, (0, pad_len))               # (B, n_chunks*chunk_size)

        # Reshape: treat each chunk as an independent batch element.
        # B must be 1 at inference (guaranteed by evaluate.py / bin.py).
        x_chunks = x_pad.reshape(n_chunks, chunk_size, x_pad.shape[-1])   # (n_chunks, C, D)
        ts_chunks = ts_pad.reshape(n_chunks, chunk_size)                   # (n_chunks, C)
        for layer in self.encoder_attn:
            x_chunks = layer(x_chunks, ts_chunks)
        return x_chunks.reshape(1, n_chunks * chunk_size, x_pad.shape[-1])[:, :T]

    def forward(self, pose_data: torch.Tensor, timestamps: torch.Tensor = None,
                lengths: torch.Tensor = None) -> dict:
        encoded = self.encode(pose_data, timestamps)
        return {
            "sign": F.log_softmax(self.sign_bio_head(encoded), dim=-1),
            "sentence": F.log_softmax(self.sentence_bio_head(encoded), dim=-1),
        }

    def training_step(self, batch, *_):
        return self.step(batch, name="train")

    def validation_step(self, batch, *_):
        loss = self.step(batch, name="validation")
        with torch.no_grad():
            log_probs = self.forward(batch["pose"], timestamps=batch.get("timestamps"))

            def _compute_iou(probs_key: str, gold_key: str) -> float:
                gold_bio = batch["bio"][gold_key]
                ious = []
                for i in range(len(batch["pose"])):
                    probs_i = log_probs[probs_key][i].cpu()
                    gold_i = gold_bio[i]
                    mask_i = (gold_i != BIO["UNK"])
                    if mask_i.sum() == 0:
                        continue
                    num_frames = int(mask_i.sum())
                    pred_segs = likeliest_probs_to_segments(probs_i[:num_frames])
                    gold_segs = bio_labels_to_segments(gold_i[:num_frames])
                    ious.append(segment_IoU(pred_segs, gold_segs, num_frames))
                return sum(ious) / len(ious) if ious else 0.0

            sign_iou = _compute_iou("sign", "sign")
            phrase_iou = _compute_iou("sentence", "sentence")
            denom = sign_iou + phrase_iou
            hm_iou = (2 * sign_iou * phrase_iou / denom) if denom > 0 else 0.0
            bs = len(batch["pose"])
            self.log("validation_sign_iou", sign_iou, prog_bar=True, batch_size=bs)
            self.log("validation_phrase_iou", phrase_iou, prog_bar=True, batch_size=bs)
            self.log("validation_hm_iou", hm_iou, prog_bar=True, batch_size=bs)
        return loss

    def test_step(self, batch, *_):
        return self.step(batch, name="test")

    def step(self, batch, name: str):
        pose_data = batch["pose"]
        batch_size = len(pose_data)
        log_probs = self.forward(pose_data, timestamps=batch.get("timestamps"))

        total_loss = torch.zeros(1, device=self.device).squeeze()
        for pred_type, loss_fn in [("sign", self.sign_loss_fn), ("sentence", self.phrase_loss_fn)]:
            gold = batch["bio"][pred_type]
            loss = loss_fn(log_probs[pred_type].transpose(1, 2), gold)
            mask = (gold != BIO["UNK"]).float()
            masked_loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            total_loss = total_loss + masked_loss
            self.log(f"{name}_{pred_type}_loss", masked_loss, batch_size=batch_size,
                     prog_bar=(pred_type == "sign"))

        self.log(f"{name}_loss", total_loss, batch_size=batch_size)

        if self.hparams.dice_loss_weight > 0.0:
            sign_gold = batch["bio"]["sign"]
            mask = (sign_gold != BIO["UNK"]).float()
            sign_probs = log_probs["sign"].exp()
            pred_sign = (sign_probs[:, :, BIO["B"]] + sign_probs[:, :, BIO["I"]]) * mask
            gold_sign = (sign_gold >= BIO["B"]).float() * mask
            dice_num = 2.0 * (pred_sign * gold_sign).sum()
            dice_den = pred_sign.sum() + gold_sign.sum() + 1e-6
            dice_loss = (1.0 - dice_num / dice_den) * self.hparams.dice_loss_weight
            total_loss = total_loss + dice_loss
            self.log(f"{name}_dice_loss", dice_loss, batch_size=batch_size)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        if self._optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self._optimizer_name in ("adamw-onecycle", "adamw", None):
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=self.max_epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=0.1,
                div_factor=10,
                final_div_factor=100,
                anneal_strategy='cos',
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        elif self._optimizer_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs * self.steps_per_epoch,
                eta_min=self.learning_rate / 100,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        elif self._optimizer_name == "constant":
            return {"optimizer": optimizer}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.7)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "validation_sign_loss"}}
