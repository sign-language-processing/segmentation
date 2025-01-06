import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from sign_language_segmentation.args import args
from sign_language_segmentation.data.utils import BIO
from sign_language_segmentation.metrics import frame_accuracy, frame_f1, frame_precision, frame_recall
from sign_language_segmentation.model.pose_encoder import ConvDef, PoseEncoderUNetBlock


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class PoseTaggingModel(pl.LightningModule):

    def __init__(self,
                 pose_dims: (int, int) = (178, 3),
                 hidden_dim: int = 256,
                 num_classes=len(BIO),
                 learning_rate=1e-3):
        super().__init__()

        self.learning_rate = learning_rate

        self.pose_dims = pose_dims

        self.encoder = nn.Sequential(
            # Shallow pose encoder with little context
            PoseEncoderUNetBlock(input_size=pose_dims[0], output_size=hidden_dim, convolutions=[
                ConvDef(in_channels=3, out_channels=8, kernel_size=5, stride=1),
                ConvDef(in_channels=8, out_channels=16, kernel_size=11, stride=1)
            ]),
            # Deep pose encoder with more context
            Unsqueeze(dim=-1),
            PoseEncoderUNetBlock(input_size=hidden_dim, output_size=hidden_dim, convolutions=[
                ConvDef(in_channels=1, out_channels=8, kernel_size=5, stride=1),
                ConvDef(in_channels=8, out_channels=16, kernel_size=11, stride=2),
                ConvDef(in_channels=16, out_channels=32, kernel_size=21, stride=2),
                ConvDef(in_channels=32, out_channels=64, kernel_size=21, stride=2),
            ]),
        )

        self.sign_bio_head = nn.Linear(hidden_dim, num_classes)
        self.sentence_bio_head = nn.Linear(hidden_dim, num_classes)

        self.loss_function = nn.NLLLoss()

    def forward(self, pose_data: torch.Tensor):
        pose_encoding = self.encoder(pose_data)

        sign_bio_logits = self.sign_bio_head(pose_encoding)
        sentence_bio_logits = self.sentence_bio_head(pose_encoding)

        return {
            "sign": F.log_softmax(sign_bio_logits, dim=-1),
            "sentence": F.log_softmax(sentence_bio_logits, dim=-1)
        }

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="validation")

    def test_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="test")

    def get_metrics(self, name: str, gold: torch.Tensor, log_probs: torch.Tensor):
        unmasked_loss = self.loss_function(log_probs.transpose(1, 2), gold)
        mask = gold != BIO["UNK"]  # Mask out UNKs
        masked_loss = unmasked_loss * mask

        unbatched_gold = gold.view(-1)
        unbatched_log_probs = log_probs.view(-1, log_probs.shape[-1])

        return {
            f"{name}_loss": masked_loss.sum(),
            # f"{name}_frame_f1": frame_f1(unbatched_log_probs, unbatched_gold),
            # f"{name}_frame_precision": frame_precision(unbatched_log_probs, unbatched_gold),
            # f"{name}_frame_recall": frame_recall(unbatched_log_probs, unbatched_gold),
            # f"{name}_frame_accuracy": frame_accuracy(unbatched_log_probs, unbatched_gold),
        }

    def step(self, batch, *unused_args, name: str):
        pose_data = batch["pose"]
        batch_size = len(pose_data)

        log_probs = self.forward(pose_data)

        losses = []
        for pred_type in ["sign", "sentence"]:
            metrics = self.get_metrics(pred_type, batch["bio"][pred_type], log_probs[pred_type])
            losses.append(metrics[f"{pred_type}_loss"])
            for metric, value in metrics.items():
                self.log(f"{name}_{pred_type}_{metric}", value, batch_size=batch_size)

        total_loss = sum(losses)
        self.log(f"{name}_loss", total_loss, batch_size=batch_size)

        return total_loss

    def configure_optimizers(self):
        # AdamW is generally better than Adam for CNNs
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01  # L2 regularization helps prevent overfitting
        )

        # One Cycle LR scheduler with cosine annealing
        # Gradually increases lr from small to large then back down
        # This helps escape local minima and converge to better solutions
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=100,
            steps_per_epoch=args.steps_per_epoch,
            pct_start=0.3,  # Spend 30% of training warming up
            div_factor=25,  # Initial lr = max_lr/25
            final_div_factor=1000,  # Min lr = initial_lr/1000
            anneal_strategy='cos'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update lr every step rather than epoch
            }
        }


if __name__ == "__main__":
    model = PoseTaggingModel(pose_dims=(178, 3), hidden_dim=256)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    pose_data = torch.randn(2, 1000, 178, 3)

    out = model(pose_data)

    print("Input shape:", pose_data.shape)
    print("Output shape:", out["sign"].shape)
