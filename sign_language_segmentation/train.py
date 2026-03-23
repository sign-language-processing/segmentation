import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from sign_language_segmentation.args import args
from sign_language_segmentation.data.dataset import DGSSegmentationDataset, Split, collate_fn
from sign_language_segmentation.model.model import PoseTaggingModel


def get_dataloader(split: Split, batch_size: int = None, num_frames: int = None,
                   persistent_workers: bool = True) -> DataLoader:
    dataset = DGSSegmentationDataset(
        corpus_dir=args.corpus,
        poses_dir=args.poses,
        split=split,
        num_frames=num_frames if num_frames is not None else args.num_frames,
        velocity=args.velocity,
        fps_aug=args.fps_aug,
        frame_dropout=args.frame_dropout,
        body_part_dropout=args.body_part_dropout if split == "train" else 0.0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=persistent_workers,
        prefetch_factor=4 if persistent_workers else 2,
        pin_memory=True,
    )


if __name__ == '__main__':
    LOGGER = None
    if not args.no_wandb:
        LOGGER = WandbLogger(project="pose-to-segments", log_model=False,
                             offline=False, name=args.run_name,
                             save_dir=args.wandb_dir)

    train_loader = get_dataloader("train")
    validation_loader = get_dataloader("dev", batch_size=1)

    example_datum = train_loader.dataset[0]
    pose_joints, pose_dims = example_datum["pose"].shape[1:3]

    steps_per_epoch = len(train_loader)
    print(f"Steps/epoch: {steps_per_epoch}")

    model_kwargs = dict(
        pose_dims=(pose_joints, pose_dims),
        hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth,
        learning_rate=args.learning_rate,
        steps_per_epoch=steps_per_epoch,
        max_epochs=args.epochs,
        dice_loss_weight=args.dice_loss_weight,
        optimizer=args.optimizer,
        attn_nhead=args.attn_nhead,
        attn_ff_mult=args.attn_ff_mult,
        attn_dropout=args.attn_dropout,
        fps_aug=args.fps_aug,
        frame_dropout=args.frame_dropout,
        num_frames=args.num_frames,
    )

    if args.finetune_from:
        print(f"Fine-tuning from: {args.finetune_from}")
        model = PoseTaggingModel.load_from_checkpoint(args.finetune_from, **model_kwargs)
    else:
        model = PoseTaggingModel(**model_kwargs)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    model_dir = f"models/{args.run_name or 'model'}"
    os.makedirs(model_dir, exist_ok=True)

    monitor_metric = "validation_hm_iou"

    callbacks = [
        EarlyStopping(monitor=monitor_metric, patience=args.patience, verbose=True, mode='max'),
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=model_dir,
            filename='best',
            verbose=True,
            save_top_k=1,
            save_last=True,
            monitor=monitor_metric,
            every_n_epochs=1,
            mode='max',
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_time=args.max_time,
        precision="bf16-mixed",
        logger=LOGGER,
        callbacks=callbacks,
        log_every_n_steps=10,
        accelerator=args.device,
        devices=args.gpus,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    best_ckpt = os.path.join(model_dir, "best.ckpt")
    if os.path.exists(best_ckpt):
        print(f"\nBest checkpoint: {best_ckpt}")
        print("Copy to dist/2026/best.ckpt to deploy for inference.")
