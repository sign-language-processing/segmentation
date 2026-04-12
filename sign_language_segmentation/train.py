import json
import os
from datetime import datetime, timezone

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from sign_language_segmentation.args import args
from sign_language_segmentation.datasets.common import DatasetType, Split, collate_fn
from sign_language_segmentation.model.model import PoseTaggingModel


def _build_dataset(dataset_type: DatasetType, split: Split, num_frames: int) -> Dataset:
    """build dataset(s) based on type selection."""
    augment_kwargs = dict(
        num_frames=num_frames,
        velocity=args.velocity,
        fps_aug=args.fps_aug,
        frame_dropout=args.frame_dropout,
        body_part_dropout=args.body_part_dropout if split == Split.TRAIN else 0.0,
    )

    if dataset_type == DatasetType.DGS:
        from sign_language_segmentation.datasets.dgs.dataset import DGSSegmentationDataset
        return DGSSegmentationDataset(
            corpus_dir=args.corpus,
            poses_dir=args.poses,
            split=split,
            **augment_kwargs,
        )

    if dataset_type == DatasetType.PLATFORM:
        from sign_language_segmentation.datasets.annotation_platform.dataset import (
            AnnotationPlatformSegmentationDataset,
        )
        if not args.annotations_path:
            raise ValueError("--annotations_path required for platform dataset")
        return AnnotationPlatformSegmentationDataset(
            annotations_path=args.annotations_path,
            poses_dir=args.poses,
            split=split,
            quality_percentile=args.quality_percentile,
            **augment_kwargs,
        )

    if dataset_type == DatasetType.COMBINED:
        from sign_language_segmentation.datasets.annotation_platform.dataset import (
            AnnotationPlatformSegmentationDataset,
        )
        from sign_language_segmentation.datasets.dgs.dataset import DGSSegmentationDataset
        if not args.annotations_path:
            raise ValueError("--annotations_path required for combined dataset")
        dgs = DGSSegmentationDataset(
            corpus_dir=args.corpus,
            poses_dir=args.poses,
            split=split,
            **augment_kwargs,
        )
        platform = AnnotationPlatformSegmentationDataset(
            annotations_path=args.annotations_path,
            poses_dir=args.poses,
            split=split,
            quality_percentile=args.quality_percentile,
            **augment_kwargs,
        )
        return ConcatDataset([dgs, platform])

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def _collect_split_manifest(dataset: Dataset, dataset_type: DatasetType) -> dict:
    """collect split manifest from dataset(s)."""
    manifests: list[dict] = []
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if hasattr(ds, "get_split_manifest"):
                manifests.append(ds.get_split_manifest())
    elif hasattr(dataset, "get_split_manifest"):
        manifests.append(dataset.get_split_manifest())
    return {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_type": dataset_type.value,
        "manifests": manifests,
    }


def get_dataloader(
    split: Split,
    dataset_type: DatasetType,
    batch_size: int | None = None,
    num_frames: int | None = None,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = _build_dataset(
        dataset_type=dataset_type,
        split=split,
        num_frames=num_frames if num_frames is not None else args.num_frames,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=(split == Split.TRAIN),
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=persistent_workers,
        prefetch_factor=4 if persistent_workers else 2,
        pin_memory=True,
    )


if __name__ == '__main__':
    LOGGER = None
    if not args.no_wandb:
        LOGGER = WandbLogger(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_name,
            save_dir=args.wandb_dir,
            log_model=False,
        )
        LOGGER.log_hyperparams(vars(args))

    dataset_type = DatasetType(args.dataset)
    train_loader = get_dataloader(Split.TRAIN, dataset_type=dataset_type)
    validation_loader = get_dataloader(Split.DEV, dataset_type=dataset_type, batch_size=1)

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

    model_dir = f"dist/{args.run_name or 'model'}"
    os.makedirs(model_dir, exist_ok=True)

    # write split manifest
    manifest = _collect_split_manifest(train_loader.dataset, dataset_type)
    val_manifest = _collect_split_manifest(validation_loader.dataset, dataset_type)
    manifest["manifests"].extend(val_manifest["manifests"])
    manifest_path = os.path.join(model_dir, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Split manifest: {manifest_path}")

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
