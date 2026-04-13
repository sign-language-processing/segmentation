import json
from datetime import datetime, timezone
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from sign_language_segmentation.args import args
from sign_language_segmentation.datasets.dgs.dataset import DGSSegmentationDataset
from sign_language_segmentation.datasets.common import Split, collate_fn
from sign_language_segmentation.model.model import PoseTaggingModel


def get_dataloader(
    split: Split,
    batch_size: int | None = None,
    num_frames: int | None = None,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = DGSSegmentationDataset(
        corpus_dir=args.corpus,
        poses_dir=args.poses,
        split=split,
        num_frames=num_frames if num_frames is not None else args.num_frames,
        velocity=args.velocity,
        fps_aug=args.fps_aug,
        frame_dropout=args.frame_dropout,
        body_part_dropout=args.body_part_dropout if split == Split.TRAIN else 0.0,
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


_DEFAULT_MONITOR_METRIC = "validation_hm_iou"


def train(overrides: dict | None = None, monitor_metric: str = _DEFAULT_MONITOR_METRIC) -> float:
    """run a single training loop. returns best monitor_metric value.

    Without Optuna, all hyperparameters come from CLI args (see args.py for
    defaults). With Optuna, sampled values are passed via overrides and
    monitor_metric is read from the YAML search space config.

    overrides: dict of arg names -> values that replace the CLI defaults
    (used by Optuna to inject sampled hyperparams).
    monitor_metric: validation metric to maximize and monitor for early stopping.
    """
    overrides = overrides or {}

    def _get(name: str):
        return overrides[name] if name in overrides else getattr(args, name)

    logger = None
    if not args.no_wandb:
        if overrides and "_trial" in overrides:
            import wandb
            trial_num = overrides["_trial"].number
            run_name = f"{args.run_name}-t{trial_num}"
            wandb.run.name = run_name
            logger = WandbLogger(experiment=wandb.run)
        else:
            logger = WandbLogger(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.run_name,
                save_dir=args.wandb_dir,
                log_model=False,
            )
        effective_args = {**vars(args), **overrides}
        logger.log_hyperparams(effective_args)

    train_loader = get_dataloader(Split.TRAIN, batch_size=_get("batch_size"))
    validation_loader = get_dataloader(Split.DEV, batch_size=1)

    example_datum = train_loader.dataset[0]
    pose_joints, pose_dims = example_datum["pose"].shape[1:3]

    steps_per_epoch = len(train_loader)
    print(f"Steps/epoch: {steps_per_epoch}")

    model_kwargs = dict(
        pose_dims=(pose_joints, pose_dims),
        hidden_dim=_get("hidden_dim"),
        encoder_depth=_get("encoder_depth"),
        learning_rate=_get("learning_rate"),
        lr_scale_backbone=_get("lr_scale_backbone"),
        steps_per_epoch=steps_per_epoch,
        max_epochs=_get("epochs"),
        dice_loss_weight=_get("dice_loss_weight"),
        optimizer=_get("optimizer"),
        attn_nhead=_get("attn_nhead"),
        attn_ff_mult=_get("attn_ff_mult"),
        attn_dropout=_get("attn_dropout"),
        fps_aug=args.fps_aug,
        frame_dropout=_get("frame_dropout"),
        num_frames=args.num_frames,
    )

    if args.finetune_from:
        print(f"Fine-tuning from: {args.finetune_from}")
        model = PoseTaggingModel.load_from_checkpoint(args.finetune_from, **model_kwargs)
    else:
        model = PoseTaggingModel(**model_kwargs)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    model_dir = Path("dist") / (args.run_name or "model")
    model_dir.mkdir(parents=True, exist_ok=True)

    # write split manifest
    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset": "dgs",
    }
    manifest_path = model_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Split manifest: {manifest_path}")

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

    # add Optuna pruning callback when running a sweep
    if overrides and "_trial" in overrides:
        from optuna_integration import PyTorchLightningPruningCallback
        callbacks.append(PyTorchLightningPruningCallback(
            trial=overrides["_trial"], monitor=monitor_metric,
        ))

    trainer = pl.Trainer(
        max_epochs=_get("epochs"),
        max_time=args.max_time,
        precision="bf16-mixed",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        accelerator=args.device,
        devices=args.gpus,
        enable_progress_bar=not overrides,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    best_score = trainer.callback_metrics.get(monitor_metric)
    best_val = float(best_score) if best_score is not None else 0.0

    if not overrides:
        best_ckpt = model_dir / "best.ckpt"
        if best_ckpt.exists():
            print(f"\nBest checkpoint: {best_ckpt}")
            print("Copy to dist/2026/best.ckpt to deploy for inference.")

    return best_val


if __name__ == '__main__':
    if args.optuna:
        if not args.run_name:
            raise ValueError("--run_name is required when using --optuna")

        from functools import partial

        from sign_language_segmentation.hpo import load_search_space, run_study

        search_space, metric = load_search_space(
            path=args.optuna,
            skip_architecture=args.finetune_from is not None,
        )

        study = run_study(
            train_fn=partial(train, monitor_metric=metric),
            search_space=search_space,
            n_trials=args.optuna_trials,
            metric_name=metric,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            no_wandb=args.no_wandb,
        )

        print("\n--- Optuna results ---")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best {metric}: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    else:
        train()
