import json
import os
from datetime import datetime, timezone
from pathlib import Path

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


_ARCHITECTURE_PARAMS = {"hidden_dim", "encoder_depth", "attn_nhead", "attn_ff_mult"}


def _sample_hyperparams(trial, search_space: dict, skip_arch: bool = False) -> dict:
    """sample hyperparameters from an Optuna trial using a YAML-derived search space."""
    overrides: dict = {}
    for name, spec in search_space.items():
        if skip_arch and name in _ARCHITECTURE_PARAMS:
            continue
        param_type = spec["type"]
        if param_type == "float":
            overrides[name] = trial.suggest_float(
                name, low=spec["low"], high=spec["high"], log=spec.get("log", False),
            )
        elif param_type == "int":
            overrides[name] = trial.suggest_int(
                name, low=spec["low"], high=spec["high"], step=spec.get("step", 1),
            )
        elif param_type == "categorical":
            overrides[name] = trial.suggest_categorical(name, choices=spec["choices"])
        else:
            raise ValueError(f"Unknown param type '{param_type}' for '{name}'")
    return overrides


def train(overrides: dict | None = None) -> float:
    """run a single training loop. returns best validation_hm_iou.

    overrides: dict of arg names -> values that replace the CLI defaults
    (used by Optuna to inject sampled hyperparams).
    """
    overrides = overrides or {}

    # resolve effective values: CLI args with optional overrides
    def _get(name: str):
        return overrides[name] if name in overrides else getattr(args, name)

    logger = None
    if not args.no_wandb:
        if overrides and "_trial" in overrides:
            # optuna mode: WeightsAndBiasesCallback already created the run
            import wandb
            trial_num = overrides["_trial"].number
            run_name = f"optuna-{args.run_name}-t{trial_num}" if args.run_name else f"optuna-t{trial_num}"
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

    dataset_type = DatasetType(args.dataset)
    train_loader = get_dataloader(
        Split.TRAIN, dataset_type=dataset_type,
        batch_size=_get("batch_size"),
    )
    validation_loader = get_dataloader(Split.DEV, dataset_type=dataset_type, batch_size=1)

    # optionally mix in DGS dev split for validation
    if args.val_dgs and dataset_type != DatasetType.DGS:
        from sign_language_segmentation.datasets.dgs.dataset import DGSSegmentationDataset
        dgs_dev = DGSSegmentationDataset(
            corpus_dir=args.corpus,
            poses_dir=args.poses,
            split=Split.DEV,
            num_frames=args.num_frames,
            velocity=args.velocity,
            fps_aug=args.fps_aug,
            frame_dropout=_get("frame_dropout"),
            body_part_dropout=0.0,
        )
        combined_val = ConcatDataset([validation_loader.dataset, dgs_dev])
        validation_loader = DataLoader(
            combined_val,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
        )

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
        best_ckpt = os.path.join(model_dir, "best.ckpt")
        if os.path.exists(best_ckpt):
            print(f"\nBest checkpoint: {best_ckpt}")
            print("Copy to dist/2026/best.ckpt to deploy for inference.")

    return best_val


if __name__ == '__main__':
    if args.optuna:
        import optuna
        import yaml

        search_space_path = Path(args.optuna)
        if not search_space_path.exists():
            raise FileNotFoundError(f"Optuna search space file not found: {search_space_path}")

        with open(search_space_path) as f:
            search_space = yaml.safe_load(f)

        skip_arch = args.finetune_from is not None
        if skip_arch:
            skipped = _ARCHITECTURE_PARAMS & search_space.keys()
            if skipped:
                print(f"Fine-tuning: skipping architecture params {skipped}")

        wandb_kwargs = None
        wandbc = None
        if not args.no_wandb:
            from optuna_integration import WeightsAndBiasesCallback
            wandb_kwargs = {
                "entity": args.wandb_entity,
                "project": args.wandb_project,
            }
            wandbc = WeightsAndBiasesCallback(
                metric_name="validation_hm_iou",
                wandb_kwargs=wandb_kwargs,
                as_multirun=True,
            )

        def objective(trial: optuna.Trial) -> float:
            overrides = _sample_hyperparams(trial=trial, search_space=search_space, skip_arch=skip_arch)
            overrides["_trial"] = trial
            return train(overrides=overrides)

        if wandbc:
            objective = wandbc.track_in_wandb()(objective)

        study = optuna.create_study(direction="maximize", study_name="segmentation-hpo")
        study.optimize(
            objective,
            n_trials=args.optuna_trials,
            callbacks=[wandbc] if wandbc else [],
        )

        print("\n--- Optuna results ---")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation_hm_iou: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    else:
        train()
