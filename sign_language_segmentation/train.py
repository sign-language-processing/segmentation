import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from sign_language_segmentation.args import args
from sign_language_segmentation.data.dataset import SegmentationDataset, Split
from sign_language_segmentation.model.model import PoseTaggingModel


def get_dataset(split: Split) -> DataLoader:
    dataset = SegmentationDataset(data_dir=args.dataset, split=split, num_frames=args.num_frames_per_item)
    return DataLoader(dataset, batch_size=args.batch_size)


if __name__ == '__main__':
    LOGGER = None
    if not args.no_wandb:
        LOGGER = WandbLogger(project="pose-to-segments", log_model=False,
                             offline=False, name=args.run_name,
                             save_dir=args.wandb_dir)

    train_loader = get_dataset("train")
    validation_loader = get_dataset("dev")

    example_datum = next(iter(train_loader))
    _, _, pose_joints, pose_dims = example_datum["pose"].shape

    model = PoseTaggingModel(
        pose_dims=(pose_joints, pose_dims),
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate
    )

    callbacks = [
        EarlyStopping(monitor='train_loss', patience=args.patience, verbose=True, mode='min'),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    if LOGGER is not None:
        os.makedirs("models", exist_ok=True)

        callbacks.append(
            ModelCheckpoint(dirpath=f"models/{LOGGER.experiment.name}",
                            filename='best',
                            verbose=True,
                            save_top_k=1,
                            save_last=True,
                            monitor='train_sign_sign_loss', #'validation_frame_f1_avg',
                            every_n_epochs=1,
                            mode='max'))

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=LOGGER,
                         callbacks=callbacks,
                         log_every_n_steps=10,
                         accelerator=args.device,
                         val_check_interval=100,
                         devices=args.gpus)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    #
    # if args.save_jit:
    #     # TODO: how to automatically load the best weights like above?
    #     pose_data = torch.randn((1, 100, *model.pose_dims))
    #     traced_cell = torch.jit.trace(model, tuple([pose_data]), strict=False)
    #     model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../dist", f"model_{args.run_name}.pth")
    #     torch.jit.save(traced_cell, model_path)
