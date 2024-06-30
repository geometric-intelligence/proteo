import math

import torch
import wandb
from pytorch_lightning.callbacks import Callback
from ray import train
import tempfile
from pathlib import Path
import os
import shutil
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train import Checkpoint


class CustomRayCheckpointCallback(Callback):
    """Callback that reports checkpoints to Ray on train epoch end.

    It fetches the latest `trainer.callback_metrics` and reports together with
    the checkpoint on each training epoch end.

    Checkpoints will be saved in the following structure::

        checkpoint_00000*/      Ray Train Checkpoint: * is checkpoint ID
        └─ checkpoint_epoch_*.ckpt      PyTorch Lightning Checkpoint: * is epoch

    If we save every 3 epochs, then we have filenames such as:
    - checkpoint_000000/checkpoint-epoch=0-val_loss=0.81.ckpt
    - checkpoint_000001/checkpoint-epoch=3-val_loss=0.82.ckpt
    - checkpoint_000002/checkpoint-epoch=6-val_loss=0.79.ckpt
    - etc.

    Most of the code is pasted from:
    https://docs.ray.io/en/latest/_modules/ray/train/lightning/_lightning_utils.html#RayTrainReportCallback
    with small modifications, to allow:
    - saving checkpoint every checkpoint_interval
    - adding the epoch to the checkpoint_name
    """

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(self, checkpoint_interval) -> None:
        self.checkpoint_interval = checkpoint_interval
        super().__init__()
        self.trial_name = train.get_context().get_trial_name()
        self.local_rank = train.get_context().get_local_rank()
        self.tmpdir_prefix = Path(tempfile.gettempdir(), self.trial_name).as_posix()
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

        record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYTRAINREPORTCALLBACK, "1")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        if epoch % self.checkpoint_interval != 0:
            return
        tmpdir = Path(self.tmpdir_prefix, str(trainer.current_epoch)).as_posix()
        os.makedirs(tmpdir, exist_ok=True)

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        # Save checkpoint to local, using name pattern similar than
        # when running train.py
        val_loss = metrics["val_loss"]
        checkpoint_name = f"checkpoint-epoch={epoch}-val_loss={val_loss:.4f}.ckpt"
        ckpt_path = Path(tmpdir, checkpoint_name).as_posix()
        trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Report to train session
        checkpoint = Checkpoint.from_directory(tmpdir)
        train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        trainer.strategy.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)


class CustomRayWandbCallback(Callback):
    """Callback that logs losses and plots to Wandb."""

    # def on_train_batch_end(self, trainer, pl_module, outputs, *args):
    #     # FIXME: if loss is not the MSE (regularization, or L1), then sqrt(loss) is not the RMSE
    #     loss = outputs["loss"]
    #     print(f"- train batch end at {pl_module.current_epoch}: outputs={outputs}")
    #     print(f"- train batch end at {pl_module.current_epoch}: loss=outputs['loss']={loss}")
    #     print(f"train_loss in custom ray callback: {loss}")
    #     wandb.log(
    #         {'train_loss': loss, 'train_RMSE': math.sqrt(loss), "epoch": pl_module.current_epoch}
    #     )

    # def on_validation_batch_end(self, trainer, pl_module, outputs, *args):
    #     # FIXME: if loss is not the MSE (regularization, or L1), then sqrt(loss) is not the RMSE
    #     # FIXME: val_loss on wandb might differ from val_loss on Ray because of the sync_dist=True
    #     # val_loss on wandb *on each step* corresponds to what is printed below.
    #     # val_loss on wandb *on each epoch* corresponds to the first step, ie, first batch, whose loss is lower (?)
    #     # what is printed below.
    #     # FIXME: Why doesn't it average the loss across batches?

    #     if not trainer.sanity_checking:
    #         loss = outputs
    #         print(f"- val batch end at epoch {pl_module.current_epoch}: outputs={outputs}")
    #         print(f"- val batch end at epoch {pl_module.current_epoch}: loss=outputs={loss}")
    #         wandb.log(
    #             {'val_loss': loss, "val_RMSE": math.sqrt(loss), "epoch": pl_module.current_epoch}
    #         )

    def on_train_epoch_end(self, trainer, pl_module):
        """Save train predictions, targets, and parameters as histograms.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            Unused. Here for API compliance.
        pl_module : Proteo LightningModule
            Lightning's module for training.
        """
        train_preds = torch.vstack(pl_module.train_preds).detach().cpu()
        train_targets = torch.vstack(pl_module.train_targets).detach().cpu()
        params = torch.concat([p.flatten() for p in pl_module.parameters()]).detach().cpu()
        wandb.log(
            {
                "train_loss_epoch": pl_module.trainer.callback_metrics["train_loss"],
                "train_preds": wandb.Histogram(train_preds),
                "train_targets": wandb.Histogram(train_targets),
                "parameters": wandb.Histogram(params),
                "epoch": pl_module.current_epoch,
            }
        )
        pl_module.train_preds.clear()  # free memory
        pl_module.train_targets.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save val predictions and targets as histograms.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            Lightning's trainer object.
        pl_module : Proteo LightningModule
            Lightning's module for training.
        """
        if not trainer.sanity_checking:
            val_preds = torch.vstack(pl_module.val_preds).detach().cpu()
            val_targets = torch.vstack(pl_module.val_targets).detach().cpu()
            wandb.log(
                {
                    "val_loss_epoch": pl_module.trainer.callback_metrics["val_loss"],
                    "val_preds": wandb.Histogram(val_preds),
                    "val_targets": wandb.Histogram(val_targets),
                    "epoch": pl_module.current_epoch,
                }
            )
        pl_module.val_preds.clear()  # free memory
        pl_module.val_targets.clear()


class CustomRayReportLossCallback(Callback):
    """Callback that reports val loss to Ray."""

    def on_train_batch_end(self, trainer, pl_module, outputs, *args):
        loss = outputs["loss"]
        pl_module.log(
            'train_loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=pl_module.config.batch_size,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args):
        if not trainer.sanity_checking:
            loss = outputs
            pl_module.log(
                'val_loss',
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
                batch_size=pl_module.config.batch_size,
            )
