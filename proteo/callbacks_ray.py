import gc
import math
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from callbacks import get_accuracy, reverse_log_transform
from pytorch_lightning.callbacks import Callback
from ray import train
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train import Checkpoint

from proteo.datasets.ftd import BINARY_Y_VALS_MAP, CONTINOUS_Y_VALS, MULTICLASS_Y_VALS_MAP


# TODO: not using below function
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
    - saving checkpoint every checkpoint_every_n_epochs
    - adding the epoch to the checkpoint_name
    """

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(self, checkpoint_every_n_epochs) -> None:
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        super().__init__()
        self.trial_name = train.get_context().get_trial_name()
        self.local_rank = train.get_context().get_local_rank()
        self.tmpdir_prefix = Path(tempfile.gettempdir(), self.trial_name).as_posix()
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

        record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYTRAINREPORTCALLBACK, "1")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        # if epoch % self.checkpoint_every_n_epochs != 0:
        #    return
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

    # FIXME: if loss is not the MSE (because loss has regularization, or loss=L1),
    # then sqrt(loss) is not the RMSE
    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                gradients = param.grad.detach().cpu()
                wandb.log({"gradients": wandb.Histogram(gradients)})

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
        train_loss = pl_module.trainer.callback_metrics["train_loss"]
        train_RMSE = math.sqrt(train_loss)
        # Log the first graph ([0, :]) of x0, x1, and x2 to see if oversmoothing is happening, aka if all features across 1 person are the same
        x0 = torch.vstack(pl_module.x0).detach().cpu()[0, :]
        x1 = torch.vstack(pl_module.x1).detach().cpu()[0, :]
        x2 = torch.vstack(pl_module.x2).detach().cpu()[0, :]
        multiscale = torch.vstack(pl_module.multiscale).detach().cpu()
        multiscale = (
            torch.norm(multiscale, dim=1).detach().cpu()
        )  # Average the features across the 3 layers per person to get one value per person
        wandb.log(
            {
                "train_loss": train_loss,
                "train_RMSE": train_RMSE,
                "train_preds": wandb.Histogram(train_preds, num_bins=500),
                "train_targets": wandb.Histogram(train_targets, num_bins=500),
                "parameters (weights+biases)": wandb.Histogram(params),
                "x0": wandb.Histogram(x0),
                "x1": wandb.Histogram(x1),
                "x2": wandb.Histogram(x2),
                "multiscale norm for all people": wandb.Histogram(multiscale),
                "epoch": pl_module.current_epoch,
            }
        )
        if pl_module.config.y_val in CONTINOUS_Y_VALS:
            if train_loss < pl_module.min_train_loss:
                pl_module.min_train_loss = train_loss
                scatter_plot_data = [
                    [pred, target] for (pred, target) in zip(train_preds, train_targets)
                ]
                table = wandb.Table(data=scatter_plot_data, columns=["pred", "target"])
                wandb.log(
                    {
                        "Regression Scatter Plot Train": wandb.plot.scatter(
                            table,
                            "pred",
                            "target",
                            title=f"Train Pred vs Train Target Scatter Plot",
                        ),
                        "epoch": pl_module.current_epoch,
                    }
                )
        elif pl_module.config.y_val in BINARY_Y_VALS_MAP:
            train_preds_sigmoid = torch.sigmoid(train_preds)
            predicted_classes = (train_preds_sigmoid > 0.5).int()
            train_accuracy = (predicted_classes == train_targets).float().mean().item()
            # Convert tensors to numpy arrays and ensure they are integers
            train_targets_np = train_targets.numpy().astype(int).flatten()
            predicted_classes_np = predicted_classes.numpy().astype(int).flatten()
            wandb.log(
                {
                    "train_preds_sigmoid": train_preds_sigmoid,
                    "train_accuracy": train_accuracy,
                    "conf_mat train": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=train_targets_np,
                        preds=predicted_classes_np,
                        class_names=['Control (nfl), 0 (cdr)', 'Carrier (nfl), >0 (cdr)'],
                    ),  # TODO: hacky
                    "epoch": pl_module.current_epoch,
                }
            )
        elif pl_module.config.y_val in MULTICLASS_Y_VALS_MAP:
            softmax_probs = F.softmax(train_preds, dim=1)
            class_preds = torch.argmax(softmax_probs, dim=1)
            train_accuracy = (class_preds == train_targets).float().mean().item()
            class_preds_np = class_preds.numpy().astype(int).flatten()
            train_targets_np = train_targets.numpy().astype(int).flatten()
            wandb.log(
                {
                    "val_preds_softmax": wandb.Histogram(softmax_probs),
                    "val_preds_class": wandb.Histogram(class_preds),
                    "train_accuracy": train_accuracy,
                    "conf_matrix train": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=train_targets_np,
                        preds=class_preds_np,
                        class_names=['0', '0.5', '1', '2', '3'],  # TODO: Hardcoded
                    ),
                    "epoch": pl_module.current_epoch,
                }
            )
        pl_module.train_preds.clear()  # free memory
        pl_module.train_targets.clear()
        pl_module.x0.clear()
        pl_module.x1.clear()
        pl_module.x2.clear()

        gc.collect()  # Clean up Python's garbage
        torch.cuda.empty_cache()  # Clear any GPU memory cache

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save val predictions and targets as histograms and log confusion matrix.

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
            val_loss = pl_module.trainer.callback_metrics["val_loss"]

            # Log histograms and metrics
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_preds": wandb.Histogram(val_preds, num_bins=500),
                    "val_targets": wandb.Histogram(val_targets, num_bins=500),
                    "epoch": pl_module.current_epoch,
                }
            )
            if pl_module.config.y_val in CONTINOUS_Y_VALS:
                if val_loss < pl_module.min_val_loss:
                    pl_module.min_val_loss = val_loss
                    scatter_plot_data = [
                        [pred, target] for (pred, target) in zip(val_preds, val_targets)
                    ]
                    table = wandb.Table(data=scatter_plot_data, columns=["pred", "target"])
                    wandb.log(
                        {
                            "Regression Scatter Plot Val": wandb.plot.scatter(
                                table,
                                "pred",
                                "target",
                                title=f"Val Pred vs Val Target Scatter Plot",
                            ),
                            "epoch": pl_module.current_epoch,
                        }
                    )
            elif pl_module.config.y_val in BINARY_Y_VALS_MAP:
                val_preds_sigmoid = torch.sigmoid(val_preds)
                # Note this assumes binary classification
                predicted_classes = (val_preds_sigmoid > 0.5).int()
                val_accuracy = (predicted_classes == val_targets).float().mean().item()
                # Convert tensors to numpy arrays and ensure they are integers
                val_targets_np = val_targets.numpy().astype(int).flatten()
                predicted_classes_np = predicted_classes.numpy().astype(int).flatten()

                wandb.log(
                    {
                        "val_preds_sigmoid": wandb.Histogram(val_preds_sigmoid),
                        "val_accuracy": val_accuracy,
                        "conf_mat val": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=val_targets_np,
                            preds=predicted_classes_np,
                            class_names=[
                                'Control (nfl), 0 (cdr)',
                                'Carrier (nfl), >0 (cdr)',
                            ],  # TODO: hacky
                        ),
                        "epoch": pl_module.current_epoch,
                    }
                )
            elif pl_module.config.y_val in MULTICLASS_Y_VALS_MAP:
                softmax_probs = F.softmax(val_preds, dim=1)
                class_preds = torch.argmax(softmax_probs, dim=1)
                val_accuracy = (class_preds == val_targets).float().mean().item()
                class_preds_np = class_preds.numpy().astype(int).flatten()
                val_targets_np = val_targets.numpy().astype(int).flatten()
                wandb.log(
                    {
                        "val_preds_softmax": wandb.Histogram(softmax_probs),
                        "val_preds_class": wandb.Histogram(class_preds),
                        "val_accuracy": val_accuracy,
                        "conf_matrix val": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=val_targets_np,
                            preds=class_preds_np,
                            class_names=['0', '0.5', '1', '2', '3'],  # TODO: Hardcoded
                        ),
                        "epoch": pl_module.current_epoch,
                    }
                )

        pl_module.val_preds.clear()  # free memory
        pl_module.val_targets.clear()

        gc.collect()  # Clean up Python's garbage
        torch.cuda.empty_cache()  # Clear any GPU memory cache


class CustomRayReportLossCallback(Callback):
    """Callback that reports val loss to Ray."""

    # TODO: train_loss logged in ray_output_csv is one epoch later than the val_loss
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
