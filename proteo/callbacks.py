import math

import torch
import wandb
from pytorch_lightning.callbacks import Callback, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


class RayCustomWandbLoggerCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, *args):
        loss = outputs["loss"]
        wandb.log({'train/loss': loss, "epoch": pl_module.current_epoch})
        # FIXME: if loss is not the MSE (regularization, or L1), then sqrt(loss) is not the RMSE
        wandb.log({'train/RMSE': math.sqrt(loss), "epoch": pl_module.current_epoch})

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args):
        if not trainer.sanity_checking:
            loss = outputs
            wandb.log({'val/loss': loss, "epoch": pl_module.current_epoch})
            # Log to lightning so that the hyperparameter search can access val_loss
            pl_module.log(
                'val_loss',
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=pl_module.config.batch_size,
            )

            # FIXME: if loss is not the MSE (regularization, or L1), then sqrt(loss) is not the RMSE
            wandb.log({"val/RMSE": math.sqrt(loss), "epoch": pl_module.current_epoch})

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
                "train/preds": wandb.Histogram(train_preds),
                "train/targets": wandb.Histogram(train_targets),
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
                    "val_preds": wandb.Histogram(val_preds),
                    "val_targets": wandb.Histogram(val_targets),
                    "epoch": pl_module.current_epoch,
                }
            )
        pl_module.val_preds.clear()  # free memory
        pl_module.val_targets.clear()


class CustomWandbLoggerCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, *args):
        loss = outputs["loss"]
        pl_module.log(
            'train/loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=pl_module.config.batch_size,
        )
        # FIXME: if loss is not the MSE (regularization, or L1), then sqrt(loss) is not the RMSE
        pl_module.log('train/RMSE', math.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args):
        if not trainer.sanity_checking:
            loss = outputs
            pl_module.log(
                'val/loss',
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
                batch_size=pl_module.config.batch_size,
            )
            # HACKALETER: Relogging as val_loss to accommodate the monitoring of the loss
            # by the hyperparameter search algorithm.
            pl_module.log(
                'val_loss',
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=pl_module.config.batch_size,
            )
            # FIXME: if loss is not the MSE (regularization, or L1), then sqrt(loss) is not the RMSE
            pl_module.log('val/RMSE', math.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)

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
        pl_module.logger.experiment.log(
            {
                "train/preds": wandb.Histogram(train_preds),
                "train/targets": wandb.Histogram(train_targets),
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
            pl_module.logger.experiment.log(
                {
                    "val/preds": wandb.Histogram(val_preds),
                    "val/targets": wandb.Histogram(val_targets),
                    "epoch": pl_module.current_epoch,
                }
            )
            pl_module.val_preds.clear()  # free memory
            pl_module.val_targets.clear()


def progress_bar():
    custom_theme = RichProgressBarTheme(
        **{
            "description": "deep_sky_blue1",
            "progress_bar": "orange1",
            "progress_bar_finished": "orange3",
            "progress_bar_pulse": "deep_sky_blue3",
            "batch_progress": "deep_sky_blue1",
            "time": "grey82",
            "processing_speed": "grey82",
            "metrics": "grey82",
            "metrics_text_delimiter": "\n",
            "metrics_format": ".3e",
        }
    )
    progress_bar = RichProgressBar(theme=custom_theme)
    return progress_bar
