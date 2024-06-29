import torch
import wandb
from pytorch_lightning.callbacks import Callback, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


class RayHistogramCallback(Callback):
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
                "train_preds": wandb.Histogram(train_preds),
                "train_targets": wandb.Histogram(train_targets),
                "parameters": wandb.Histogram(params),
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
            #print(f"\n\npl_module.__dict__ = {pl_module.__dict__}")
            print(f"\n\ntrainer.connectors.logger.__dict__ = {trainer.connectors.logger.__dict__}")
            
            #print(f"\n\npl_module.logger.__dict__ = {pl_module.logger.__dict__}")
            pl_module.logger.experiment.log(
                {
                    "val_preds": wandb.Histogram(val_preds),
                    "val_targets": wandb.Histogram(val_targets),
                }
            )
        pl_module.val_preds.clear()  # free memory
        pl_module.val_targets.clear()


class HistogramCallback(Callback):
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
                "train_preds": wandb.Histogram(train_preds),
                "train_targets": wandb.Histogram(train_targets),
                "parameters": wandb.Histogram(params),
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
                    "val_preds": wandb.Histogram(val_preds),
                    "val_targets": wandb.Histogram(val_targets),
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
