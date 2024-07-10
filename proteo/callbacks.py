import math

import torch
import wandb
from pytorch_lightning.callbacks import Callback, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


class CustomWandbCallback(Callback):
    """Custom callback for logging to Wandb.

    The histograms are logged to wandb, but do not appear on the workspace view.
    They only appear on each run's view.
    """

    # FIXME: if loss is not the MSE (because loss has regularization, or loss=L1),
    # then sqrt(loss) is not the RMSE
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
        pl_module.log('train_RMSE', math.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)

    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                gradients = param.grad.detach().cpu()
                pl_module.logger.experiment.log({"gradients": wandb.Histogram(gradients)})

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
            pl_module.log('val_RMSE', math.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)

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
                "train_preds_logits": wandb.Histogram(train_preds),
                "train_targets": wandb.Histogram(train_targets),
                "parameters": wandb.Histogram(params),
                "epoch": pl_module.current_epoch,
            }
        )
        if pl_module.config.task_type == "classification":
            pl_module.logger.experiment.log(
                {"train_preds_sigmoid": wandb.Histogram(torch.sigmoid(train_preds))}
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
                    "val_preds_logits": wandb.Histogram(val_preds),
                    "val_preds_sigmoid": wandb.Histogram(torch.sigmoid(val_preds)),
                    "val_targets": wandb.Histogram(val_targets),
                    "epoch": pl_module.current_epoch,
                }
            )
            if pl_module.config.task_type == "classification":
                pl_module.logger.experiment.log(
                    {
                        "val_preds_sigmoid": wandb.Histogram(torch.sigmoid(val_preds)),
                        "val_accuracy": get_val_accuracy(val_preds, val_targets),
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


def get_val_accuracy(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total
