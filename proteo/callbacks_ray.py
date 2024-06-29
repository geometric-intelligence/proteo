import math

import torch
import wandb
from pytorch_lightning.callbacks import Callback


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
            # FIXME: val/loss (on wandb) and val_loss (printed in the console) are not the same
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
                    "val/preds": wandb.Histogram(val_preds),
                    "val/targets": wandb.Histogram(val_targets),
                    "epoch": pl_module.current_epoch,
                }
            )
        pl_module.val_preds.clear()  # free memory
        pl_module.val_targets.clear()
