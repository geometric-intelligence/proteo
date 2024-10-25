import io
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import Callback, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from sklearn.metrics import confusion_matrix

from proteo.datasets.ftd import BINARY_Y_VALS_MAP, CONTINOUS_Y_VALS, MULTICLASS_Y_VALS_MAP


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
                pl_module.logger.experiment.log({"gradients": wandb.Histogram(gradients.numpy())})

    # def on_validation_batch_end(self, trainer, pl_module, outputs, *args):
    #     if not trainer.sanity_checking:
    #         loss = outputs
    #         pl_module.log(
    #             'val_loss',
    #             loss,
    #             on_step=False,
    #             on_epoch=True,
    #             sync_dist=True,
    #             prog_bar=True,
    #             batch_size=1,
    #         )
    #         pl_module.log('val_RMSE', math.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)
            

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
        if pl_module.config.y_val == "clinical_dementia_rating_global":
            pl_module.train_targets = reshape_targets(pl_module.train_targets)
        train_targets = torch.vstack(pl_module.train_targets).detach().cpu()
        params = torch.concat([p.flatten() for p in pl_module.parameters()]).detach().cpu()
        if pl_module.config.model == "gat-v4":
            # Log the first graph ([0, :]) of x0, x1, and x2 to see if oversmoothing is happening, aka if all features across 1 person are the same
            x0 = torch.vstack(pl_module.x0).detach().cpu()[0, :]
            x1 = torch.vstack(pl_module.x1).detach().cpu()[0, :]
            x2 = torch.vstack(pl_module.x2).detach().cpu()[0, :]
            multiscale = torch.vstack(pl_module.multiscale).detach().cpu()
            multiscale = (
                torch.norm(multiscale, dim=1).detach().cpu().numpy()
            )  # Average the features across the 3 layers per person to get one value per person
            pl_module.logger.experiment.log(
            {
                "x0 oversmoothing 1 person": wandb.Histogram(x0.numpy()),
                "x1 oversmoothing 1 person": wandb.Histogram(x1.numpy()),
                "x2 oversmoothing 1 person": wandb.Histogram(x2.numpy()),
                "multiscale norm for all people": wandb.Histogram(multiscale),
                "epoch": pl_module.current_epoch,
            }
            )
            pl_module.x0.clear()
            pl_module.x1.clear()
            pl_module.x2.clear()
            pl_module.multiscale.clear()
        pl_module.logger.experiment.log(
            {
                "train_preds": wandb.Histogram(train_preds.numpy()),
                "train_targets": wandb.Histogram(train_targets.numpy()),
                "parameters (weights + biases)": wandb.Histogram(params.numpy()),
                "epoch": pl_module.current_epoch,
            }
        )
        if pl_module.config.y_val in BINARY_Y_VALS_MAP:
            train_preds_binary = (torch.sigmoid(train_preds) > 0.5).int()
            # Convert tensors to numpy arrays and ensure they are integers
            train_targets_np = train_targets.numpy().astype(int).flatten()
            train_preds_binary_np = train_preds_binary.numpy().astype(int).flatten()
            conf_matrix = confusion_matrix(train_targets_np, train_preds_binary_np)
            conf_matrix_df = pd.DataFrame(
                conf_matrix,
                index=[f'True_{i}' for i in range(conf_matrix.shape[0])],
                columns=[f'Pred_{i}' for i in range(conf_matrix.shape[1])],
            )
            pl_module.logger.experiment.log(
                {
                    "train_preds_sigmoid": wandb.Histogram(torch.sigmoid(train_preds).numpy()),
                    "train_accuracy": get_accuracy(train_preds_binary, train_targets),
                    "confusion_matrix train": wandb.Table(dataframe=conf_matrix_df),
                    "epoch": pl_module.current_epoch,
                }
            )
        elif pl_module.config.y_val in MULTICLASS_Y_VALS_MAP:
            softmax_probs = F.softmax(train_preds, dim=1)
            class_preds = torch.argmax(softmax_probs, dim=1)
            conf_matrix = confusion_matrix(train_targets, class_preds)
            conf_matrix_df = pd.DataFrame(
                conf_matrix,
                index=[f'True_{i}' for i in range(conf_matrix.shape[0])],
                columns=[f'Pred_{i}' for i in range(conf_matrix.shape[1])],
            )
            pl_module.logger.experiment.log(
                {
                    "train_preds_softmax": wandb.Histogram(softmax_probs.numpy()),
                    "train_preds_class": wandb.Histogram(class_preds.numpy()),
                    "train_accuracy": get_accuracy(class_preds, train_targets),
                    "confusion_matrix train": wandb.Table(dataframe=conf_matrix_df),
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
            loss = torch.vstack(pl_module.val_losses).detach().cpu().mean()
            pl_module.log(
                'val_loss',
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            pl_module.log('val_RMSE', math.sqrt(loss), on_step=False, on_epoch=True)
            
            val_preds = torch.vstack(pl_module.val_preds).detach().cpu()
            if pl_module.config.y_val == "clinical_dementia_rating_global":
                pl_module.val_targets = reshape_targets(pl_module.val_targets)
            val_targets = torch.vstack(pl_module.val_targets).detach().cpu()

            pl_module.logger.experiment.log(
                {
                    "val_preds": wandb.Histogram(val_preds.numpy()),
                    "val_targets": wandb.Histogram(val_targets.numpy()),
                    "epoch": pl_module.current_epoch,
                }
            )
            if pl_module.config.y_val in BINARY_Y_VALS_MAP:
                val_preds_binary = (torch.sigmoid(val_preds) > 0.5).int()
                # Convert tensors to numpy arrays and ensure they are integers
                val_targets_np = val_targets.numpy().astype(int).flatten()
                val_preds_binary_np = val_preds_binary.numpy().astype(int).flatten()
                conf_matrix = confusion_matrix(val_targets_np, val_preds_binary_np)
                conf_matrix_df = pd.DataFrame(
                    conf_matrix,
                    index=[f'True_{i}' for i in range(conf_matrix.shape[0])],
                    columns=[f'Pred_{i}' for i in range(conf_matrix.shape[1])],
                )
                # Log the confusion matrix plot
                pl_module.logger.experiment.log(
                    {
                        "val_preds_sigmoid": wandb.Histogram(torch.sigmoid(val_preds).numpy()),
                        "val_accuracy": get_accuracy(val_preds_binary, val_targets),
                        "epoch": pl_module.current_epoch,
                        "confusion_matrix val": wandb.Table(dataframe=conf_matrix_df),
                    }
                )
            elif pl_module.config.y_val in MULTICLASS_Y_VALS_MAP:
                softmax_probs = F.softmax(val_preds, dim=1)
                class_preds = torch.argmax(softmax_probs, dim=1)
                conf_matrix = confusion_matrix(val_targets, class_preds)
                conf_matrix_df = pd.DataFrame(
                    conf_matrix,
                    index=[f'True_{i}' for i in range(conf_matrix.shape[0])],
                    columns=[f'Pred_{i}' for i in range(conf_matrix.shape[1])],
                )
                pl_module.logger.experiment.log(
                    {
                        "val_preds_softmax": wandb.Histogram(softmax_probs.numpy()),
                        "val_preds_class": wandb.Histogram(class_preds.numpy()),
                        "val_accuracy": get_accuracy(class_preds, val_targets),
                        "confusion_matrix val": wandb.Table(dataframe=conf_matrix_df),
                        "epoch": pl_module.current_epoch,
                    }
                )
        pl_module.val_preds.clear()  # free memory
        pl_module.val_targets.clear()
        pl_module.val_losses.clear()


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


def get_accuracy(preds, targets):
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total


def reshape_targets(val_targets):
    '''When using multiclass classification, the targets need to be reshaped'''
    reshaped_targets = [tensor.view(-1, 1) for tensor in val_targets]
    return reshaped_targets


# TODO: Not currently being used
def reverse_log_transform(y, y_mean, y_std):
    log_data = (y * y_std) + y_mean
    original_data = torch.exp(log_data)
    return original_data
