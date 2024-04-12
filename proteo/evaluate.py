import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as Data
import proteo.datasets as datasets
from models.gat_v4 import define_reg
from sklearn.model_selection import StratifiedKFold

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_metrics(config, model, model_parameters, test_loader):
    model.eval()

    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    loss_test = 0
    for batch in test_loader:
        batch.to(device)
        censor = batch.y[:, 0]
        survtime = batch.y[:, 0]
 
        # HACK ALERT: Pred is survival only
        _, _, test_pred = model(batch, model_parameters)

        loss_cox = (
            datasets.CoxLoss(survtime, censor, test_pred)
            if config.task == "survival"
            else 0
        )
        loss_reg = define_reg(model)
        loss = (
            model_parameters.lambda_cox * loss_cox
            + model_parameters.lambda_reg * loss_reg
        )
        loss_test += loss.data.item()

        if config.task == "survival":
            risk_pred_all = np.concatenate(
                (risk_pred_all, test_pred.detach().cpu().numpy().reshape(-1))
            )
            censor_all = np.concatenate(
                (censor_all, censor.detach().cpu().numpy().reshape(-1))
            )
            survtime_all = np.concatenate(
                (survtime_all, survtime.detach().cpu().numpy().reshape(-1))
            )
        else:
            raise ValueError(f"Task not recognized: got {config.task}")

    loss_test /= len(test_loader.dataset)
    cindex_test = (
        datasets.CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        if config.task == 'survival'
        else None
    )
    pvalue_test = (
        datasets.cox_log_rank(risk_pred_all, censor_all, survtime_all)
        if config.task == 'survival'
        else None
    )
    surv_acc_test = (
        datasets.accuracy_cox(risk_pred_all, censor_all) if config.task == 'survival' else None
    )
    return (
        loss_test,
        cindex_test,
        pvalue_test,
        surv_acc_test,
    )
