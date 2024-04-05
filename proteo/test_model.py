import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as Data
import utils
from models.gat_v4 import define_reg
from sklearn.model_selection import StratifiedKFold


def test(config, model, test_features, test_labels, adj_matrix, model_parameters, test_loader):
    model.eval()

    test_dataset = Data.TensorDataset(test_features, test_labels)
    print("Dimensions of test dataset:", test_features.shape, test_labels.shape)
    test_loader_torch = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config.batch_size, shuffle=False
    )

    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0
    for (batch_idx, batch), (batch_features_torch, batch_labels_torch) in zip(enumerate(test_loader), enumerate(test_loader_torch)):
        print("Batch labels ", batch_labels_torch)
        censor = batch_labels_torch[:, 0]
        survtime = batch_labels_torch[:, 1]
        grade = batch_labels_torch[:, 2]
        censor_batch_labels = censor.cuda() if "surv" in config.task else censor
        surv_batch_labels = survtime
        # print(surv_batch_labels)
        grad_batch_labels = grade.cuda() if "grad" in config.task else grade
        # TO DO: Fix this and understand what line 37 is doing.
        test_features, te_fc_features, test_predictions, gradients, feature_importance = model(
            batch, model_parameters
        )

        # print("surv_batch_labels:", surv_batch_labels)
        # print("test_predictions:", test_predictions)

        if batch_idx == 0:
            features_all = test_features.detach().cpu().numpy()
            fc_features_all = te_fc_features.detach().cpu().numpy()
        else:
            features_all = np.concatenate(
                (features_all, test_features.detach().cpu().numpy()), axis=0
            )
            fc_features_all = np.concatenate(
                (fc_features_all, te_fc_features.detach().cpu().numpy()), axis=0
            )
        # print(features_all.shape, test_features.shape)

        loss_cox = (
            utils.CoxLoss(surv_batch_labels, test_predictions)
            if config.task == "surv"
            else 0
        )
        loss_reg = define_reg(model)
        loss_func = nn.CrossEntropyLoss()
        grad_loss = loss_func(test_predictions, grad_batch_labels) if config.task == "grad" else 0
        loss = (
            config.lambda_cox * loss_cox
            + config.lambda_nll * grad_loss
            + config.lambda_reg * loss_reg
        )
        loss_test += loss.data.item()

        gt_all = np.concatenate(
            (gt_all, grad_batch_labels.detach().cpu().numpy().reshape(-1))
        )  # Logging Information

        if config.task == "surv":
            risk_pred_all = np.concatenate(
                (risk_pred_all, test_predictions.detach().cpu().numpy().reshape(-1))
            )  # Logging Information
            censor_all = np.concatenate(
                (censor_all, censor_batch_labels.detach().cpu().numpy().reshape(-1))
            )  # Logging Information
            survtime_all = np.concatenate(
                (survtime_all, surv_batch_labels.detach().cpu().numpy().reshape(-1))
            )  # Logging Information

        elif config.task == "grad":
            pred = test_predictions.argmax(dim=1, keepdim=True)
            grad_acc_test += pred.eq(grad_batch_labels.view_as(pred)).sum().item()
            probs_np = test_predictions.detach().cpu().numpy()
            probs_all = (
                probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)
            )  # Logging Information

    # print(survtime_all)
    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)
    cindex_test = (
        utils.CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        if config.task == 'surv'
        else None
    )
    pvalue_test = (
        utils.cox_log_rank(risk_pred_all, censor_all, survtime_all)
        if config.task == 'surv'
        else None
    )
    surv_acc_test = utils.accuracy_cox(risk_pred_all, censor_all) if config.task == 'surv' else None
    grad_acc_test = grad_acc_test / len(test_loader.dataset) if config.task == 'grad' else None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

    return (
        loss_test,
        cindex_test,
        pvalue_test,
        surv_acc_test,
        grad_acc_test,
        pred_test,
        features_all,
        fc_features_all,
    )
