# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn as nn
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import os


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("graph_prediction_bridge", dataclass=GraphPredictionConfig)
class GraphPredictionLossBridge(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        logits = model(**sample["net_input"])[0]
        # logits = logits[:,0,:]
        # targets = model.get_targets(sample, [logits])
        # loss = nn.L1Loss(reduction='sum')(logits, targets)

        targets = sample['net_input']['batched_data']['y']
        targets = targets - 1.0
        targets = targets.masked_fill((targets == -2.0).bool(), 1.0).long()
        targets = targets.contiguous().view(-1)
        logits = logits.contiguous().view(-1, logits.shape[-1])
        loss = modules.cross_entropy(
            logits,
            targets,
            ignore_index=-1,
            reduction="sum"
        )

        # acc:
        y_pred = logits.argmax(dim=-1)
        all_acc = (targets == y_pred).float()
        all_acc = all_acc.masked_fill((targets == -1).bool(), 0.0)
        acc_0 = all_acc[targets == 1.0].sum()
        acc_0_sample = (targets == 1.0).sum()
        acc_1 = all_acc[targets == 0.0].sum()
        acc_1_sample = (targets == 0.0).sum()
        all_acc = all_acc.sum()

        # graph_level acc
        bsz, n_node = sample['net_input']['batched_data']['y'].shape
        y_pred = y_pred.contiguous().view(bsz, n_node)
        targets = targets.contiguous().view(bsz, n_node)
        acc_0_mask = ~(targets == 1.0).bool()
        acc_1_mask = ~(targets == 0.0).bool()
        acc_pad_mask = (targets == -1.0).bool()

        all_acc_graph = ((y_pred == targets) + acc_pad_mask).all(dim=-1)
        acc_0_graph = ((y_pred == targets) + acc_0_mask).all(dim=-1)
        acc_1_graph = ((y_pred == targets) + acc_1_mask).all(dim=-1)

        sample_compute = torch.ones_like(acc_0_graph).float()
        acc_0_graph_sample = sample_compute.masked_fill(acc_0_mask.all(dim=-1), 0.0)
        acc_0_graph[acc_0_mask.all(dim=-1)] = 0.0
        acc_1_graph_sample = sample_compute.masked_fill(acc_1_mask.all(dim=-1), 0.0)
        acc_1_graph[acc_1_mask.all(dim=-1)] = 0.0

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "effective_num_nodes": (targets != -1.0).sum().item(),
            "nsentences": sample_size,
            "ntokens": natoms,
            "all_acc": all_acc,
            "acc_0": acc_0,
            "acc_1": acc_1,
            "acc_0_sample": acc_0_sample,
            "acc_1_sample": acc_1_sample,

            "all_acc_graph": all_acc_graph.sum().item(),
            "all_acc_sample": sample_compute.sum().item(),
            "acc_0_graph": acc_0_graph.sum().item(),
            "acc_1_graph": acc_1_graph.sum().item(),
            "acc_0_graph_sample": acc_0_graph_sample.sum().item(),
            "acc_1_graph_sample": acc_1_graph_sample.sum().item(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=6
        )

        effective_num_nodes = sum(log.get("effective_num_nodes", 0) for log in logging_outputs)

        all_acc = sum(log.get("all_acc", 0) for log in logging_outputs)
        acc_0 = sum(log.get("acc_0", 0) for log in logging_outputs)
        acc_1 = sum(log.get("acc_1", 0) for log in logging_outputs)
        acc_0_sample = sum(log.get("acc_0_sample", 0) for log in logging_outputs)
        acc_1_sample = sum(log.get("acc_1_sample", 0) for log in logging_outputs)
        all_acc_graph = sum(log.get("all_acc_graph", 0) for log in logging_outputs)
        all_acc_graph_sample = sum(log.get("all_acc_sample", 0) for log in logging_outputs)
        acc_0_graph = sum(log.get("acc_0_graph", 0) for log in logging_outputs)
        acc_1_graph = sum(log.get("acc_1_graph", 0) for log in logging_outputs)
        acc_0_graph_sample = sum(log.get("acc_0_graph_sample", 0) for log in logging_outputs)
        acc_1_graph_sample = sum(log.get("acc_1_graph_sample", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / effective_num_nodes, effective_num_nodes, round=3
        )

        metrics.log_scalar(
            "all_acc", all_acc / effective_num_nodes, effective_num_nodes, round=4
        )
        metrics.log_scalar(
            "all_acc_graph", all_acc_graph / all_acc_graph_sample if all_acc_graph_sample > 0 else 0, all_acc_graph_sample, round=4
        )
        metrics.log_scalar(
            "acc_0", acc_0 / acc_0_sample, acc_0_sample, round=4
        )
        metrics.log_scalar(
            "acc_1", acc_1 / acc_1_sample, acc_1_sample, round=4
        )
        metrics.log_scalar(
            "acc_0_graph", acc_0_graph / acc_0_graph_sample if acc_0_graph_sample > 0 else 0, acc_0_graph_sample, round=4
        )
        metrics.log_scalar(
            "acc_1_graph", acc_1_graph / acc_1_graph_sample if acc_1_graph_sample > 0 else 0, acc_1_graph_sample, round=4
        )
        metrics.log_scalar(
            "acc_0_graph_sample", acc_0_graph_sample, acc_0_graph_sample, round=4
        )
        metrics.log_scalar(
            "acc_1_graph_sample", acc_1_graph_sample, acc_1_graph_sample, round=4
        )

        metrics.log_scalar(
            "acc_0_sample", acc_0_sample, acc_0_sample, round=4
        )
        metrics.log_scalar(
            "acc_1_sample", acc_1_sample, acc_1_sample, round=4
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("graph_prediction_ap", dataclass=GraphPredictionConfig)
class GraphPredictionLossAP(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        logits = model(**sample["net_input"])[0][:, 1:, :]
        # logits = logits[:,0,:]
        # targets = model.get_targets(sample, [logits])
        targets = sample['net_input']['batched_data']['y']

        # logits = logits.masked_fill((targets == 0).bool(), 0.0)

        # loss = nn.L1Loss(reduction='sum')(logits, targets)
        targets = targets - 1.0
        targets = targets.masked_fill((targets == -2.0).bool(), 1.0).long()
        targets = targets.contiguous().view(-1)
        logits = logits.contiguous().view(-1, logits.shape[-1])
        loss = modules.cross_entropy(
            logits,
            targets,
            ignore_index=-1,
            reduction='sum')

        # acc:
        y_pred = logits.argmax(dim=-1)
        all_acc = (targets == y_pred).float()
        all_acc = all_acc.masked_fill((targets == -1).bool(), 0.0)
        acc_0 = all_acc[targets == 1.0].sum()
        acc_0_sample = (targets == 1.0).sum()
        acc_1 = all_acc[targets == 0.0].sum()
        acc_1_sample = (targets == 0.0).sum()
        all_acc = all_acc.sum()

        # graph_level acc
        bsz, n_node = sample['net_input']['batched_data']['y'].shape
        y_pred = y_pred.contiguous().view(bsz, n_node)
        targets = targets.contiguous().view(bsz, n_node)
        acc_0_mask = ~(targets == 1.0).bool()
        acc_1_mask = ~(targets == 0.0).bool()
        acc_pad_mask = (targets == -1.0).bool()

        all_acc_graph = ((y_pred == targets) + acc_pad_mask).all(dim=-1)
        acc_0_graph = ((y_pred == targets) + acc_0_mask).all(dim=-1)
        acc_1_graph = ((y_pred == targets) + acc_1_mask).all(dim=-1)

        sample_compute = torch.ones_like(acc_0_graph).float()
        acc_0_graph_sample = sample_compute.masked_fill(acc_0_mask.all(dim=-1), 0.0)
        acc_0_graph[acc_0_mask.all(dim=-1)] = 0.0
        acc_1_graph_sample = sample_compute.masked_fill(acc_1_mask.all(dim=-1), 0.0)
        acc_1_graph[acc_1_mask.all(dim=-1)] = 0.0

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "effective_num_nodes": (targets != -1.0).sum().item(),
            "nsentences": sample_size,
            "ntokens": natoms,
            "all_acc": all_acc,
            "acc_0": acc_0,
            "acc_1": acc_1,
            "acc_0_sample": acc_0_sample,
            "acc_1_sample": acc_1_sample,
            "all_acc_graph": all_acc_graph.sum().item(),
            "all_acc_sample": sample_compute.sum().item(),
            "acc_0_graph": acc_0_graph.sum().item(),
            "acc_1_graph": acc_1_graph.sum().item(),
            "acc_0_graph_sample": acc_0_graph_sample.sum().item(),
            "acc_1_graph_sample": acc_1_graph_sample.sum().item(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=6
        )

        effective_num_nodes = sum(log.get("effective_num_nodes", 0) for log in logging_outputs)

        all_acc = sum(log.get("all_acc", 0) for log in logging_outputs)
        acc_0 = sum(log.get("acc_0", 0) for log in logging_outputs)
        acc_1 = sum(log.get("acc_1", 0) for log in logging_outputs)
        acc_0_sample = sum(log.get("acc_0_sample", 0) for log in logging_outputs)
        acc_1_sample = sum(log.get("acc_1_sample", 0) for log in logging_outputs)
        all_acc_graph = sum(log.get("all_acc_graph", 0) for log in logging_outputs)
        all_acc_graph_sample = sum(log.get("all_acc_sample", 0) for log in logging_outputs)
        acc_0_graph = sum(log.get("acc_0_graph", 0) for log in logging_outputs)
        acc_1_graph = sum(log.get("acc_1_graph", 0) for log in logging_outputs)
        acc_0_graph_sample = sum(log.get("acc_0_graph_sample", 0) for log in logging_outputs)
        acc_1_graph_sample = sum(log.get("acc_1_graph_sample", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / effective_num_nodes, effective_num_nodes, round=3
        )

        metrics.log_scalar(
            "all_acc", all_acc / effective_num_nodes, effective_num_nodes, round=4
        )
        metrics.log_scalar(
            "all_acc_graph", all_acc_graph / all_acc_graph_sample if all_acc_graph_sample > 0 else 0.0, all_acc_graph_sample, round=4
        )
        metrics.log_scalar(
            "acc_0", acc_0 / acc_0_sample, acc_0_sample, round=4
        )
        metrics.log_scalar(
            "acc_1", acc_1 / acc_1_sample, acc_1_sample, round=4
        )
        metrics.log_scalar(
            "acc_0_graph", acc_0_graph / acc_0_graph_sample if acc_0_graph_sample > 0 else 0.0, acc_0_graph_sample, round=4
        )
        metrics.log_scalar(
            "acc_1_graph", acc_1_graph / acc_1_graph_sample if acc_1_graph_sample > 0 else 0.0, acc_1_graph_sample, round=4
        )
        metrics.log_scalar(
            "acc_0_graph_sample", acc_0_graph_sample, acc_0_graph_sample, round=4
        )
        metrics.log_scalar(
            "acc_1_graph_sample", acc_1_graph_sample, acc_1_graph_sample, round=4
        )

        metrics.log_scalar(
            "acc_0_sample", acc_0_sample, acc_0_sample, round=4
        )
        metrics.log_scalar(
            "acc_1_sample", acc_1_sample, acc_1_sample, round=4
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True