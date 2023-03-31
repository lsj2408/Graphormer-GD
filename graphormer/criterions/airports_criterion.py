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


@register_criterion("airports_criterion", dataclass=GraphPredictionConfig)
class GraphPredictionLoss(FairseqCriterion):
    """
    Implementation for the loss used in airports node-classification.
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
        logits = logits[:, 1:, :]
        targets = model.get_targets(sample, [logits])

        # Batch_size, Num_of_nodes, Classes
        B, N, _ = logits.shape
        loss_mask = torch.zeros((B, N, 1)).to(logits)
        loss_mask[range(B), sample['net_input']['batched_data']['idx'], :] = 1.0
        logits = logits[loss_mask.bool().repeat(1, 1, _)].contiguous().view(-1, _)
        targets = targets[loss_mask.bool().squeeze(-1)]

        loss = nn.CrossEntropyLoss(reduction='sum')(logits, targets)

        acc = (logits.argmax(dim=-1) == targets).sum()
        acc_total = targets.shape[0]

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "acc": int(acc.data),
            "acc_total": acc_total,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        acc_sum = sum(log.get("acc", 0) for log in logging_outputs)
        acc_total_sum = sum(log.get("acc_total", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=6
        )

        metrics.log_scalar(
            "acc", acc_sum / acc_total_sum, acc_total_sum, round=6
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
