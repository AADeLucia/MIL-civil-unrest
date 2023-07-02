"""
Base Multi-instance learning framework in PyTorch/HuggingFace

Author: Alexandra DeLucia
"""
import random

import transformers
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.file_utils import ModelOutput
from dataclasses import dataclass, field
import torch
import logging
import sys
import os

# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class InstanceModel(torch.nn.Module):
    """Instance -> score"""
    def __init__(self, model_path="vinai/bertweet-base"):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, normalization=True)

    def forward(self, input_ids, attention_mask, reshape=None):
        """Input is output of tokenizer. Output is probabilities"""
        # Flatten bags to pass through model
        # Store shape to re-shape after
        output = self.model(input_ids, attention_mask).logits
        # Reshape to [# bags, # instances]
        if reshape:
            output = output.view(reshape)
        output = torch.sigmoid(output)
        return output

    def calculate_loss(self, y_hat, y):
        """Unreduced MSE"""
        return torch.nn.functional.mse_loss(y_hat, y, reduction="none")

    def get_classifier_parameters(self):
        return self.model.classifier.parameters()

    def get_tokenizer(self):
        return self.tokenizer


class TopKBagModel(torch.nn.Module):
    """
    Instance scores -> score for bag

    This bag model uses the top instance scores to calculate the final
    bag prediction
    """
    def __init__(self,
                 key_instance_ratio=1.0,
                 positive_class_weight=1.0):
        super().__init__()
        self.register_buffer("key_instance_ratio", torch.tensor(key_instance_ratio))
        self.loss = torch.nn.BCELoss()

    def forward(self, X, mask):
        """
        X is shape [# bags (batch), # instances]
        """
        # Bag probability is the average of the top key_instances
        # Only consider the non-padded values
        num_key_instances = self.calc_num_key_instances(mask)
        num_key_instances = num_key_instances.unsqueeze(1)
        bag_probs = torch.empty(X.size(0), device=mask.device)
        key_instances = torch.ones((X.size(0), num_key_instances.max()), dtype=torch.long) * -100
        for i, (probs, k, m) in enumerate(zip(X, num_key_instances, mask)):
            top_instance_probs, top_instance_indices = torch.topk(probs * m, k.item())
            bag_probs[i] = self.aggregate_function(top_instance_probs)
            key_instances[i, :k.item()] = top_instance_indices
            logger.debug(f"{k.item()=}\n{top_instance_probs=}\n{top_instance_indices=}")
        return bag_probs.unsqueeze(-1), key_instances

    def calc_num_key_instances(self, mask):
        """
        Mask is of shape [batch (i.e., bags), 1, instances]
        """
        k = torch.max(
            torch.floor(self.key_instance_ratio * mask.sum(dim=-1)),
            torch.tensor([1], device=mask.device)
        ).type(dtype=torch.long)
        return k

    @staticmethod
    def calculate_loss(bag_probs, y):
        return torch.nn.functional.binary_cross_entropy(bag_probs, y)

    def aggregate_function(self, instance_probabilities):
        return torch.mean(instance_probabilities)


@dataclass
class MILClassifierOutput(ModelOutput):
    """
    Base class for outputs of multi-instance classification models.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    instance_probs: Optional[Tuple[torch.FloatTensor]] = None
    key_instances: Optional[Tuple[torch.FloatTensor]] = None


class MILModel(torch.nn.Module):
    def __init__(self,
                 instance_model_path="vinai/bertweet-base",
                 key_instance_ratio=1.0,
                 instance_level_loss=0.0,
                 finetune_instance_model=True
                 ):
        super().__init__()
        self.bag_model = TopKBagModel(key_instance_ratio)
        self.instance_model = InstanceModel(instance_model_path)
        self.register_buffer("instance_level_loss", torch.tensor(instance_level_loss))  # beta
        self.finetune_instance_model = finetune_instance_model

        if not self.finetune_instance_model:
            # Freeze instance model
            # Leave classification head alone
            for name, param in self.instance_model.model.roberta.named_parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, bag_mask, labels, instance_scores, instance_ids, **kwargs):
        """
        Bags are flattened
        Accepts **kwargs to handle HuggingFace trainer
        Only arguments specified in signature are split by the dataloader
        """
        # Get instance probs
        instance_probs = self.instance_model(input_ids, attention_mask, reshape=instance_scores.shape)
        # y and bag_probs = [#batch, 1]
        bag_probs, key_instance_idx = self.bag_model(instance_probs, bag_mask)
        # Map the key instance indices to the actual instance IDs for later analysis
        # Needs to be a tensor (i.e., all the same size) for the distributed batching
        key_instances = torch.ones_like(instance_ids) * -100
        for i, (b_idx, b_id) in enumerate(zip(key_instance_idx.squeeze(), instance_ids.squeeze())):
            # Remove padded values because -100 is treated as an index
            b_idx = np.delete(b_idx, np.where(b_idx==-100))
            key_instances[i, 0, :b_idx.shape[0]] = b_id[b_idx]
        output = {
            "loss": None,
            "logits": bag_probs,
            "instance_probs": instance_probs,
            "key_instances": key_instances
        }
        # Calculate loss
        if labels is not None:
            labels = labels.unsqueeze(-1)
            # if len(labels.shape) == 1:  # Fix for batch size of 1
            #     labels = labels.unsqueeze(0)
            output["loss"] = self.calculate_loss(bag_probs, instance_probs, bag_mask, labels, instance_scores)
        output = MILClassifierOutput(output)
        logger.debug(f"{output.loss=}")
        return output

    def get_parameters(self):
        if self.finetune_instance_model:
            # Return all parameters
            return self.parameters()
        else:
            # Only return classifier head
            return [p[1] for p in self.named_parameters() if "classifier" in p[0]]

    def calculate_loss(self, bag_probs, instance_probs, bag_mask, y=None, y_instance=None):
        # Empty loss in case labels are not passed
        loss = torch.tensor(0.0, device=bag_probs.device)
        # 1. Bag-level loss (Binary cross-entropy)
        if y is not None:
            loss = self.bag_model.calculate_loss(bag_probs, y)
        # 2. Instance-level loss (Binary cross-entropy)
        # Use instance_scores as ground truth. Want to minimize gap.
        # mean squared error = (error per instance) / (# of instances)
        # Scale instance loss by amount of effect we want
        if y_instance is not None:
            instance_loss = self.instance_model.calculate_loss(instance_probs, y_instance)
            num_key_instances = self.bag_model.calc_num_key_instances(bag_mask)
            instance_loss = (instance_loss * bag_mask).sum() / num_key_instances.sum()
            loss = loss + (instance_loss * self.instance_level_loss)
        return loss

    @classmethod
    def from_pretrained(cls, model_path):
        """Load a pretrained model. Function models HuggingFace from_pretrained"""
        # Load training arguments. Sloppy but works.
        # Eventually should try to do a nice HF model with a config
        settings = torch.load(
            f"{model_path}/training_args.bin"
        )
        m = MILModel(
            instance_model_path=settings.instance_model,
            key_instance_ratio=settings.key_instance_ratio,
            finetune_instance_model=settings.finetune_instance_model,
            instance_level_loss=settings.instance_level_loss
        )
        m.load_state_dict(
            torch.load(f"{model_path}/pytorch_model.bin")
        )
        return m


def compute_metrics(eval_prediction):
    # Model returns probabilities instead of logits
    # Also, uses -100 as a padding value. Do not keep the padding.
    # Only need the bag probabilities, not the instance or key instances
    # eval_prediction.predictions is a tuple when no ignore_keys are set
    if isinstance(eval_prediction.predictions, tuple):
        probs = eval_prediction.predictions[0]
    else:
        probs = eval_prediction.predictions
    probs = probs.reshape(-1)
    probs = np.delete(probs, np.where(probs == -100))
    predictions = (probs > 0.5).astype(np.uint8)
    label_ids = eval_prediction.label_ids
    try:
        prec, recall, f1, support = precision_recall_fscore_support(label_ids, predictions, zero_division=0, average="weighted")
    except ValueError as err:
        logger.error(f"{err=}\n{eval_prediction.predictions.shape=}\n{eval_prediction.label_ids.shape=}")
        exit(1)
    return {
        "precision": prec,
        "recall": recall,
        "f1": f1
    }
