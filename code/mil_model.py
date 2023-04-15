"""
Base Multi-instance learning framework in PyTorch/HuggingFace

Author: Alexandra DeLucia
"""
import random
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.file_utils import ModelOutput
from dataclasses import dataclass, field
import torch
import logging

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


class BagModel(torch.nn.Module):
    """Instance scores -> score for bag"""
    def __init__(self,
                 key_instance_ratio=1.0,
                 positive_class_weight=1.0):
        super().__init__()
        self.key_instance_ratio = key_instance_ratio
        self.positive_class_weight = positive_class_weight  # NOT USED
        self.loss = torch.nn.BCELoss()

    def forward(self, X, mask):
        """
        X is shape [# bags (batch), # instances]
        """
        # Bag probability is the average of the top key_instances
        # Only consider the non-padded values
        num_key_instances = self.calc_num_key_instances(mask)
        bag_probs = torch.empty(X.size(0), device=mask.device)
        key_instances = []
        for i, (probs, k, m) in enumerate(zip(X, num_key_instances, mask)):
            top_instance_probs, top_instance_indices = torch.topk(probs * m, k.item())
            bag_probs[i] = torch.mean(top_instance_probs)
            key_instances.append(top_instance_indices)
            logger.debug(f"{k.item()=}\n{top_instance_probs=}\n{top_instance_indices=}")
        return bag_probs.unsqueeze(-1), key_instances

    def calc_num_key_instances(self, mask):
        k = torch.max(
            torch.floor(self.key_instance_ratio * mask.sum(dim=1)),
            torch.tensor([1], device=mask.device)
        ).type(dtype=torch.long)
        return k

    def calculate_loss(self, bag_probs, y):
        return self.loss(bag_probs, y)


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
                 instance_model=InstanceModel(),
                 bag_model=BagModel(),
                 instance_level_loss=0.0,
                 finetune_instance_model=True
                 ):
        super().__init__()
        self.bag_model = bag_model
        self.instance_model = instance_model
        self.instance_level_loss = instance_level_loss
        self.finetune_instance_model = finetune_instance_model

        if not self.finetune_instance_model:
            # Freeze instance model
            # Leave classification head alone
            for name, param in self.instance_model.model.roberta.named_parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, mask, labels, instance_scores, **kwargs):
        """
        Bags are flattened
        Accepts **kwargs to handle HuggingFace trainer
        """
        # Get instance probs
        instance_probs = self.instance_model(input_ids, attention_mask, reshape=instance_scores.shape)
        # y and bag_probs = [#batch, 1]
        bag_probs, key_instance_idx = self.bag_model(instance_probs, mask)
        # Map the key instance indices to the actual instance IDs for later analysis
        key_instances = []
        for b_idx, b_id in zip(key_instance_idx, kwargs["instance_ids"]):
            key_instances.append([b_id[i] for i in b_idx])
        # Calculate loss
        loss = None
        if labels is not None:
            labels = labels.unsqueeze(-1)
            # if len(labels.shape) == 1:  # Fix for batch size of 1
            #     labels = labels.unsqueeze(0)
            loss = self.calculate_loss(bag_probs, instance_probs, mask, labels, instance_scores)
        logger.debug(f"{os.environ.get('LOCAL_RANK')=}\n{input_ids.shape=}\n{mask.shape=}\n{instance_probs.shape=}\n{bag_probs.shape=}")
        output = {}
        if loss:
            output["loss"] = loss
        output.update({
            "logits": bag_probs,
            "instance_probs": instance_probs,
            "key_instances": key_instances
        })
        output = MILClassifierOutput(output)
        logger.debug(f"{output=}")
        return output

    def get_parameters(self):
        if self.finetune_instance_model:
            # Return all parameters
            return self.parameters()
        else:
            # Only return classifier head
            return [p[1] for p in self.named_parameters() if "classifier" in p[0]]

    def calculate_loss(self, bag_probs, instance_probs, mask, y=None, y_instance=None):
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
            num_key_instances = self.bag_model.calc_num_key_instances(mask)
            instance_loss = (instance_loss * mask).sum() / num_key_instances.sum()
            loss = loss + (instance_loss * self.instance_level_loss)
        return loss


def compute_metrics(eval_prediction):
    logger.info(f"{eval_prediction=}")
    # Model returns probabilities instead of logits
    # Also, uses -100 as a padding value. Do not keep the padding.
    probs = eval_prediction.predictions.reshape(-1)
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

