"""
Train instance model for civil unrest classification on CUT dataset

Specific to tweets but can be modified for other purposes.
"""
import logging
import os
import re
import sys
import pandas as pd
import random
import transformers
from transformers import Trainer, TrainingArguments, HfArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass, field
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class CUTDataset(torch.utils.data.Dataset):
    """
    https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
    """
    def __init__(self, data):
        super().__init__()
        self.data = data.loc[:, ["text", "label"]].values.tolist()

    def __getitem__(self, index):
        # Retrieve item at index
        # Of the format tweet_id,text,label
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_function(batch, tokenizer):
    collated = {
        "labels": []
    }
    text = []
    for b in batch:
        text.append(b[0])
        collated["labels"].append(torch.tensor(b[1], dtype=torch.float))

    inputs = tokenizer.batch_encode_plus(
        text,
        return_tensors="pt",
        truncation="longest_first",
        padding=True
    )
    collated["labels"] = torch.stack(collated["labels"], dim=0).reshape(-1, 1)
    collated["input_ids"] = inputs["input_ids"]
    collated["attention_mask"] = inputs["attention_mask"]
    return collated


def prepare_data(filepath, seed, test_size, label):
    # Load full labels
    # Remove non-English
    data = pd.read_csv(filepath)
    data = data[data["is_event"]!="notenglish"]

    # Label according to desired annotation
    # See https://aclanthology.org/2020.wnut-1.28/ for details
    data["label"] = data[label].map(lambda x: 1 if x == "yes" else 0)
    logger.info(f"{data.label.sum()=}\t{len(data)}")

    # Split
    train_val, test = train_test_split(data, random_state=seed, test_size=test_size, stratify=data["label"])
    train, val = train_test_split(train_val, random_state=seed, test_size=0.1, stratify=train_val["label"])
    train = CUTDataset(train)
    val = CUTDataset(val)
    test = CUTDataset(test)
    return train, val, test


def compute_metrics(eval_prediction):
    probs = eval_prediction.predictions.reshape(-1)
    predictions = (probs > 0.5).astype(np.uint8)
    label_ids = eval_prediction.label_ids
    prec, recall, f1, support = metrics.precision_recall_fscore_support(label_ids, predictions, zero_division=0, average="weighted")
    acc = metrics.accuracy_score(label_ids, predictions)
    pos_f1 = metrics.f1_score(label_ids, predictions, average="binary", pos_label=1)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1,
        "positive_f1": pos_f1
    }


################
# Main
################
@dataclass
class CUTTrainingArguments(TrainingArguments):
    """
    Extend the HuggingFace Trainer to accept experiment-specific
    arguments

    Example: https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/training_args_seq2seq.py#L28
    """
    model_name_or_path: str = field(
        default="vinai/bertweet-base"
    )
    data_path: str = field(
        default=f"{os.environ['MINERVA_HOME']}/data/CUT_dataset.csv",
    )
    label: str = field(
        default="unrest"
    )
    n_trials: int = field(
        default=20
    )
    patience: int = field(
        default=-1
    )


def parse_args():
    parser = HfArgumentParser(CUTTrainingArguments)

    # Fix booleans
    train_args = parser.parse_args_into_dataclasses()[0]  # Hack because returns tuple
    train_args.resume_from_checkpoint = train_args.resume_from_checkpoint == "True"
    return train_args


def main():
    # Load commandline arguments
    train_args = parse_args()

    # Set rank in script instead of program arguments
    # Set variable to -1 if not using distributed training
    train_args.local_rank = int(os.environ.get('LOCAL_RANK', -1))

    # Set the main code and the modules it uses to the same log-level according to the node
    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Set CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not train_args.no_cuda else "cpu")

    # Set random seed
    torch.manual_seed(train_args.seed)
    random.seed(train_args.seed)

    # Load data
    train_dataset, eval_dataset, test_dataset = prepare_data(train_args.data_path, train_args.seed, 0.1, train_args.label)
    logger.info(f"{len(train_dataset)=}\t{len(eval_dataset)=}\t{len(test_dataset)=}")

    # Initialize model
    def model_init(trial):
        # https://huggingface.co/docs/transformers/hpo_train#how-to-enable-hyperparameter-search-in-example
        return AutoModelForSequenceClassification.from_pretrained(
            train_args.model_name_or_path,
            num_labels=1
        )

    def wandb_hp_space(trial):
        # https://huggingface.co/docs/transformers/hpo_train#how-to-enable-hyperparameter-search-in-example
        # https://hasty.ai/docs/mp-wiki/solvers-optimizers/weight-decay#:~:text=Typically%2C%20the%20parameter%20for%20weight,might%20not%20be%20powerful%20enough.
        # https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#parameters
        return {
            "method": "random",
            "metric": {
                "name": "objective",
                "goal": "minimize"
            },
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-2},
                "weight_decay": {"distribution": "categorical", "values": [1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0]},
            },
        }

    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name_or_path)
    if train_args.patience != -1:
        # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/callback#transformers.EarlyStoppingCallback
        callbacks = [
            transformers.EarlyStoppingCallback(early_stopping_patience=train_args.patience)
        ]
    else:
        callbacks = None

    # Hyperparameter search
    # Does not play nice with 'compute metrics', so just basing best model
    # off of eval_loss is fine.
    hyp_trainer = Trainer(
        args=train_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset,
        model_init=model_init,
        data_collator=lambda x: collate_function(x, tokenizer),
        callbacks=callbacks
    )
    best_trial = hyp_trainer.hyperparameter_search(
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=train_args.n_trials
    )
    logger.info(f"{best_trial=}")
    if best_trial.hyperparameters is None:
        logger.error(f"Hyperparameter sweep failed. Check logs.")
        exit(1)

    # HuggingFace trainer does not load the best model at end
    # and trainer.state only holds the LAST model, not the BEST model
    # Re-train with best settings
    for param, value in best_trial.hyperparameters.items():
        setattr(train_args, param, value)
    # Hack because of stupid bug
    # https://github.com/huggingface/transformers/issues/22429
    setattr(train_args, "report_to", ["wandb"])
    logger.info(f"{train_args=}")
    trainer = Trainer(
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=model_init,
        data_collator=lambda x: collate_function(x, tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    result = trainer.train()
    trainer.save_model()
    trainer.save_state()

    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    metrics = trainer.evaluate(eval_dataset=test_dataset)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()

