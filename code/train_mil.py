"""
Train a MIL model from mil_model.py

Follows HuggingFace/PyTorch conventions
"""
import logging
from argparse import ArgumentParser
import os
import sys
from tqdm import tqdm
import random
import transformers
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from mil_model import MILModel, compute_metrics
from mil_dataset import MILTwitterDataset


# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


##################
# Main
##################
@dataclass
class MILTrainingArguments(TrainingArguments):
    """
    Extend the HuggingFace Trainer to accept experiment-specific
    arguments

    Example: https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/training_args_seq2seq.py#L28
    """
    dataset_dir: str = field(
        default=f"{os.environ['MINERVA_HOME']}/data/premade_mil",
    )
    instance_model: str = field(
        default="vinai/bertweet-base"
    )
    sample_instances: bool = field(
        default=False
    )
    finetune_instance_model: bool = field(
        default=True
    )
    num_tweets_per_day: int = field(
        default=10
    )
    key_instance_ratio: float = field(
        default=0.2
    )
    instance_level_loss: float = field(
        default=0.0
    )
    positive_class_weight: float = field(
        default=1.0
    )


def parse_args():
    parser = HfArgumentParser(MILTrainingArguments)
    return parser.parse_args_into_dataclasses()[0]  # Hack because returns tuple


def main():
    # Load commandline arguments
    train_args = parse_args()

    # Set rank in script instead of program arguments
    # Set variable to -1 if not using distributed training
    train_args.local_rank = int(os.environ.get('LOCAL_RANK', -1))

    # Fix --resume_from_checkpoint
    train_args.resume_from_checkpoint = train_args.resume_from_checkpoint == "True"

    # Set the main code and the modules it uses to the same log-level according to the node
    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Set CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not train_args.no_cuda else "cpu")

    # Set random seed
    torch.manual_seed(train_args.seed)
    random.seed(train_args.seed)

    # Set up model
    model = MILModel(
        instance_model_path=train_args.instance_model,
        key_instance_ratio=train_args.key_instance_ratio,
        finetune_instance_model=train_args.finetune_instance_model
    )
    tokenizer = AutoTokenizer.from_pretrained(train_args.instance_model)

    # Set up dataset
    train_dataset = MILTwitterDataset(
        f"{train_args.dataset_dir}/train.jsonl",
        tokenizer,
        samples_per_bag=train_args.num_tweets_per_day,
        sample_instances=train_args.sample_instances,
        random_seed=train_args.seed
    )
    eval_dataset = MILTwitterDataset(
        f"{train_args.dataset_dir}/val.jsonl",
        tokenizer,
        samples_per_bag=train_args.num_tweets_per_day,
        random_seed=train_args.seed
    )
    test_dataset = MILTwitterDataset(
        f"{train_args.dataset_dir}/test.jsonl",
        tokenizer,
        samples_per_bag=train_args.num_tweets_per_day,
        random_seed=train_args.seed
    )

    # loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_function)
    # for batch in loader:
    #     logger.info(batch.keys())
    #     import pdb;pdb.set_trace()

    # Training
    # Make sure there is a checkpoint if --resume_from_checkpoint
    if train_args.resume_from_checkpoint:
        if not os.path.exists(f"{train_args.output_dir}/trainer_state.json"):
            train_args.resume_from_checkpoint = False
            logger.warning(f"Checkpoint not found at {train_args.output_dir}. Not resuming from checkpoint.")
        else:
            logger.info(f"Resuming from checkpoint {train_args.output_dir}")
    trainer = Trainer(
        args=train_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if train_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=train_dataset.collate_function
    )
    ignore_keys = ["instance_probs", "key_instances"]
    if train_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(
            ignore_keys_for_eval=ignore_keys,
            resume_from_checkpoint=train_args.resume_from_checkpoint
        )
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # else:
    #     # Load model from file
    #     # This step is for when script is used in inference-only mode
    #     model.load_state_dict(torch.load(f"{train_args.output_dir}/pytorch_model.bin"))

    if train_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        metrics = trainer.evaluate(eval_dataset=test_dataset, ignore_keys=ignore_keys)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Perform inference. Does not return key instances, only returns bag prediction
    if train_args.do_predict:
        logger.warning(f"Prediction does not work. Skipping.")
    #     logger.info("*** Predict ***")
    #     prediction_output = trainer.predict(eval_dataset, metric_key_prefix="predict", ignore_keys=ignore_keys)
    #     prediction_output = trainer.predict(test_dataset, metric_key_prefix="test_predict", ignore_keys=ignore_keys)


if __name__ == "__main__":
    main()

