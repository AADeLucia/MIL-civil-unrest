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
from mil_dataset import MILTwitterDatasetLazy, get_acled_labels


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
class ExperimentArguments:
    train_files: str = field(
        default=f"{os.environ['MINERVA_HOME']}/data/tweets_en/201[456]_.*.gz",
    )
    valid_files: str = field(
        default=f"{os.environ['MINERVA_HOME']}/data/tweets_en/2017_.*.gz"
    )
    test_files: str = field(
        default=f"{os.environ['MINERVA_HOME']}/data/tweets_en/201[89]_.*.gz"
    )
    acled_event_data: str = field(
        default=f"{os.environ['MINERVA_HOME']}/data/2014-01-01-2020-01-01_acled_reduced_all.csv"
    )
    instance_model: str = field(
        default="vinai/bertweet-base"
    )
    shuffle_samples: bool = field(
        default=False
    )
    finetune_instance_model: bool = field(
        default=True
    )
    num_tweets_per_day: int = field(
        default=1000
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
    parser = HfArgumentParser((TrainingArguments, ExperimentArguments))
    return parser.parse_args_into_dataclasses()


def main():
    # Load commandline arguments
    train_args, exp_args = parse_args()
    # Set rank in script instead of program arguments
    # Set variable to -1 if not using distributed training
    train_args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    logger.info(f"{train_args.local_rank=}")

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
        instance_model_path=exp_args.instance_model,
        key_instance_ratio=exp_args.key_instance_ratio,
        finetune_instance_model=exp_args.finetune_instance_model
    )
    tokenizer = AutoTokenizer.from_pretrained(exp_args.instance_model)

    # Set up dataset
    positive_bags = get_acled_labels(exp_args.acled_event_data)
    train_dataset = MILTwitterDatasetLazy.from_glob(exp_args.train_files, positive_bags, tokenizer,
                                                samples_per_file=exp_args.num_tweets_per_day,
                                                shuffle_samples=exp_args.shuffle_samples,
                                                random_seed=train_args.seed)
    eval_dataset = MILTwitterDatasetLazy.from_glob(exp_args.valid_files, positive_bags, tokenizer,
                                               samples_per_file=exp_args.num_tweets_per_day,
                                               shuffle_samples=False, random_seed=train_args.seed)
    test_dataset = MILTwitterDatasetLazy.from_glob(exp_args.test_files, positive_bags, tokenizer,
                                               samples_per_file=exp_args.num_tweets_per_day,
                                               shuffle_samples=False, random_seed=train_args.seed)

    # Training
    # Make sure there is a checkpoint if --resume_from_checkpoint
    if train_args.resume_from_checkpoint:
        if not os.path.exists(f"{train_args.output_dir}/trainer_state.json"):
            train_args.resume_from_checkpoint = False
            logger.warning(f"Checkpoint not found at {train_args.output_dir}. Not resuming from checkpoint.")
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
            resume_from_checkpoint=bool(train_args.resume_from_checkpoint)
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
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(eval_dataset, metric_key_prefix="predict", ignore_keys=ignore_keys)
        logger.info(f"{prediction_output=}")

        prediction_output = trainer.predict(test_dataset, metric_key_prefix="test_predict", ignore_keys=ignore_keys)
        logger.info(f"{prediction_output=}")


if __name__ == "__main__":
    main()

