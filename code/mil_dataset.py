"""
PyTorch dataloader for Multi-Instance learning

Specific to tweets but can be modified for other purposes.

Author: Alexandra DeLucia
"""
import logging
import datetime as dt
import os
import sys
import re
import pathlib
import pandas as pd
import random
import jsonlines as jl
from iso3166 import countries as country_codes
from littlebird import TweetTokenizer, TweetReader
from typing import Optional, Tuple
import torch
import torchnet as tnt


# Setup logging
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class MILTwitterDataset(torch.utils.data.Dataset):
    """
    https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
    """
    def __init__(self,
                 filepath,
                 tokenizer,
                 samples_per_bag=10,
                 sample_instances=False,
                 random_seed=None
                 ):
        super().__init__()
        # Load data
        with jl.open(filepath) as reader:
            self.data = [i for i in reader.iter()]
        self.sample_instances = sample_instances
        self.samples_per_bag = samples_per_bag
        self.sampler = random.Random(random_seed)
        self.tokenizer = tokenizer
        self.pad = -100

    def __getitem__(self, index):
        # Retrieve bag at index
        # Of the format
        # {"bag_id":, "filename":, "label":, "instances":, "num_instances":,}
        item = self.data[index]
        bag = item
        # Limit number of instances to self.samples_per_bag
        # Only need to sample instances if there are more than self.samples_per_bag
        # otherwise, use the first X instances
        if bag["num_instances"] > self.samples_per_bag:
            if self.sample_instances:
                bag["instances"] = self.sampler.choices(bag["instances"], k=self.samples_per_bag)
            else:
                bag["instances"] = bag["instances"][:self.samples_per_bag]
        return bag

    def __len__(self):
        return len(self.data)

    def collate_function(self, batch):
        """
        Collate output from __get_item__ into a batch

        Add in an instance mask and the tokenizer-encoded inputs. Also pad so all tensors are same size.
        """
        collated = {
            "instance_scores": [],
            "bag_mask": [],
            "instance_ids": []
        }
        instance_text = []
        # max_instances = max([b['num_instances'] for b in batch])
        max_instances = self.samples_per_bag
        for i, temp in enumerate(batch):
            for key in temp.keys():
                if key == "instances":
                    continue
                val = temp[key]
                if key == "label":
                    key = "labels"  # Change to plural for clarity
                    val = torch.tensor(val, dtype=torch.float)
                if key in collated:
                    collated[key].append(val)
                else:
                    collated[key] = [val]

            # Handle padding
            mask = torch.zeros((1, max_instances))
            scores = torch.ones((1, max_instances)) * self.pad
            text = [self.tokenizer.pad_token] * max_instances
            ids = torch.ones((1, max_instances)) * self.pad

            for j, t in enumerate(temp["instances"]):
                scores[0][j] = t["civil_unrest_score"]
                text[j] = t["tweet_text"]
                mask[0][j] = 1
                ids[0][j] = torch.tensor(t["id_str"], dtype=torch.long)

            collated["instance_scores"].append(scores)
            collated["bag_mask"].append(mask)
            instance_text.extend(text)
            collated["instance_ids"].append(ids)

        # Stack tensors here
        collated["bag_mask"] = torch.stack(collated["bag_mask"], dim=0)
        collated["instance_scores"] = torch.stack(collated["instance_scores"], dim=0)
        collated["instance_ids"] = torch.stack(collated["instance_ids"], dim=0)
        collated["labels"] = torch.stack(collated["labels"], dim=0)
        # Pass tokenized text to model
        # DataParallel cannot split lists across GPUs, only tensors
        token_inputs = self.tokenizer.batch_encode_plus(
            instance_text,
            return_tensors="pt",
            truncation="longest_first",
            padding=True
        )
        collated["input_ids"] = token_inputs["input_ids"]
        collated["attention_mask"] = token_inputs["attention_mask"]
        return collated


class MILTwitterDatasetLazy(tnt.dataset.ListDataset):
    """
    'Lazy' version of MILTwitterDataset which loads examples from each file for
    each iteration.

    A lot of duplicate code from create_dataset.py
    """
    def __init__(self,
                 data_files,
                 positive_bag_ids,
                 tokenizer,
                 samples_per_file=10,
                 sample_instances=False,
                 random_seed=None
                 ):
        super().__init__(data_files, self._load_function)
        self.positive_bag_ids = positive_bag_ids
        self.sample_instances = sample_instances
        self.samples_per_file = samples_per_file
        self.random_seed = random_seed
        self.tweet_processor = TweetTokenizer(include_retweeted_and_quoted_content=True)  # Use to get tweet text
        self.tokenizer = tokenizer

    def _load_function(self, filename):
        """Load a single file. Can be pre-shuffled"""
        tweet_ids = []
        tweet_text = ["<pad>" for _ in range(self.samples_per_file)]
        tweet_scores = [-1 for _ in range(self.samples_per_file)]
        mask = [0 for _ in range(self.samples_per_file)]
        if self.sample_instances:
            # Use the following command to sample from files with reproducibility
            # Issue: samples the SAME tweets from a file for each iteration
            # f"zcat {filename} | ./seeded-shuf -n {self.samples_per_file} -s {self.random_seed}"
            command = f"zcat {filename} | shuf -n {self.samples_per_file}"
            sample = os.popen(command).read()
            reader = jl.Reader(sample.split("\n"))
            iterator = reader.iter(skip_invalid=True)
        else:
            reader = TweetReader(filename)
            iterator = reader.read_tweets()
        for i, tweet in enumerate(iterator):
            tweet_ids.append(tweet["id_str"])
            tweet_text[i] = self.tweet_processor.get_tweet_text(tweet)
            tweet_scores[i] = tweet.get("civil_unrest_score", -1)
            mask[i] = 1
            if i == (self.samples_per_file-1):  # 0-indexing
                break

        target = self._label_sample(filename)
        return {
            "bag_id": filename,  # Filenames are the bag ID
            "instance_ids": tweet_ids,
            "instance_text": tweet_text,
            "instance_scores": torch.tensor(tweet_scores),  # instance scores
            "mask": torch.tensor(mask),
            "labels": torch.tensor(target, dtype=torch.float)  # bag label
        }

    def _label_sample(self, filename):
        # Get filename and label sample
        filename = self._rename_file(filename)
        return int(filename in self.positive_bag_ids)

    def collate_function(self, batch):
        """Collate output from _load_function into a batch"""
        collated = {}
        for i, temp in enumerate(batch):
            # Skip files with no tweets
            if len(temp["instance_ids"]) == 0:
                continue
            for key in temp.keys():
                if key in collated:
                    collated[key].append(temp[key])
                else:
                    collated[key] = [temp[key]]
        # Stack tensors here
        collated["mask"] = torch.stack(collated["mask"], dim=0)
        collated["instance_scores"] = torch.stack(collated["instance_scores"], dim=0)
        collated["labels"] = torch.stack(collated["labels"], dim=0)

        # Pass tokenized text to model
        # DataParallel cannot split lists across GPUs, only tensors
        tweet_text = []
        for text in collated["instance_text"]:
            tweet_text.extend(text)
        token_inputs = self.tokenizer.batch_encode_plus(tweet_text, return_tensors="pt", truncation="longest_first", padding=True)
        collated["input_ids"] = token_inputs["input_ids"]
        collated["attention_mask"] = token_inputs["attention_mask"]
        return collated

    @staticmethod
    def _rename_file(filename):
        # Get filename and label sample
        filename = filename.split("/")[-1].split('.')[0]
        date = "_".join(filename.split("_")[:-1])
        country = country_codes.get(filename.split("_")[-1]).alpha3  # 2-digit -> UN code
        country_date = f"{date}_{country}"
        return country_date

    @classmethod
    def from_glob(cls, glob, *args, **kwargs):
        data_files = cls._files_from_glob(glob)
        return cls(data_files, *args, **kwargs)

    @staticmethod
    def _files_from_glob(glob):
        """
        List of files from a regular expression

        From StackOverflow
        https://stackoverflow.com/questions/13031989/regular-expression-usage-in-glob-glob-for-python
        """
        path = pathlib.Path(glob)
        path, pattern = path.parent, path.name
        pattern = re.compile(pattern)
        return [str(i) for i in path.iterdir() if pattern.search(str(i))]


def get_acled_labels(acled_file=f"{os.environ['MINERVA_HOME']}/data/2014-01-01-2020-01-01_acled_reduced_all.csv"):
    # Set up dataset
    # Ground truth labels from ACLED
    acled_df = pd.read_csv(
        acled_file,
        keep_default_na=False,  # Preserve "NA" country code
        parse_dates=[4],  # Event dates
    )
    positive_dates = set(acled_df.apply(
        lambda x: f"{x.event_date.strftime('%Y_%m_%d')}_{x.iso3}",
        axis=1))
    return positive_dates
