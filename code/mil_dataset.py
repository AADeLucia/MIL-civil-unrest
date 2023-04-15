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


class MILTwitterDataset(tnt.dataset.ListDataset):
    def __init__(self,
                 data_files,
                 positive_files,
                 tokenizer,
                 samples_per_file=10,
                 shuffle_samples=False,
                 random_seed=None
                 ):
        super().__init__(data_files, self._load_function)
        self.positive_files = positive_files
        self.shuffle_samples = shuffle_samples
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
        if self.shuffle_samples:
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
        return int(filename in self.positive_files)

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
        token_inputs = self.tokenizer.batch_encode_plus(tweet_text, return_tensors="pt",
                                                   truncation="longest_first", padding=True)
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

