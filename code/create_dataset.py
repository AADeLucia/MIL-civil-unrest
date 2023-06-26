"""
Given tweet files and data labels, create an MIL dataset that fits in memory
"""
import logging
from argparse import ArgumentParser
import re
import os
import pandas as pd
import pathlib
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from iso3166 import countries as country_codes
from functools import partial
import jsonlines
from littlebird import TweetReader, TweetTokenizer
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()

COUNTRY_RE = re.compile(r"[A-Z]{2}")
DATE_RE = re.compile(r"\d{4}_\d{2}_\d{2}")
SEED = 42
KEEP_METADATA = [
    "created_at",
    "id_str",
    "tweet_text",
    "civil_unrest_score",
    "favorite_count",
    "retweet_count"
]


########
# Helpers
########
class SpamTokenizer(TweetTokenizer):
    def __init__(self):
        super().__init__(replace_usernames_with="@USER")
        self.hashtag_sub = "HASHTAG"

    def tokenize(self, tweet):
        # Remove "RT"
        tweet = self.RT_RE.sub(" ", tweet)
        # Lowercase
        tweet = tweet.lower()
        # Replace hashtags
        tweet = self.HASHTAG_RE.sub(self.hashtag_sub, tweet)
        # Replace usernames
        tweet = self.HANDLE_RE.sub(self.handle_sub, tweet)
        # Replace URLs
        tweet = self.URL_RE.sub(self.url_sub, tweet)
        # Remove pesky ampersand
        tweet = re.sub("(&amp)", " ", tweet)
        tweet = self.LONE_DIGIT_RE.sub(" ", tweet)
        # Tokenize
        tokens = self.TOKEN_RE.findall(tweet)
        return tokens

    def get_token_and_hashtag_count(self, tweet):
        tokens = self.tokenize(tweet)
        num_hashtag_tokens = sum([1 for t in tokens if t==self.hashtag_sub])
        num_handle_tokens = sum([1 for t in tokens if t==self.handle_sub])
        num_content_tokens = len(tokens) - num_hashtag_tokens - num_handle_tokens
        return num_content_tokens, num_hashtag_tokens, num_handle_tokens


def get_acled_labels(acled_file):
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


def files_from_glob(glob):
    """
    List of files from a regular expression

    From StackOverflow
    https://stackoverflow.com/questions/13031989/regular-expression-usage-in-glob-glob-for-python
    """
    path = pathlib.Path(glob)
    path, pattern = path.parent, path.name
    pattern = re.compile(pattern)
    return [str(i) for i in path.iterdir() if pattern.search(str(i))]


def country_date_from_file(filename):
    # Get filename and label sample
    filename = filename.split("/")[-1].split('.')[0]
    file_date = DATE_RE.findall(filename)[0]
    file_country = COUNTRY_RE.findall(filename)[0]
    country = country_codes.get(file_country).alpha3  # 2-digit -> UN code
    country_date = f"{file_date}_{country}"
    return country_date


def is_spam(tweet_text):
    """
    Identifies a tweet as spam or advertisement if
    - has less than 3 non-URL/hashtag/user tokens
    - has more than 3 hashtags
    - has more than 3 user mentions
    """
    tokenizer = SpamTokenizer()
    num_content_tokens, num_hashtag_tokens, num_handle_tokens = tokenizer.get_token_and_hashtag_count(tweet_text)
    return (num_content_tokens < 3) or (num_hashtag_tokens > 3) or (num_handle_tokens > 3)


def create_bag_from_file(file, labels, num_tweets):
    # Get bag ID
    id = country_date_from_file(file)
    label = int(id in labels)

    # Read file
    df = pd.read_json(file, lines=True)
    text_col = "full_text" if "full_text" in df.columns else "text"
    df.rename(columns={text_col: "tweet_text"}, inplace=True)

    # Drop duplicates
    df.drop_duplicates(subset="id_str", inplace=True)
    df.drop_duplicates(subset="tweet_text", inplace=True)

    # Remove retweets and quote tweets
    for col in ["retweeted", "is_quote_tweet"]:
        if col not in df.columns:
            continue
        df[col] = df[col].map(lambda x: False if pd.isna(x) else bool(x))
        df = df[~df[col]]

    # Remove spam-like tweets
    df = df[~df.tweet_text.map(is_spam)]

    # Sample
    if num_tweets < len(df):
        df = df.sample(n=num_tweets, random_state=SEED)

    # Fill in scores
    if "civil_unrest_score" not in df.columns:
        df.insert(len(df.columns), "civil_unrest_score", -1)

    # Fix dates for JSON serialization
    df["created_at"] = df.created_at.map(str)

    instances = df[KEEP_METADATA].to_dict(orient="records")

    return {
        "bag_id": id,
        "filename": file,
        "num_instances": len(instances),
        "label": label,
        "instances": instances
    }


def create_dataset(files, labels, num_tweets, output_file, n_cpu):
    if n_cpu == -1:
        bags = []
        for input_file in tqdm(files, ncols=0, desc="File"):
            bag = create_bag_from_file(input_file, labels, num_tweets)
            bags.append(bag)
            if logger.getEffectiveLevel()==logging.DEBUG and len(bags) > 100:
                break
    else:
        # https://docs.python.org/3/library/functools.html#functools.partial
        # https://miguendes.me/how-to-pass-multiple-arguments-to-a-map-function-in-python#using-partial-with-a-processpoolexecutor-or-threadpoolexecutor
        # https://towardsdatascience.com/parallelism-with-python-part-1-196f0458ca14
        partial_bag_fn = partial(create_bag_from_file, labels=labels, num_tweets=num_tweets)
        bags = process_map(
            partial_bag_fn,
            files,
            max_workers=n_cpu,
            ncols=0,
            desc="File",
            chunksize=(len(files)//n_cpu)+1,
            total=len(files)
        )

    # Save files
    with open(output_file, "w") as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(bags)


########
# Main
########
def main():
    args = parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Get the labels
    labels = set(get_acled_labels(args.acled_file))

    # Train set
    logger.info(f"On train set")
    train_files = files_from_glob(args.train_files)
    create_dataset(train_files, labels, args.max_instances, f"{args.output_dir}/train.jsonl", args.n_cpu)

    # Validation set
    logger.info(f"On val set")
    val_files = files_from_glob(args.val_files)
    create_dataset(val_files, labels, args.max_instances, f"{args.output_dir}/val.jsonl", args.n_cpu)

    # Test set
    logger.info(f"On test set")
    test_files = files_from_glob(args.test_files)
    create_dataset(test_files, labels, args.max_instances, f"{args.output_dir}/test.jsonl", args.n_cpu)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train-files", required=True, type=str, help="Tweet files for the training set")
    parser.add_argument("--val-files", required=True, type=str, help="Tweet files for the validation set")
    parser.add_argument("--test-files", required=True, type=str, help="Tweet files for the test set")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for file results")
    parser.add_argument("--max-instances", default=-1, type=int, help="Max instances (tweets) per bag (file). -1 means keep all.")
    parser.add_argument("--min-tokens", default=3, type=int, help="Minimum # of tokens to keep tweet. does not include URLs and handles.")
    parser.add_argument("--acled-file", type=str, default=f"{os.environ['MINERVA_HOME']}/data/2014-01-01-2020-01-01_acled_reduced_all.csv")
    parser.add_argument("--n-cpu", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
