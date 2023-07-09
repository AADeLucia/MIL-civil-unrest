"""
Use baseline code from Study of Manifestation of Civil Unrest on Twitter (Chinta et al., WNUT 2021)
Paper: https://aclanthology.org/2021.wnut-1.44/
Code: https://github.com/JHU-CLSP/civil-unrest-case-study
"""
import sys
import os
import logging
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import json
from baseline_models import RandomThresholdClassifier, CountryThresholdClassifier, MIL_I
from argparse import ArgumentParser
from mil_dataset import MILTwitterDataset
from transformers import AutoTokenizer, AutoModel
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()


########
# Helper
########
def prepare_data(dataset_dir, instance_threshold):
    datasets = []
    for split in ["train", "val", "test"]:
        samples = []
        split_file = f"{dataset_dir}/{split}.jsonl"
        split_data = MILTwitterDataset(
            split_file,
            None,
            samples_per_bag=10**10,  # High number so instances are not sampled
        )
        for bag in tqdm(split_data, ncols=0, desc=split):
            # Need the following fields: ID, COUNTRY, TEXT
            text, scores = [], []
            for i in bag['instances']:
                # Check if tweet meets score threshold
                if i["instance_score"] < instance_threshold:
                    continue
                text.append(i['tweet_text'])
                scores.append(i["instance_score"])
            sample = {
                "ID": bag["bag_id"],
                "label": bag['label'],
                "COUNTRY": bag["bag_id"].split("_")[-1],
                "TEXT": text,
                "SCORES": scores,
                "NUM_INSTANCES": bag["num_instances"]
            }
            samples.append(sample)
        datasets.append(
            pd.DataFrame(samples)
        )
    return datasets


def eval_model(clf, val, test, output_file):
    pred_y_val = clf.predict(val)
    pred_y_test = clf.predict(test)
    f1_val = precision_recall_fscore_support(val["label"], pred_y_val)[:3]
    f1_test = precision_recall_fscore_support(test["label"], pred_y_test)[:3]
    results = {
        "eval": {
            "f1": f1_val[2][1],
            "precision": f1_val[0][1],
            "recall": f1_val[1][1],
            "predictions": pred_y_val.tolist(),
            "ground_truth": val["label"].values.tolist()
        },
        "test": {
            "f1": f1_test[2][1],
            "precision": f1_test[0][1],
            "recall": f1_test[1][1],
            "predictions": pred_y_test.tolist(),
            "ground_truth": test["label"].values.tolist()
        }
    }
    with open(output_file, "w") as f:
        json.dump(results, f)


def run_random_baselines(train, val, test, output_dir):
    for name, clf in [("random", RandomThresholdClassifier), ("country_random", CountryThresholdClassifier)]:
        logger.info(f"Running {name}")
        clf = clf()
        clf.fit(train, train["label"])
        eval_model(clf, val, test, f"{output_dir}/{name}_results.json")


tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
def tokenize(x):
    return " ".join(tokenizer.tokenize(x))


def run_ngram_baseline(train, val, test, output_dir):
    logger.info(f"Running Ngram model")
    train = train[["TEXT", "label"]]
    val = val[["TEXT", "label"]]
    test = test[["TEXT", "label"]]

    # Fit model
    pipe = Pipeline([
        # Convert DataFrame to list of text
        ("preprocess", FunctionTransformer(lambda df: df["TEXT"].map(" ".join))),
        # Generate token count features. Use same vocabulary as BERTweet
        ("vectorizer", CountVectorizer(preprocessor=tokenize, tokenizer=str.split)),
        # Model. Settings are from the models.py code
        ("clf", RandomForestClassifier(n_estimators=10, max_depth=32, min_samples_split=32, class_weight='balanced'))
    ])
    pipe.fit(train, train["label"])

    # Evaluate
    eval_model(pipe, val, test, f"{output_dir}/ngram_results.json")


def run_mil_i_baseline(train, val, test, output_dir):
    logger.info(f"Running MIL-I model")
    for k in np.arange(0.0, 1.1, step=0.1):
        clf = MIL_I(k)
        clf.fit(train, train['label'])
        eval_model(clf, val, test, f"{output_dir}/MIL-I-{k:.1}_results.json")


def get_bag_features(model_path, data):
    # Set CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_path).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    features = []
    with torch.inference_mode():
        for row in tqdm(data.itertuples(index=False), ncols=0, total=len(data)):
            batch = tokenizer(row.TEXT, return_tensors="pt", truncation="longest_first", padding=True).to(device)
            outputs = model(**batch)

            # Average for day
            # Use pooled output (CLS token for BERT)
            features.append(torch.mean(outputs.pooler_output, axis=0).cpu().numpy())
    data["features"] = features
    return data


def run_avg_bag_baseline(train, val, test, output_dir):
    logger.info(f"Running AVG bag model")
    logger.info(f"Getting features")
    model_path = f"{os.environ['MINERVA_HOME']}/models/minerva_instance_models"

    train = get_bag_features(model_path, train)
    val = get_bag_features(model_path, val)
    test = get_bag_features(model_path, test)

    # Model. Settings are from the models.py code
    pipe = Pipeline([
        ("transform", FunctionTransformer(lambda x: np.stack(x['features'].values))),
        ("clf", RandomForestClassifier(n_estimators=10, max_depth=32, min_samples_split=32, class_weight='balanced'))
    ])
    pipe.fit(train, train.label)
    eval_model(pipe, val, test, f"{output_dir}/avg-bag_results.json")


def run_avg_bag_bertweet_baseline(train, val, test, output_dir):
    logger.info(f"Running AVG Bag BERTweet model")
    logger.info(f"Getting features")
    model_path = f"vinai/bertweet-base"

    train = get_bag_features(model_path, train)
    val = get_bag_features(model_path, val)
    test = get_bag_features(model_path, test)

    # Model. Settings are from the models.py code
    pipe = Pipeline([
        ("transform", FunctionTransformer(lambda x: np.stack(x['features'].values))),
        ("clf", RandomForestClassifier(n_estimators=10, max_depth=32, min_samples_split=32, class_weight='balanced'))
    ])
    pipe.fit(train, train.label)
    eval_model(pipe, val, test, f"{output_dir}/avg-bag-bertweet_results.json")


########
# Main
########
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", help="Folder to save results", required=True)
    parser.add_argument("--instance-threshold", type=float, default=0)
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of processes for models")
    # parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Preparing data from {args.dataset_dir}")
    train, val, test = prepare_data(args.dataset_dir, args.instance_threshold)

    # Not optimized, takes ~7 hours
    run_avg_bag_bertweet_baseline(train, val, test, args.output_dir)
    # run_mil_avg_baseline(train, val, test, args.output_dir)

    # run_mil_i_baseline(train, val, test, args.output_dir)

    # Random baselines
    # run_random_baselines(train, val, test, args.output_dir)

    # Ngram baseline
    # run_ngram_baseline(train, val, test, args.output_dir)


if __name__ == "__main__":
    main()

