# MIL Civil Unrest
Repository for future submission _A Multi-instance Learning Approach to Civil Unrest Event Forecasting on Twitter_

This repo is for the social media portion (Twitter) of the Dredze Group Minerva project. This work is funded by a Minerva grant and is in collaboration with APL.


## Environment Setup
The following snippet downloads this repository and other dependencies. Assumes a Linux environment with `conda` installed.

```bash
git clone git@github.com:AADeLucia/MIL-civil-unrest.git
cd MIL-civil-unrest
conda env create -f environment.yml
conda activate minerva-proj
```

Add the following to your `~/.bashrc`
```bash
export MINERVA_HOME=/path/to/repo
```


## Getting Started
Only the scripts, code, and small data files are stored on GitHub.

* `code`:
  * `mil_dataset.py`: File containing PyTorch-based custom dataloader for the tweets
  * `mil_model.py`: Model class
* `data`:
  * `tweets_en`: Tweet files aggregated by date and country of origin. Only English tweets are included, as identified by the language metadata tag. Country of origin identified by tweet location metadata. Files are in compressed JSONlines form and can be read with [`jsonlines`](https://jsonlines.readthedocs.io/en/latest/) or [`littlebird`](https://github.com/AADeLucia/littlebird) packages.
  * `2014-01-01-2020-01-01_acled_reduced_all.csv`: Civil unrest labels for the Twitter data. Provided by [Armed Conflict Location & Event Data Project (ACLED)](https://acleddata.com/data-export-tool/)
* `results`: Models and prediction files
  
