import os
from os.path import dirname

import pandas as pd


PROJECT_DIR = dirname(dirname(dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(PROJECT_DIR, "res", "datasets")


def read_twenty_news():
    directory = os.path.join(DATASET_DIR, "20news", "20news-bydate-train")
    data = []
    categories = os.listdir(directory)
    for category in categories:
        subdir = os.path.join(directory, category)
        for filename in os.listdir(subdir):
            data.append(dict(fname=filename, category=category))
    df = pd.DataFrame(data)

    def f(r):
        return os.path.join(directory, r['category'], r['fname'])
    df['fullpath'] = df.apply(f, axis=1)
    df['content'] = df.apply(lambda r: open(r['fullpath'],
                                            encoding="latin1").read(), axis=1)
    return df


def read_wiki_20020():
    directory = os.path.join(DATASET_DIR, "wiki20020", "wiki20020")
    data = []
    for fname in os.listdir(directory):
        data.append(dict(fname=fname, fullpath=os.path.join(directory, fname)))
    df = pd.DataFrame(data)
    df['content'] = df.apply(lambda r: open(r['fullpath'],
                                            encoding="utf8").read(), axis=1)
    return df


def perplexity(model, test_doc):
    z = model.predict(test_doc)
    loglikelihood = model.approximate_likelihood(test_doc, z)
    return loglikelihood
