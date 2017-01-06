import pickle

import pandas as pd
from dataground import DataGround
from dataground.utils import read_twenty_news, read_wiki_20020
#  from dataground.measures import compare_experiment
from dataground.repr_models import compare_experiment


if "news_df" not in locals():
    try:
        news_df = pickle.load(open("news_df.pickle", "rb"))
    except:
        news_df = read_twenty_news()
        pickle.dump(news_df, open("news_df.pickle", "wb"))

if "wiki_df" not in locals():
    try:
        wiki_df = pickle.load(open("wiki_df.pickle", "rb"))
    except:
        wiki_df = read_wiki_20020()
        pickle.dump(wiki_df, open("wiki_df.pickle", "wb"))

if "dg" not in locals():
    content = list(news_df["content"]) + list(wiki_df["content"])
    # content = content[:8000]
    # content = content[-8000:]
    dg = DataGround(K=60, min_tf=5, min_df=2, max_dict_len=18000)
    # dg = DataGround(K=30, min_tf=5, min_df=2, max_dict_len=8000)
    dg.load_data(content)
    w_mi = dg.subset_w_mi()
    z_mi = dg.subset_z_mi()

if "measure_result" not in locals():
    measure_result = pd.DataFrame(index=["vl", "m1", "m3", "m5"])
# compare_experiment(dg)
