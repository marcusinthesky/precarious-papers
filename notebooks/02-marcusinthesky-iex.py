# %%
import json
import time
from typing import Dict, Union

import numpy as np
import pandas as pd
import requests
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_distances

from iexfinance.refdata import get_symbols


SECRETS = context.config_loader.get("secrets.yml")
IEX_APIKEY = SECRETS["iex"]

#%%
if context.catalog.exists("symbols"):
    symbols = context.catalog.load("symbols")
else:
    symbols = get_symbols(output_format="pandas", token=IEX_APIKEY)
    context.catalog.save("symbols", symbols)


#%%
entities = (
    pd.concat(
        [
            context.io.load(c)
            for c in context.catalog.list()
            if c.endswith("_nodes_entity")
        ],
        axis=0,
    )
    .reset_index(drop=True)
    .rename(columns={"index": "source_index"})
)

# %%
entities_in = entities.name.str.lower().isin(symbols.name.str.lower())
entities.name.loc[entities_in]

# %%
stop_words = [
    "limited",
    "ltd",
    "llc",
    "corp",
    "inc",
    "corperation",
    "incorperated",
    "limited",
    "holdings",
    "foundation",
    "sa",
    "systems",
    "consulting",
    "enterprises",
    "group",
    "financial",
    "services",
    "communications",
    "management",
    "enterprise",
    "foundation",
    "global",
    "international",
]

analyzer = TfidfVectorizer(analyzer="word", smooth_idf=False, stop_words=stop_words)

symbols_vec = analyzer.fit_transform(symbols.name)

# %%
def get_best_match(names, reference, reference_analyzed, analyzer, chunk_size=100):

    names_vec = analyzer.transform(names.fillna(""))
    match_argmin, match_min = pairwise_distances_argmin_min(
        names_vec, reference_analyzed, metric="cosine", metric_kwargs={"n_jobs": -1}
    )

    return pd.DataFrame(
        {
            "idx": names.index,
            "entities": names.to_numpy(),
            "match": reference[match_argmin],
        }
    )


matches = (
    entities.assign(chunk=lambda df: df.index % 10)
    .groupby(["chunk"])
    .name.apply(lambda df: get_best_match(df, symbols.name, symbols_vec, analyzer, 100))
    .reset_index(drop=True)
    .set_index("idx")
)

# %%
pat = r"\b(?:{})\b".format("|".join(stop_words))

fuzzy_name = (
    matches.astype(str)
    .apply(lambda df: (df.str.lower().str.replace("[^\w\s]", "").str.replace(pat, "")))
    .assign(fuzzy_ratio=lambda df: df.apply(lambda s: fuzz.ratio(s[0], s[1]), axis=1))
)

d = paired_distances(
    analyzer.transform(matches.entities.fillna("")),
    analyzer.transform(matches.match.fillna("")),
    "cosine",
)

# %%
scores_matches = (
    matches.join(fuzzy_name.fuzzy_ratio)
    .assign(cosine_similarity=100 * (1 - d))
    .assign(score=lambda df: (df.cosine_similarity + df.fuzzy_ratio) / 2)
    .merge(symbols, left_on=["match"], right_on=["name"])
)

scores_matches.nlargest(1000, "score")


# %%
context.catalog.save("iex_matched_entities", scores_matches)


# %%
