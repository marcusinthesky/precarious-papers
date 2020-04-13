# %%
import json
import time
from typing import Dict, Union

import numpy as np
import pandas as pd
import requests
from fuzzywuzzy import fuzz

SECRETS = context.config_loader.get("secrets.yml")
OPENFIGI_APIKEY = SECRETS["openfigi"]

# %%
def openfigisearch(query: str, sleep: float = 0.01) -> Union[Dict, None]:
    time.sleep(sleep)
    """submits search to openfigi API

    :param query: search term
    :raises Exception: if no response from api
    :return: dictionary of data from first response
    """
    openfigi_url = "https://api.openfigi.com/v2/search"
    openfigi_headers = {"Content-Type": "application/json"}
    if OPENFIGI_APIKEY:
        openfigi_headers["X-OPENFIGI-APIKEY"] = OPENFIGI_APIKEY
    response = requests.post(
        url=openfigi_url,
        headers=openfigi_headers,
        data=json.dumps({"query": query, "marketSecDes": "Equity"}),
    )

    if response.status_code != 200:
        # raise Exception("Bad response code {}".format(str(response.status_code)))
        return np.nan

    decoded = response.json()
    if (
        "data" in decoded.keys()
        and isinstance(decoded["data"], list)
        and len(decoded["data"]) > 0
    ):
        return decoded["data"]


#%%
entities = context.io.load("paradise_nodes_entity")

# %%
sample = (
    entities.where(lambda df: ~df.company_type.isna()).dropna(0, how="all").sample(100)
)
figi = sample.assign(openfigi=lambda x: x.name.apply(lambda s: openfigisearch(s)))

# %%
names = (
    figi.assign(
        openfigi_names=lambda df: df.openfigi.apply(
            lambda d: [i["name"].lower() for i in d] if isinstance(d, list) else np.nan
        )
    )
    .loc[:, ["name", "openfigi_names"]]
    .explode("openfigi_names")
    .dropna()
)


# %%
fuzzy_name = names.assign(
    fuzzy_ratio=lambda df: df.apply(lambda s: fuzz.ratio(s[0], s[1]), axis=1)
)


# %%
names.sort_values("fuzzy_ratio", ascending=False)
