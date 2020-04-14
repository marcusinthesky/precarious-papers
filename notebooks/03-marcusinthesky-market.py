# %%
from datetime import datetime
from typing import Dict, Union

import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
import holoviews as hv
from iexfinance.stocks import Stock, get_historical_data

SECRETS = context.config_loader.get("secrets.yml")
IEX_APIKEY = SECRETS["iex"]

hv.extension("bokeh")

#%%
paradise_entities = context.io.load("paradise_nodes_entity")
iex_matched_entities = context.catalog.load("iex_matched_entities")


#%%
stocks = (
    paradise_entities.merge(
        iex_matched_entities.drop(columns=["name"]), left_on="name", right_on="entities"
    )
    .where(lambda df: df.score == 100)
    .dropna(how="all")
)


#%%
historical = get_historical_data(
    symbols=(stocks.symbol.tolist()),
    start=datetime(2017, 10, 28),
    end=datetime(2017, 11, 12),
    close_only=True,
    token=IEX_APIKEY,
    output_format="pandas",
)

#%%
context.io.save("paradise_price", historical)

# %%
(
    historical
    # .loc[: , pd.IndexSlice[:, 'close']]
    .pct_change()
    .reset_index()
    .melt(id_vars="date", var_name=["stock", "metric"])
    .hvplot.line(x="date", y="value", by="stock", groupby="metric")
    .layout()
)
