#%%
import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
import holoviews as hv
import matplotlib.pyplot as plt
import seaborn as sns
import pysal as ps
from pysal import explore
from scipy.stats import norm
from iexfinance.stocks import Stock, get_historical_data, HistoricalReader
from pandas.tseries.offsets import BDay
from precarious_papers.io import APIDataSet

SECRETS = context.config_loader.get("secrets.yml")
IEX_APIKEY = SECRETS["iex"]

hv.extension("bokeh")

#%%
# matched entities
paradise_entities = context.io.load("paradise_nodes_entity")
iex_matched_entities = context.catalog.load("iex_matched_entities")
distances = context.io.load("paradise_distances")
price = context.io.load("paradise_price")


#%%
release = pd.to_datetime(context.params["release"]["paradise_papers"])
start = (
    release + pd.tseries.offsets.BDay(context.params["window"]["start"])
).to_pydatetime()
end = (
    release + pd.tseries.offsets.BDay(context.params["window"]["end"])
).to_pydatetime()

#%%

companies = (
    paradise_entities.merge(
        iex_matched_entities.drop(columns=["name"]), left_on="name", right_on="entities"
    )
    .set_index("node_id")
    .drop_duplicates()
    .where(lambda x: x.score == 100)
    .dropna(how="all")
)

# %%
symbol = "AAPL"
data_set = APIDataSet(
    url=f"https://cloud.iexapis.com/stable/stock/{symbol}/balance-sheet",
    params={"period": "annual", "last": 4, "token": IEX_APIKEY},
    json=True,
)

# %%
data = data_set.load()

# %%
