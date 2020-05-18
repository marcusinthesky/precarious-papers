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
matched = context.catalog.load("iex_matched_entities")


#%%
release = pd.to_datetime(context.params["release"]["paradise_papers"])
start = (
    release + pd.tseries.offsets.BDay(context.params["window"]["start"])
).to_pydatetime()
end = (
    release + pd.tseries.offsets.BDay(context.params["window"]["end"])
).to_pydatetime()


# %%
def get_financials(ticker):
    data_set = APIDataSet(
        url=f"https://cloud.iexapis.com/stable/stock/{ticker}/balance-sheet",
        params={"period": "annual", "last": 4, "token": IEX_APIKEY},
        json=True,
    )
    return data_set.load()


# %%
financials = (
    matched.loc[:, ["symbol"]]
    .drop_duplicates()
    .assign(financials=lambda df: df.symbol.apply(get_financials))
)

# %%
def balancesheet_to_frame(d):
    if "balancesheet" in d.keys() and "symbol" in d.keys():
        return pd.DataFrame(d["balancesheet"]).assign(symbol=d["symbol"])
    else:
        return pd.DataFrame()


balancesheet = pd.concat(financials.financials.apply(balancesheet_to_frame).tolist())
context.catalog.save("financials", balancesheet)
