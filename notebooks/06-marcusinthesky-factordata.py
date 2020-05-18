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
def get_balance_sheet(ticker):
    data_set = APIDataSet(
        url=f"https://cloud.iexapis.com/stable/stock/{ticker}/balance-sheet",
        params={"period": "annual", "last": 4, "token": IEX_APIKEY},
        json=True,
    )
    return data_set.load()


# %%
balance_sheet_data = (
    matched.loc[:, ["symbol"]]
    .drop_duplicates()
    .assign(balance_sheet=lambda df: df.symbol.apply(get_balance_sheet))
)

# %%
def balancesheet_to_frame(d):
    if "balancesheet" in d.keys() and "symbol" in d.keys():
        return pd.DataFrame(d["balancesheet"]).assign(symbol=d["symbol"])
    else:
        return pd.DataFrame()


balancesheet = pd.concat(
    balance_sheet_data.balance_sheet.apply(balancesheet_to_frame).tolist()
)
context.catalog.save("balance_sheet", balancesheet)

# %%
def get_income_statement(ticker):
    try:
        data_set = APIDataSet(
            url=f"https://cloud.iexapis.com/stable/stock/{ticker}/income",
            params={"period": "annual", "last": 4, "token": IEX_APIKEY},
            json=True,
        )
        return data_set.load()
    except:
        return {}


income_statement_data = (
    matched.loc[:, ["symbol"]]
    .drop_duplicates()
    .assign(income_statement=lambda df: df.symbol.apply(get_income_statement))
)

#%%
def income_statement_to_frame(d):
    if "income" in d.keys() and "symbol" in d.keys():
        return pd.DataFrame(d["income"]).assign(symbol=d["symbol"])
    else:
        return pd.DataFrame()


income_statement = pd.concat(
    income_statement_data.income_statement.apply(income_statement_to_frame).tolist()
)
context.catalog.save("income_statement", income_statement)


# %%
def get_market_cap(ticker):
    data_set = APIDataSet(
        url=f"https://cloud.iexapis.com/stable/stock/{ticker}/stats/marketcap",
        params={"period": "annual", "last": 4, "token": IEX_APIKEY},
        json=True,
    )
    return data_set.load()


market_cap_data = (
    matched.loc[:, ["symbol"]]
    .drop_duplicates()
    .assign(market_cap=lambda df: df.symbol.apply(get_market_cap))
)

context.catalog.save("market_cap", market_cap_data)
