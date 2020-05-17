# %%
from datetime import datetime, timedelta
from typing import Dict, Union

import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
import holoviews as hv
from iexfinance.stocks import Stock, get_historical_data

from pandas.tseries.offsets import BDay

SECRETS = context.config_loader.get("secrets.yml")
IEX_APIKEY = SECRETS["iex"]

hv.extension("bokeh")

#%%
# matched entities
paradise_entities = context.io.load("paradise_nodes_entity")
stocks = context.catalog.load("iex_matched_entities")

#%%
release = pd.to_datetime(context.params["release"]["paradise_papers"])
start = (
    release + pd.tseries.offsets.BDay(context.params["window"]["start"])
).to_pydatetime()
end = (
    release + pd.tseries.offsets.BDay(context.params["window"]["end"])
).to_pydatetime()

#%%
unique_tickers = stocks.symbol.drop_duplicates().tolist()

historical_prices = []
chunks = np.array_split(unique_tickers, (len(unique_tickers)) // 100 + 1)
for c in chunks:
    historical_prices.append(
        get_historical_data(
            symbols=c.tolist(),
            start=start,
            end=end,
            close_only=True,
            token=IEX_APIKEY,
            output_format="pandas",
        )
    )

historical = pd.concat(historical_prices, axis=1)

#%%
context.io.save("paradise_price", historical)

# %%
(
    historical.loc[:, pd.IndexSlice[:, "close"]]
    .pct_change()
    .reset_index()
    .melt(id_vars="date", var_name=["stock", "metric"])
    .hvplot.line(x="date", y="value", by="stock", groupby="metric")
    .layout()
)


# %%
