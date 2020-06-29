#%%
from functools import reduce
from operator import add

import holoviews as hv
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysal as ps
import seaborn as sns
import statsmodels.api as sm
from pysal import explore
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, poisson
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.api as sms

# from bromoulle
from statsmodels.sandbox.regression.gmm import IV2SLS

hv.extension("bokeh")

#%%
# matched entities
features = context.catalog.load("features")
renamed_distances = context.catalog.load("renamed_distances")
price = context.io.load("paradise_price")
indices = context.io.load("indices")
matched = context.catalog.load("iex_matched_entities")  # scores_matches)


# %%
returns = (
    price.loc[:, pd.IndexSlice[:, "close"]]
    .pct_change()
    .reset_index()
    .melt(id_vars="date", var_name=["symbol", "type"], value_name="returns")
    .where(lambda df: df.symbol.isin(renamed_distances.index))
    .assign(regime=lambda df: (df.date - df.date.min()).dt.days)
)

data = (
    returns.merge(
        features.drop(columns=["returns", "rm"]),
        left_on=["symbol"],
        right_index=True,
        how="left",
    )
    .merge(
        matched.drop(columns=["date"]),
        left_on=["symbol"],
        right_on=["symbol"],
        how="left",
    )
    .merge(
        indices.pct_change()
        .reset_index()
        .melt(id_vars=["Date"], var_name="exchange", value_name="rm")
        .rename(columns={"Date": "date"}),
        left_on=["date", "exchange"],
        right_on=["date", "exchange"],
        how="left",
    )
)


# %%
F = (
    data.loc[
        :,
        [
            "rm",
            "price_to_earnings",
            "market_capitalization",
            "profit_margin",
            "price_to_research",
            "regime",
            "returns",
            "symbol",
        ],
    ]
    .dropna()
    .set_index("symbol")
)
X, r, y = (
    F.loc[
        :,
        [
            "rm",
            "price_to_earnings",
            "market_capitalization",
            "profit_margin",
            "price_to_research",
        ],
    ],
    F.regime,
    F.loc[:, ["returns"]],
)
distribution = stats.poisson(4)

f = r.to_numpy().reshape(-1, 1)

G = (
    renamed_distances.loc[F.index, F.index]
    .replace(0, np.nan)
    .apply(distribution.pmf)
    .fillna(0)
    .multiply(np.equal(f, f.T))
    .apply(lambda x: x / x.sum(), axis=1)
)


ols = ps.model.spreg.OLS(
    y=y.apply(stats.mstats.winsorize, limits=[0.075, 0.075]).to_numpy(),
    x=X.to_numpy(),
    spat_diag=True,
    moran=True,
    name_x=X.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(ols.summary)

# %%
sar = ps.model.spreg.GM_Lag(
    y=y.apply(stats.mstats.winsorize, limits=[0.075, 0.075]).to_numpy(),
    x=X.to_numpy(),
    spat_diag=True,
    w_lags=2,
    name_x=X.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(sar.summary)

# %%
ser = ps.model.spreg.GM_Error_Het(
    y=y.apply(stats.mstats.winsorize, limits=[0.075, 0.075]).to_numpy(),
    x=X.to_numpy(),
    step1c=True,
    name_x=X.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(ser.summary)

# %%
XGX = pd.concat(
    [
        X,
        (G @ X.drop(columns=["market_capitalization", "price_to_research"])).rename(
            columns=lambda s: s + "_exogenous"
        ),
    ],
    axis=1,
)
sdm = ps.model.spreg.GM_Lag(
    y=y.apply(stats.mstats.winsorize, limits=[0.075, 0.075]).to_numpy(),
    x=XGX.to_numpy(),
    spat_diag=True,
    name_x=XGX.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(sdm.summary)

# Anselin-Kelejian Test             1           4.990          0.0255
# There is residual spatial correlation


# %%
sdem = ps.model.spreg.GM_Error_Het(
    y=y.apply(stats.mstats.winsorize, limits=[0.075, 0.075]).to_numpy(),
    x=XGX.to_numpy(),
    step1c=True,
    name_x=XGX.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(sdem.summary)
