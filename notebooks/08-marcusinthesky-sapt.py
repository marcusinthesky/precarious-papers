#%%
import holoviews as hv
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysal as ps
import seaborn as sns
from pysal import explore
from scipy.stats import norm, poisson
from statsmodels.regression.linear_model import OLS

hv.extension("bokeh")

#%%
# matched entities
matched = context.catalog.load("iex_matched_entities")  # scores_matches)

distances = context.io.load("paradise_distances")
price = context.io.load("paradise_price")
indices = context.io.load("indices")
balancesheet = context.catalog.load("financials")
release = pd.to_datetime(context.params["release"]["paradise_papers"])


# %%
index = (
    matched.groupby("symbol")
    .apply(lambda df: df.sample(1))
    .set_index("symbol")
    .exchange
)

rm_rates = (
    index.rename("index")
    .to_frame()
    .merge(
        indices.pct_change()
        .add(1)
        .cumprod()
        .tail(1)
        .subtract(1)
        .T.rename(columns=lambda x: "rm"),
        left_on="index",
        right_index=True,
    )
    .rm
)

# %%
returns = (
    price.loc[:, pd.IndexSlice[:, "close"]]
    .pct_change()
    .add(1)
    .cumprod()
    .tail(1)
    .subtract(1)
)

returns.columns = returns.columns.droplevel(1)
returns = returns.T.rename(columns=lambda x: "returns").dropna()
returns["excess"] = returns["returns"].subtract(rm_rates.loc[returns.index])
returns["rm"] = rm_rates.loc[returns.index].to_frame().to_numpy()

# remove islands
inner_index = returns.join(
    (
        distances.where(lambda df: df.sum() > 0)
        .dropna(how="all")
        .T.where(lambda df: df.sum() > 0)
        .dropna(how="all")
    ),
    how="inner",
).index

#%%
renamed_distances = distances.loc[inner_index, inner_index]
returns = returns.loc[inner_index, :]
exchange = pd.get_dummies(data=index).loc[inner_index, :]
communities = (
    pd.get_dummies(
        renamed_distances.where(
            lambda df: df.isna(), lambda df: df.apply(lambda x: x.index)
        )
        .fillna("")
        .apply(lambda df: hash(df.str.cat()))
    )
    .T.reset_index(drop=True)
    .T
)
communities = communities.loc[:, communities.sum() > 1]

factors = pd.merge_asof(
    returns.reset_index()
    .rename(columns={"index": "symbol"})
    .assign(reportDate=release)
    .sort_values("reportDate"),
    balancesheet.assign(
        reportDate=lambda df: pd.to_datetime(df.reportDate)
    ).sort_values("reportDate"),
    on="reportDate",
    by="symbol",
    direction="backward",
)

average_price = (
    price.loc[:, pd.IndexSlice[:, "close"]]
    .mean()
    .reset_index()
    .rename(columns={"level_0": "symbol", 0: "price"})
    .set_index("symbol")
    .price
)
features = (
    factors.set_index("symbol")
    .fillna(factors.mean())
    .assign(alpha=1)
    .select_dtypes(include="number")
    .join(average_price)
    .assign(
        hml=lambda df: df.price / (df.totalAssets / df.commonStock),
        smb=lambda df: df.price * df.commonStock,
    )
    .loc[:, ["rm", "hml", "smb", "alpha", "returns"]]
)

model = OLS(features.returns, features.drop(columns=["returns"]))
results = model.fit()
results.summary()

# %%
distribution = poisson(3)
D = (
    renamed_distances.loc[features.index, features.index]
    .replace(0, np.nan)
    .apply(distribution.pmf)
    .fillna(0)
)
ar_features = (D.dot(features) / D.dot(features.apply(pd.np.ones_like))).rename(
    columns=lambda x: x + "_ar"
)

# simultaneous
ar_model = OLS(
    features.returns,
    features.loc[:, ["rm", "alpha"]].join(ar_features.loc[:, "returns_ar"]),
)
ar_results = ar_model.fit()
ar_results.summary()

# exogenous
slx_model = OLS(
    features.returns,
    features.loc[:, ["rm", "alpha", "smb"]].join(ar_features.loc[:, ["smb_ar"]]),
)
slx_results = slx_model.fit()
slx_results.summary()
