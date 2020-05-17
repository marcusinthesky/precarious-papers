#%%
import holoviews as hv
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysal as ps
import seaborn as sns
from pysal import explore
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS

hv.extension("bokeh")

#%%
# matched entities
matched = context.catalog.load("iex_matched_entities")  # scores_matches)

distances = context.io.load("paradise_distances")
price = context.io.load("paradise_price")
indices = context.io.load("indices")

# %%
index = (
    matched.groupby("symbol")
    .apply(lambda df: df.sample(1))
    .set_index("symbol")
    .exchange
)

rf_rates = (
    index.rename("index")
    .to_frame()
    .merge(
        indices.pct_change()
        .add(1)
        .cumprod()
        .tail(1)
        .subtract(1)
        .T.rename(columns=lambda x: "rf"),
        left_on="index",
        right_index=True,
    )
    .rf
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
returns["excess"] = returns["returns"].subtract(rf_rates.loc[returns.index])
returns["rf"] = rf_rates.loc[returns.index].to_frame().to_numpy()

inner_index = returns.join(distances, how="inner").index

#%%
renamed_distances = distances.loc[inner_index, inner_index]
returns = returns.loc[inner_index, :]
exchange = pd.get_dummies(data=index).loc[inner_index, :]

#%%
# Whittle (1954) /wo exchange
connected = renamed_distances.fillna(0).clip(upper=1)

community_model = OLS(
    returns.returns,
    pd.DataFrame(
        {
            "Community Average": (connected.multiply(returns.returns, 0)).sum(0)
            / (connected).sum(0),
            "alpha": 1,
        },
        index=returns.index,
    )
    .fillna(0)
    .assign(beta=returns.rf),
)
community_results = community_model.fit()
community_results.summary()

distance_model = OLS(
    returns.returns,
    pd.DataFrame(
        {
            "Weighted Average": (
                renamed_distances.fillna(0).multiply(returns.returns, 0)
            ).sum(0)
            / (connected).sum(0),
            "alpha": 1,
        }
    )
    .fillna(0)
    .assign(beta=returns.rf),
)
distance_results = distance_model.fit()
distance_results.summary()

model = OLS(
    returns.returns,
    pd.DataFrame(
        {
            "Community Average": (connected.multiply(returns.returns, 0)).sum(0)
            / (connected).sum(0),
            "Weighted Average": (
                renamed_distances.fillna(0).multiply(returns.returns, 0)
            ).sum(0)
            / (connected).sum(0),
            "alpha": 1,
        }
    )
    .fillna(0)
    .assign(beta=returns.rf),
)
results = model.fit()
results.summary()

size_model = OLS(
    returns.returns.to_numpy(),
    pd.DataFrame({"Community size": connected.sum(0), "alpha": 1}).assign(
        beta=returns.rf
    ),
)
size_results = size_model.fit()
size_results.summary()

weight = (1 / renamed_distances.fillna(0)).replace(np.inf, 0)
weighted_model = OLS(
    returns.returns,
    pd.DataFrame(
        {
            "Community Average": connected.multiply(returns.returns, 0).sum(0)
            / (connected).sum(0),
            "Weighted Average": (weight).multiply(returns.returns, 0).sum("rows")
            / (weight).sum("rows"),
            "alpha": 1,
        }
    )
    .fillna(0)
    .loc[returns.returns.index, :]
    .assign(beta=returns.rf),
)
weighted_results = weighted_model.fit()
weighted_results.summary()


#%%
# Whittle (1954) /wo exchange
connected = renamed_distances.fillna(0).clip(upper=1)
community_model = OLS(
    returns.excess,
    pd.DataFrame(
        {
            "Community Average": connected.multiply(returns.excess, 0).sum(0)
            / (connected).sum(0),
            "alpha": 1,
        },
        index=returns.index,
    ).fillna(0),
)
community_results = community_model.fit()
community_results.summary()

from sklearn.neighbors import kneighbors_graph

neighbor = pd.DataFrame(
    kneighbors_graph(
        renamed_distances.fillna(renamed_distances.max().max()),
        1,
        mode="connectivity",
        include_self=False,
    ).todense(),
    index=renamed_distances.index,
    columns=renamed_distances.columns,
)
neighbor_model = OLS(
    returns.excess,
    pd.DataFrame(
        {
            "Neighbor Average": (neighbor.multiply(returns.excess, 0)).sum(0)
            / (neighbor).sum(0),
            "alpha": 1,
        },
        index=returns.index,
    ).fillna(0),
)
neighbor_results = neighbor_model.fit()
neighbor_results.summary()

distance_model = OLS(
    returns.excess,
    pd.DataFrame(
        {
            "Weighted Average": (
                renamed_distances.fillna(0).multiply(returns.excess, 0)
            ).sum(0)
            / (connected).sum(0),
            "alpha": 1,
        }
    ).fillna(0),
)
distance_results = distance_model.fit()
distance_results.summary()

full_model = OLS(
    returns.excess,
    pd.DataFrame(
        {
            "Community Average": connected.multiply(returns.excess, 0).sum(0)
            / (connected).sum(0),
            "Weighted Average": (
                renamed_distances.fillna(0).multiply(returns.excess, 0)
            ).sum(0)
            / (connected).sum(0),
            "alpha": 1,
        }
    ).fillna(0),
)
full_results = full_model.fit()
full_results.summary()

size_model = OLS(
    returns.excess.to_numpy(),
    pd.DataFrame({"Community size": connected.sum(0), "alpha": 1}),
)
size_results = size_model.fit()
size_results.summary()

similarity_model = OLS(
    returns.excess,
    pd.DataFrame(
        {
            "Weighted Average": ((1 / renamed_distances.fillna(0)).replace(np.inf, 0))
            .multiply(returns.excess, 0)
            .sum("rows")
            / (connected).sum("rows"),
            "alpha": 1,
        }
    )
    .fillna(0)
    .loc[returns.excess.index, :],
)
similarity_results = similarity_model.fit()
similarity_results.summary()

weight = (1 / renamed_distances.fillna(0)).replace(np.inf, 0)
weighted_model = OLS(
    returns.excess,
    pd.DataFrame(
        {
            "Community Average": connected.multiply(returns.excess, 0).sum(0)
            / (connected).sum(0),
            "Weighted Average": (weight).multiply(returns.excess, 0).sum("rows")
            / (weight).sum("rows"),
            "alpha": 1,
        }
    )
    .fillna(0)
    .loc[returns.excess.index, :],
)
weighted_results = weighted_model.fit()
weighted_results.summary()
