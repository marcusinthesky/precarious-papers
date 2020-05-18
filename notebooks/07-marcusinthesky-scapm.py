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

from statsmodels.stats.api import anova_lm
from statsmodels.formula.api import ols

rehab_lm = ols(
    "excess ~ C(community)",
    data=returns.loc[:, ["excess"]].assign(community=communities.idxmax(1)),
).fit()
table9 = anova_lm(rehab_lm)
print(table9)

(
    renamed_distances.melt()
    .value.where(lambda x: x > 0)
    .dropna()
    .rename("Degree")
    .hvplot.hist(title="Average Degree of Connection")
)

#%%
model = OLS(returns.returns, pd.DataFrame({"beta": returns.rm, "alpha": 1,}))
results = model.fit()
results.summary()

community_model = OLS(
    returns.returns,
    pd.DataFrame(
        {
            "beta": returns.rm,
            # "alpha": 1,
            "Neighbor Average": (neighbor).multiply(returns.returns, 0).sum("rows")
            / (neighbor).sum("rows"),
            "Kernel Average": (weight).multiply(returns.returns, 0).sum("rows")
            / (weight).sum("rows"),
        }
    ).join(communities),
)
community_results = community_model.fit()
community_results.summary()

# %%
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
    .assign(beta=returns.rm),
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
    .assign(beta=returns.rm),
)
distance_results = distance_model.fit()
distance_results.summary()


size_model = OLS(
    returns.returns.to_numpy(),
    pd.DataFrame({"Community size": connected.sum(0), "alpha": 1}).assign(
        beta=returns.rm
    ),
)
size_results = size_model.fit()
size_results.summary()

from sklearn.neighbors import kneighbors_graph

neighbor = pd.DataFrame(
    kneighbors_graph(
        renamed_distances.fillna(renamed_distances.max().max()),
        1,
        metric="precomputed",
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

distribution = poisson(2.98)
kernel_weight = distribution.pmf(renamed_distances)
np.fill_diagonal(kernel_weight, 0)
kernel_weight = pd.DataFrame(
    kernel_weight, columns=renamed_distances.columns, index=renamed_distances.index
).fillna(0)
kernel_weighted_model = OLS(
    returns.returns,
    pd.DataFrame(
        {
            "Kernel Average": (kernel_weight).multiply(returns.returns, 0).sum("rows")
            / (kernel_weight).sum("rows"),
            "alpha": 1,
        }
    )
    .fillna(0)
    .loc[returns.returns.index, :]
    .assign(beta=returns.rm),
)
kernel_weighted_results = kernel_weighted_model.fit()
kernel_weighted_results.summary()


distribution = poisson(2.98)
neg_exp = distribution.pmf(renamed_distances)
np.fill_diagonal(neg_exp, 0)
neg_exp_weight = pd.DataFrame(
    neg_exp, columns=renamed_distances.columns, index=renamed_distances.index
).fillna(0)
neg_exp_weighted_model = OLS(
    returns.returns,
    pd.DataFrame(
        {
            "Community Average": connected.multiply(returns.returns, 0).sum(0)
            / (connected).sum(0),
            # "Neighbor Average": (neighbor.multiply(returns.excess, 0)).sum(0)
            # / (neighbor).sum(0),
            "Kernel Average": (neg_exp_weight).multiply(returns.returns, 0).sum("rows")
            / (neg_exp_weight).sum("rows"),
            "alpha": 1,
        }
    )
    .fillna(0)
    .loc[returns.returns.index, :]
    .assign(beta=returns.rm),
)
neg_exp_weighted_results = neg_exp_weighted_model.fit()
neg_exp_weighted_results.summary()

community_averages = (
    communities.idxmax(1)
    .rename("community")
    .to_frame()
    .merge(
        returns.returns.to_frame()
        .assign(community=communities.idxmax(1))
        .groupby("community")
        .mean(),
        left_on="community",
        right_index=True,
    )
    .returns
)
weight = renamed_distances.fillna(0)
community_degree = OLS(
    returns.returns,
    pd.DataFrame(
        {
            "Clique Average": community_averages,
            "Degree Average": (weight).multiply(returns.returns, 0).sum("rows")
            / (weight).sum("rows"),
            "alpha": 1,
        }
    )
    .fillna(0)
    .loc[returns.returns.index, :]
    .assign(beta=returns.rm),
)
community_degree.fit().summary()


weight = renamed_distances.fillna(0)
weight_weighted_model = OLS(
    returns.returns,
    pd.DataFrame(
        {
            "Community Average": connected.multiply(returns.returns, 0).sum(0)
            / (connected).sum(0),
            "Degree Average": (weight).multiply(returns.returns, 0).sum("rows")
            / (weight).sum("rows"),
            "alpha": 1,
        }
    )
    .fillna(0)
    .loc[returns.returns.index, :]
    .assign(beta=returns.rm),
)
weight_weighted_results = weight_weighted_model.fit()
weight_weighted_results.summary()


(
    pd.DataFrame(
        {
            "Community Average": connected.multiply(returns.returns, 0).sum(0)
            / (connected).sum(0),
            "Neighbor Average": (neighbor.multiply(returns.excess, 0)).sum(0)
            / (neighbor).sum(0),
            "Degree Average": (weight).multiply(returns.returns, 0).sum("rows")
            / (weight).sum("rows"),
            "Kernel Average": (neg_exp_weight).multiply(returns.returns, 0).sum("rows")
            / (neg_exp_weight).sum("rows"),
        }
    )
    .fillna(0)
    .loc[returns.returns.index, :]
    .assign(beta=returns.rm, returns=returns.returns)
    .corr()
)

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
            "Neighbor Average": (neighbor.multiply(returns.excess, 0)).sum(0)
            / (neighbor).sum(0),
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
