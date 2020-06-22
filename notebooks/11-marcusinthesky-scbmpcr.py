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
from scipy.stats import norm, poisson
from statsmodels.regression.linear_model import OLS

# from bromoulle
from statsmodels.sandbox.regression.gmm import IV2SLS

hv.extension("bokeh")

#%%
# matched entities
matched = context.catalog.load("iex_matched_entities")  # scores_matches)

distances = context.io.load("paradise_distances")
price = context.io.load("paradise_price")
indices = context.io.load("indices")
balancesheet = context.catalog.load("balance_sheet")
release = pd.to_datetime(context.params["release"]["paradise_papers"])
income = context.catalog.load("income_statement")


# %%
index = (
    matched.groupby("symbol")
    .apply(lambda df: df.sample(1))
    .set_index("symbol")
    .exchange
)

rf = ((1 + 3.18e-05) ** 5) - 1

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
    .rm.subtract(rf)
)


# %%
returns = (
    price.loc[:, pd.IndexSlice[:, "close"]]
    .pct_change()
    .add(1)
    .cumprod()
    .tail(1)
    .subtract(1)
    .subtract(rf)
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
    balancesheet.merge(
        income.drop(columns=["minorityInterest", "fiscalDate", "currency"]),
        on=["reportDate", "symbol"],
    )
    .assign(reportDate=lambda df: pd.to_datetime(df.reportDate))
    .sort_values("reportDate"),
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
    .select_dtypes(include="number")
    .join(average_price)
    .assign(
        price_to_earnings=lambda df: df.price / (df.totalAssets / df.commonStock),
        market_capitalization=lambda df: df.price * df.commonStock,
        profit_margin=lambda df: df.grossProfit / df.totalRevenue,
        price_to_research=lambda df: (df.price * df.commonStock)
        / df.researchAndDevelopment,
    )
    .loc[
        :,
        [
            "rm",
            "price_to_earnings",
            "market_capitalization",
            "profit_margin",
            "price_to_research",
            "returns",
        ],
    ]
)

model = OLS(features.returns, features.drop(columns=["returns"]))
results = model.fit()
results.summary()

# %%
samples = renamed_distances.replace(0, np.nan).subtract(1).melt().value.dropna()

# %%
distribution = stats.pareto(*stats.pareto.fit(samples))  # 3)
D = (
    renamed_distances.loc[features.index, features.index]
    .replace(0, np.nan)
    .subtract(1)
    .apply(distribution.pdf)
    .fillna(0)
)

spatial_features = pd.concat(
    [
        features,
        (D @ features.drop(columns="returns")).rename(
            columns=lambda s: s + "_exogenous"
        ),
        (D @ features.loc[:, ["returns"]]).rename(columns=lambda s: s + "_endogenous"),
    ],
    axis=1,
).assign(alpha=1)


# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pca = Pipeline([("scale", StandardScaler()), ("pca", PCA())])
pca.fit(spatial_features.drop(columns="alpha").dropna())
pd.Series(pca.named_steps["pca"].explained_variance_ratio_).hvplot.bar(
    title="Variance Explained Ratio"
)


select_pca = Pipeline([("scale", StandardScaler()), ("pca", PCA(5))])
spatial_features_components = select_pca.fit_transform(
    spatial_features.drop(columns="alpha").dropna()
)

# selected
pcr_exogenous = pd.DataFrame(
    spatial_features_components,
    index=spatial_features.drop(columns="alpha").dropna().index,
).assign(alpha=1)

pcr_model = OLS(
    features.returns.rename("ri - rf").loc[pcr_exogenous.index], pcr_exogenous
)
pcr_results = pcr_model.fit()
pcr_results.summary()

# %%
# Davison and Hinkley (1997), Bootstrap Methods and their Application, p. 141.
# Note: drop high influence causes nan's, which must be dropped
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression

n_samples = 1000
bootstap_coefficients = pd.DataFrame(
    np.zeros((n_samples, spatial_features.shape[1])), columns=spatial_features.columns,
)


def bootstrap_samples(zeros):
    y_sample, X_sample = resample(
        features.returns.rename("ri - rf"), spatial_features.drop(columns=["alpha"])
    )

    select_pcr = Pipeline(
        [("scale", StandardScaler()), ("pca", PCA(X_sample.shape[1]))]
    )

    model_pcr = LinearRegression(fit_intercept=True)

    model_pcr.fit(select_pcr.fit_transform(X_sample), y_sample)

    return pd.Series(
        np.hstack(
            (
                model_pcr.intercept_,
                select_pcr.named_steps["pca"]
                .inverse_transform(model_pcr.coef_.reshape(1, -1))
                .flatten(),
            )
        ),
        index=spatial_features.columns,
    )


bootstap_coefficients_df = bootstap_coefficients.apply(
    bootstrap_samples, axis="columns"
)
bootstap_coefficients_df.hvplot.box()

bootstap_coefficients_df.describe()
bootstap_coefficients_df.quantile([0.01, 0.5, 0.99])
