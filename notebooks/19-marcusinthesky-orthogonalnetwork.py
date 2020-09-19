#%%
from functools import reduce, partial
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
    .assign(alpha=1)
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
            "alpha",
            "returns",
        ],
    ]
)

model = OLS(features.returns, features.drop(columns=["returns"]))
results = model.fit()
results.summary()

# Reject normality assumption Leptokurtic present
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.influence_plot(results, ax=ax, criterion="cooks")

# %%
exclude = []
for i in range(25):
    # Winsorize
    X, y = features.drop(columns=["returns", "alpha"]), features.loc[:, ["returns"]]
    samples = renamed_distances.replace(0, np.nan).melt().value.dropna()
    average_degree = 2.9832619059417067

    distribution = stats.poisson(average_degree)  # samples.mean())

    D = (
        renamed_distances.loc[features.index, features.index]
        .replace(0, np.nan)
        .apply(distribution.pmf)
        .fillna(0)
    )

    try:
        X = X.drop(exclude)
        y = y.drop(exclude)
        # G = G.drop(exclude).drop(columns=exclude)
        D = D.drop(exclude).drop(columns=exclude)
    except:
        print("skipped")
        pass

    model = OLS(y, X.assign(alpha=1), cov_type="HC3")
    results = model.fit()
    results.summary()

    tests = sms.jarque_bera(results.resid)
    print(tests)
    if tests[1] > 0.1:
        break

    # Prob(JB):	0.0825

    excluder = (
        pd.Series(results.get_influence().influence, index=y.index).nlargest(1).index[0]
    )
    print(excluder)
    exclude.append(excluder)

    # cook = pd.Series(results.get_influence().cooks_distance[0], index=y.index).nlargest(1).index[0]
    # print(cook)
    # if exclude != cook:
    #     exclude.append(cook)


# %%
results.summary()


# %%
# Reject normality assumption Leptokurtic present
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.influence_plot(results, ax=ax, criterion="DFFITS")

# %%
pd.Series(results.get_influence().influence, index=y.index).nlargest(5)

# %%
y_index = y.returns.nlargest(y.shape[0] - 1).index
y = y.loc[y_index, :]
X = X.loc[y.index, :]
renamed_distances = renamed_distances.loc[y.index, y.index]
D = D.loc[y.index, y.index]
G = D.apply(lambda x: x / np.sum(x), 1)

G = G.loc[y.index, y.index]
features = features.loc[y.index, :]


# %%
# G = D
# drop_var = G.var().nsmallest(3).index
# X, y, G = X.drop(drop_var), y.drop(drop_var), G.drop(drop_var).drop(columns=drop_var)
w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())


# %%
ols = ps.model.spreg.OLS(
    y.to_numpy(),
    X.to_numpy(),
    w=w,
    name_x=X.columns.tolist(),
    spat_diag=True,
    moran=True,
)

print(ols.summary)

# %%
def model(d, G, fit_intercept=True):
    if fit_intercept:
        return OLS(d, G @ d.to_frame().assign(alpha=1)).fit()
    else:
        return OLS(d, G @ d.to_frame()).fit()


models = (
    X.apply(partial(model, G=G, fit_intercept=True), axis=0)
    .rename("model")
    .to_frame()
    .assign(
        pvalue=lambda df: df.model.apply(lambda s: s.pvalues[0]).round(3),
        coef=lambda df: df.model.apply(lambda s: s.params[0]).round(3),
    )
)
models

# %%
filtered_features = models.pvalue.where(lambda s: s > 0.01).dropna().index
X_restricted = X.loc[:, filtered_features]
