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
from scipy import stats
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm


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

# %%
# %%
samples = renamed_distances.replace(0, np.nan).melt().value.dropna()
distributions = pd.Series(
    ["gamma", "invgamma", "invgauss", "norm", "pareto"], name="Distributions"
)


def likelihood(d):
    parameters = getattr(stats, d).fit(samples)
    dist = getattr(stats, d)(*parameters)
    return samples.apply(dist.logpdf).multiply(-1).sum()


ll = distributions.to_frame().assign(
    likelihood=lambda df: df.Distributions.apply(likelihood)
)

p = stats.poisson(samples.mean())
samples.apply(p.logpmf).multiply(-1).sum()

# %%
distribution = stats.pareto(*stats.pareto.fit(samples))  # 3)
D = (
    renamed_distances.loc[features.index, features.index]
    .replace(0, np.nan)
    .apply(distribution.pdf)
    .fillna(0)
)

# # Test for rank Bramoulle et al 2009
# from numpy.linalg import matrix_rank
# assert matrix_rank(D) == D.shape[0]
# assert matrix_rank(D@D) == D.shape[0]


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

# %%
# exogenous
slx_model = OLS(
    features.returns,
    features.loc[:, ["rm", "alpha",]].join(ar_features.drop(columns=["alpha_ar"])),
)
slx_results = slx_model.fit()
slx_results.summary()

# %%


def p_elimination(x, y, sl):
    x = pd.DataFrame(x)
    noCols = x.shape[1]
    for i in range(0, noCols):
        ols_regeressor = OLS(exog=x.assign(alpha=1), endog=y).fit()
        pValues = ols_regeressor.pvalues
        if max(pValues) > sl:
            x = x.drop(np.argmax(pValues), axis=1)
        else:
            break
    return x


def backward_elimination(x, y, sl):
    x = pd.DataFrame(np.append(np.ones((x.shape[0], 1)).astype(int), x, axis=1))
    rSqData = pd.DataFrame(r_sq_elimination(x, y))
    finalModel = p_elimination(rSqData, y, sl)
    return finalModel


def r_sq_elimination(x, y):
    x = pd.DataFrame(x)
    ols_regeressor = OLS(exog=x.assign(alpha=1), endog=y).fit()
    bestValue = float("{0:.4f}".format(ols_regeressor.rsquared_adj))
    noOfColumn = x.shape[1]
    bestModel = pd.DataFrame(x)
    foundNew = False
    for i in range(1, noOfColumn):
        temp = x.drop(x.columns[i], axis=1)
        ols_regeressor = OLS(endog=y, exog=temp).fit()
        rValue = float("{0:.4f}".format(ols_regeressor.rsquared_adj))
        if bestValue < rValue:
            bestValue = rValue
            bestModel = temp
            foundNew = True

    if foundNew == True:
        bestModel = r_sq_elimination(bestModel, y)
        return bestModel
    return bestModel, ols_regeressor.aic


selected_features, aic = r_sq_elimination(
    x=features.drop(columns=["returns", "alpha"]).join(
        ar_features.drop(columns=["alpha_ar"])
    ),
    y=features.returns,
)

# selected
selected_model = OLS(
    features.returns, features.loc[:, ["alpha",]].join(selected_features),
)
selected_results = selected_model.fit()
selected_results.summary()


# %%
# [markdown] Investigate factors which describe the most likely entity-relationship depth
from scipy.optimize import minimize


def dist_param(lam):
    distribution = stats.poisson(lam)
    D = (
        renamed_distances.loc[features.index, features.index]
        .drop(["MSFT", "SAVE"])
        .drop(columns=["MSFT", "SAVE"])
        .replace(0, np.nan)
        .apply(distribution.pmf)
        .fillna(0)
    )
    spatial_corr_features = (
        D.dot(features.drop(columns=[]).drop(["MSFT", "SAVE"]))
        / D.dot(features.drop(columns=[]).drop(["MSFT", "SAVE"]).apply(pd.np.ones_like))
    ).rename(columns=lambda x: x + "_ar")
    exog = (
        features.drop(["MSFT", "SAVE"])
        .drop(columns=["returns", "alpha"])
        .join(spatial_corr_features.drop(columns=["alpha_ar"]))
    )
    feat, aic = r_sq_elimination(
        x=exog.fillna(exog.mean()), y=features.returns.drop(["MSFT", "SAVE"])
    )
    return feat, aic


best_lam = minimize(lambda l: dist_param(l)[1], x0=5, method="Nelder-Mead", tol=1e-6)

selected_features, best_aic = dist_param(best_lam.x)


selected_exogenous = (
    features.loc[:, ["alpha",]]
    .join(selected_features)
    .rename(columns={"rm": "(rm - rf)", "rm_ar": "(rm - rf)_ar"})
    .rename(
        columns=lambda s: f"W({s.split('_')[0]})"
        if len(s.split("_")) == 2
        else s.split("_")[0]
    )
    # .drop(columns=['rm_ar', 'price_to_earnings_ar', 'profit_margin_ar'])
)
# selected

X = selected_exogenous.drop(["MSFT", "SAVE"])
y = features.returns.rename("ri - rf").drop(["MSFT", "SAVE"])
Xt = X.loc[:, ["W(price_to_earnings)", "W(profit_margin)", "W(price_to_research)"]]
yt = X.loc[:, ["W(returns)"]]
Mt = OLS(yt, Xt.assign(alpha=1)).fit()
Et = Mt.resid

selected_model = OLS(y, X)
selected_results = selected_model.fit()
selected_results.summary()


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.influence_plot(selected_results, ax=ax, criterion="cooks")


fig, ax = plt.subplots(figsize=(8, 6))
fig = sm.graphics.plot_leverage_resid2(selected_results, ax=ax)

fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_partregress_grid(selected_results, fig=fig)

fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(selected_results, fig=fig)


res = selected_results.resid  # residuals
fig = sm.qqplot(res, fit=True, line="45", color="black")
plt.title("Q-Q Plots of Residuals")
plt.show()

res.hvplot.hist(xlabel="Residuals", title="Distribution of Residuals")

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
exog = (selected_features).drop(columns=[])
pd.Series(
    [variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])],
    index=exog.columns,
)
