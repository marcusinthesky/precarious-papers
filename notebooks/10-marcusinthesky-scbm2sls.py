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
samples = renamed_distances.replace(0, np.nan).melt().value.dropna()

# %%
b, loc, scale = stats.pareto.fit(samples)


def hyperparam(x):
    b, loc, scale = x
    distribution = stats.pareto(b, loc, scale)
    # k, mu = x
    # distribution = stats.poisson(k, mu)
    D = (
        renamed_distances.loc[features.index, features.index]
        .replace(0, np.nan)
        .apply(distribution.pdf)
        .fillna(0)
    )

    # G = D
    G = D.apply(lambda x: x / np.sum(x), 1)

    I = np.identity(G.shape[1])
    X = features.drop(columns=["alpha", "returns"])
    y = features.loc[:, ["returns"]]
    IV = pd.concat([(I - G) @ X, (I - G) @ G @ X, (I - G) @ G @ G @ X], axis=1)
    Xt = pd.concat(
        [
            ((I - G) @ G @ y).rename(columns=lambda x: "endogenous"),
            (I - G) @ X,
            ((I - G) @ G @ X).rename(columns=lambda x: x + "_exogenous"),
        ],
        axis=1,
    )

    B = Xt.assign(alpha=1).drop(
        columns=[
            "profit_margin_exogenous",
            "market_capitalization_exogenous",
            # "rm_exogenous", "price_to_research_exogenous",'price_to_research',
            "profit_margin",
            "price_to_earnings",
            "market_capitalization",
        ]
    )
    ivsls = IV2SLS(y, B, IV)
    ivsls_results = ivsls.fit()


best_param = minimize(
    lambda x: hyperparam(x)[0],
    x0=[1.2455455044025334, -0.001699277293891539, 2.001699277208737],
    method="Nelder-Mead",
    tol=1e-6,
)

r_squared, results, G = hyperparam(best_param.x)

# summary
results.summary()


# %%
# probablity plot
osm_osr, slope_intercept_r = stats.probplot(results.resid, dist="norm")

(
    pd.DataFrame(dict(zip(("Quantile", "Ordered Values"), osm_osr))).hvplot.scatter(
        y="Ordered Values", x="Quantile"
    )
    * pd.DataFrame(
        {
            "Quantile": [np.min(osm_osr[0]), np.max(osm_osr[0])],
            "Ordered Values": [
                slope_intercept_r[0] * np.min(osm_osr[0]) + slope_intercept_r[1],
                slope_intercept_r[0] * np.max(osm_osr[0]) + slope_intercept_r[1],
            ],
        }
    ).hvplot.line(y="Ordered Values", x="Quantile", color="black")
).opts(title="Probability Plot")

stats.ttest_1samp(results.resid, 0, axis=0)

# heteroskedasticity plots
(
    reduce(
        add,
        [
            v.to_frame()
            .assign(Residuals=results.resid)
            .hvplot.scatter(x=k, y="Residuals")
            for k, v in B.drop(columns=["alpha"])
            .rename(columns=lambda s: " ".join(s.split("_")).title())
            .items()
        ],
    )
    * hv.HLine(0).opts(color="black")
).cols(1)


# %%
# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(results.resid, results.model.exog)
list(zip(name, test))

## Goldfeld-Quandt test:
name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(results.resid, results.model.exog)
list(zip(name, test))

# Linearity:
name = ["t value", "p value"]
test = sms.linear_harvey_collier(results)
list(zip(name, test))


# SLX
Xt = X.drop(columns=["market_capitalization", "price_to_research", "price_to_earnings"])
model = OLS(
    y, Xt.assign(alpha=1, profit_margin_exogenous=G @ X.profit_margin.to_frame())
).fit()
model.summary()

# SAR
Xt = X.drop(columns=["market_capitalization", "price_to_research", "price_to_earnings"])
model = OLS(y, Xt.assign(alpha=1, endogenous=G @ y)).fit()
model.summary()

# G2SLS
S = pd.concat([(I - G) @ X, (I - G) @ G @ X, (I - G) @ G @ G @ X], axis=1)
P = S @ pd.DataFrame(np.linalg.pinv(S.T @ S), index=S.columns, columns=S.columns) @ S.T
Xt = pd.concat([(I - G) @ G @ y, (I - G) @ X, (I - G) @ G @ X], axis=1)
Th2sls = (
    pd.DataFrame(np.linalg.pinv(Xt.T @ P @ Xt), columns=Xt.columns, index=Xt.columns)
    @ Xt.T
    @ P
    @ y
).rename(columns={"returns": "Theta_2sls"})
Zh = pd.concat(
    [
        (I - G) @ G @ (Xt @ Th2sls),
        (I - G) @ X,
        ((I - G) @ G @ X).rename(columns=lambda s: s + "_exogenous"),
    ],
    axis=1,
)
Thlle = (
    pd.DataFrame(np.linalg.pinv(Zh.T @ Xt), columns=Zh.columns, index=Zh.columns)
    @ Zh.T
    @ y
).rename(columns={"returns": "Theta_lle"})
d = y.subtract((Zh @ Thlle).to_numpy()).pow(2).sum()
D = pd.DataFrame(np.identity(Zh.shape[0]), columns=Zh.index, index=Zh.index).multiply(
    d.item()
)
Vhlle = (
    pd.DataFrame(np.linalg.pinv(Zh.T @ Xt), index=Zh.columns, columns=Zh.columns)
    @ Zh.T
    @ D
    @ Zh
    @ pd.DataFrame(np.linalg.pinv(Zh.T @ Xt), index=Zh.columns, columns=Zh.columns)
)
SE_Vhlle = pd.DataFrame(np.diag(Vhlle), index=Vhlle.columns, columns=["SE"])
T_lle = (Thlle / SE_Vhlle.to_numpy()).rename(columns={"Theta_lle": "T_lle"})
p = T_lle.apply(stats.t(df=139).sf).rename(columns={"T_lle": "p_lle"})
