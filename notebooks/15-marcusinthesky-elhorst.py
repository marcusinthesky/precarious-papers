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
for i in range(50):  # 29):
    # Winsorize
    X, y = features.drop(columns=["returns", "alpha"]), features.loc[:, ["returns"]]
    samples = renamed_distances.replace(0, np.nan).melt().value.dropna()
    average_degree = 2.9832619059417067

    distribution = stats.poisson(average_degree)  # samples.mean())

    D = (
        renamed_distances.loc[features.index, features.index]
        .replace(0, np.nan)
        .subtract(1)
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

    model = OLS(y, X.assign(alpha=1))
    results = model.fit()
    results.summary()

    tests = sms.jarque_bera(results.resid)
    print(tests)
    if tests[1] > 0.2:
        break

    # Prob(JB):	0.0825

    excluder = (
        pd.Series(results.get_influence().influence, index=y.index).nlargest(1).index[0]
    )
    print(excluder)
    exclude.append(excluder)


# %%
results.summary()

# %%
# Reject normality assumption Leptokurtic present
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.influence_plot(results, ax=ax, criterion="DFFITS")

# %%
pd.Series(results.get_influence().influence, index=y.index).nlargest(5)

# %%
# G = D
G = D.apply(lambda x: x / np.sum(x), 1)
w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())


# %%
# OLS
ols_unresticted = ps.model.spreg.OLS(
    y.to_numpy(), X.to_numpy(), name_x=X.columns.tolist(), w=w, spat_diag=True
)
print(ols_unresticted.summary)

# DIAGNOSTICS FOR SPATIAL DEPENDENCE
# TEST                           MI/DF       VALUE           PROB
# Lagrange Multiplier (lag)         1           0.431           0.5117
# Robust LM (lag)                   1           3.499           0.0614
# Lagrange Multiplier (error)       1           1.080           0.2988
# Robust LM (error)                 1           4.147           0.0417
# Lagrange Multiplier (SARMA)       2           4.578           0.1014

#
XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exog")], axis=1)

ml_sdm_saturated_restricted = ps.model.spreg.ML_Lag(
    y=y.to_numpy(),
    x=XGX.drop(
        columns=[
            "profit_margin",
            "market_capitalization",
            "market_capitalization_exog",
            "price_to_earnings",
            "price_to_earnings_exog",
        ]
    ).to_numpy(),
    w=ps.lib.weights.full2W(G.to_numpy()),
    # w_lags=2,
    name_x=XGX.drop(
        columns=[
            "profit_margin",
            "market_capitalization",
            "market_capitalization_exog",
            "price_to_earnings",
            "price_to_earnings_exog",
        ]
    ).columns.tolist(),
    method="ord",
    spat_diag=True,
)
print(ml_sdm_saturated_restricted.summary)

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0056146       0.0341245       0.1645315       0.8693127
#                   rm      -5.0964789       2.0776041      -2.4530558       0.0141648
#    price_to_earnings      -0.0064092       0.0026884      -2.3840347       0.0171240
# market_capitalization       0.0000000       0.0000000       1.1763083       0.2394717
#        profit_margin       0.0184933       0.0136333       1.3564821       0.1749458
#    price_to_research       0.0000062       0.0000028       2.2092396       0.0271580
#              rm_exog     -19.4552053      10.7176699      -1.8152458       0.0694861
# price_to_earnings_exog      -0.0307462       0.0249312      -1.2332396       0.2174864
# market_capitalization_exog       0.0000000       0.0000000       0.0708137       0.9435460
#   profit_margin_exog       0.1201035       0.1408940       0.8524393       0.3939703
# price_to_research_exog       0.0000685       0.0000544       1.2589228       0.2080582
#            W_dep_var      -0.9264340       0.3608007      -2.5677171       0.0102371

# %%
# normality
sms.jarque_bera(ml_sdm_saturated_restricted.e_pred)
# p-value 0.05446924 \exclusions 0.19219642s

# %%
# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(ml_sdm_saturated_restricted.e_pred, X.drop(["GES", "ERA"]))
list(zip(name, test))
# 7.868291708399792e-07
# a p-value below an appropriate threshold (e.g. p < 0.05) then the null hypothesis of homoskedasticity is rejected and heteroskedasticity assumed

## Goldfeld-Quandt test:
name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(
    ml_sdm_saturated_restricted.e_pred, X.drop(["GES", "ERA"])
)
list(zip(name, test))
#  0.9994384339642036
# null hypothesis of homoskedastic errors

# white test:
test = sms.het_white(ml_sdm_saturated_restricted.e_pred, X.drop(["GES", "ERA"]))
list(zip(name, test))
# 'p-value', 1.0
# null hypothesis of homoskedasticity

# %%
OLS(ml_sdm_saturated_restricted.e_pred, X.drop(["GES", "ERA"])).fit().summary()

OLS(np.abs(ml_sdm_saturated_restricted.e_pred), X.drop(["GES", "ERA"])).fit().summary()


# %%
OLS(
    ml_sdm_saturated_restricted.e_pred,
    G.drop(["GES", "ERA"]).drop(columns=["GES", "ERA"]).to_numpy()
    @ X.drop(["GES", "ERA"]),
).fit().summary()


OLS(
    np.abs(ml_sdm_saturated_restricted.e_pred),
    G.drop(["GES", "ERA"]).drop(columns=["GES", "ERA"]).to_numpy()
    @ X.drop(["GES", "ERA"]),
).fit().summary()


# %%
# ML SDM : unrestricted, varying \lambda
def opt_lam(lam):
    try:
        print(lam)
        distribution = stats.poisson(*lam.tolist())

        D = (
            (
                renamed_distances.loc[features.index, features.index]
                .replace(0, np.nan)
                .subtract(1)
                .apply(distribution.pmf)
                .fillna(0)
            )
            .drop(exclude)
            .drop(columns=exclude)
        )

        # G = D
        G = D.apply(lambda x: x / np.sum(x), 1)

        XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exog")], axis=1)

        w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())
        model = ps.model.spreg.ML_Lag(
            y.to_numpy(),
            XGX.to_numpy(),
            name_x=XGX.columns.tolist(),
            w=w,
            spat_diag=True,
        )

        return -model.pr2_e, model, G

    except:
        print("error")
        return 0, None, None


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([average_degree]), tol=1e-6,)

pr2_e, ml_sdm_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(ml_sdm_unresticted_varying.summary)

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0056146       0.0341245       0.1645315       0.8693127
#                   rm      -5.0964789       2.0776041      -2.4530558       0.0141648
#    price_to_earnings      -0.0064092       0.0026884      -2.3840347       0.0171240
# market_capitalization       0.0000000       0.0000000       1.1763083       0.2394717
#        profit_margin       0.0184933       0.0136333       1.3564821       0.1749458
#    price_to_research       0.0000062       0.0000028       2.2092396       0.0271580
#              rm_exog     -19.4552053      10.7176699      -1.8152458       0.0694861
# price_to_earnings_exog      -0.0307462       0.0249312      -1.2332396       0.2174864
# market_capitalization_exog       0.0000000       0.0000000       0.0708137       0.9435460
#   profit_margin_exog       0.1201035       0.1408940       0.8524393       0.3939703
# price_to_research_exog       0.0000685       0.0000544       1.2589228       0.2080582
#            W_dep_var      -0.9264340       0.3608007      -2.5677171       0.0102371

#%%
sms.jarque_bera(ml_sdm_unresticted_varying.e_pred)
# p-value 0.05446924
