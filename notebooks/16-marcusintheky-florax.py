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
for i in range(29):
    # Winsorize
    X, y = features.drop(columns=["returns", "alpha"]), features.loc[:, ["returns"]]
    samples = renamed_distances.replace(0, np.nan).subtract(1).melt().value.dropna()
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
    if tests[1] > 0.15:
        break

    # Prob(JB):	0.0825

    # %%
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

# %%
# OLS : unrestricted, varying \lambda
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

        w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())
        model = ps.model.spreg.OLS(
            y.to_numpy(),
            X.to_numpy(),
            name_x=X.columns.tolist(),
            w=w,
            spat_diag=True,
            moran=True,
        )

        return -model.ar2, model, G

    except:
        print("error")
        return 0, None, None


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([average_degree]), tol=1e-6,)

pr2_e, ols_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(ols_unresticted_varying.summary)


# %%
# SER

ml_ser_fixed = ps.model.spreg.ML_Error(
    y.to_numpy(), X.to_numpy(), name_x=X.columns.tolist(), w=w, spat_diag=True,
)

print(ml_ser_fixed.summary)

# %%
print(sms.jarque_bera(ml_ser_fixed.e_filtered))
# Normal 0.1976079

# %%
# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(ml_ser_fixed.e_filtered, X)
list(zip(name, test))
# 6.184554197646399e-06
# a p-value below an appropriate threshold (e.g. p < 0.05) then the null hypothesis of homoskedasticity is rejected and heteroskedasticity assumed

## Goldfeld-Quandt test:
name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(ml_ser_fixed.e_filtered, X)
list(zip(name, test))
#  0.9939822296514494
# null hypothesis of homoskedastic errors


# %%
# OLS : unrestricted, varying \lambda
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

        w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())
        model = ps.model.spreg.ML_Error(
            y.to_numpy(), X.to_numpy(), name_x=X.columns.tolist(), w=w, spat_diag=True,
        )

        return -model.pr2, model, G

    except:
        print("error")
        return 0, None, None


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([average_degree]), tol=1e-6,)

pr2_e, ml_sem_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(ml_sem_unresticted_varying.summary)


# %%
print(sms.jarque_bera(ml_sem_unresticted_varying.e_filtered))
# Normal 0.19517863

# %%
# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(ml_sem_unresticted_varying.e_filtered, X)
list(zip(name, test))
# 6.222912079229905e-06
# a p-value below an appropriate threshold (e.g. p < 0.05) then the null hypothesis of homoskedasticity is rejected and heteroskedasticity assumed

## Goldfeld-Quandt test:
name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(ml_sem_unresticted_varying.e_filtered, X)
list(zip(name, test))
#  0.994032761410099
# null hypothesis of homoskedastic errors

# white test:
test = sms.het_white(ml_sem_unresticted_varying.e_filtered, X)
list(zip(name, test))
# 'p-value', 1.0
# null hypothesis of homoskedasticity

# %%
OLS(ml_sem_unresticted_varying.e_filtered, G_opt @ X).fit().summary()

# %%
OLS(
    ml_sem_unresticted_varying.e_filtered,
    G_opt @ X.loc[:, ["rm", "market_capitalization"]],
).fit().summary()
# conclude spatial heteroskedasticity

# 	coef	std err	t	P>|t|	[0.025	0.975]
# rm	-12.6364	4.619	-2.736	0.007	-21.786	-3.486
# market_capitalization	8.522e-13	3.02e-13	2.822	0.006	2.54e-13	1.45e-12

# %%
OLS(ml_sem_unresticted_varying.e_filtered, X).fit().summary()

# %%
XGX_opt_restricted = pd.concat(
    [
        X,
        (G_opt @ X.loc[:, ["rm", "market_capitalization"]]).rename(
            columns=lambda s: s + "_exogenous"
        ),
    ],
    axis=1,
)
ml_sdem_restricted = ps.model.spreg.GM_Error(
    y.drop(["ERA", "GES"]).to_numpy(),
    XGX_opt_restricted.drop(["ERA", "GES"]).to_numpy(),
    name_x=XGX_opt_restricted.columns.tolist(),
    w=ps.lib.weights.full2W(
        G_opt.drop(["ERA", "GES"]).drop(columns=["ERA", "GES"]).to_numpy(),
        ids=G.drop(["ERA", "GES"]).index.tolist(),
    ),
    # spat_diag=True,
)

print(ml_sdem_restricted.summary)

print(sms.jarque_bera(ml_sdem_restricted.e_filtered))
# 0.04378934


# %%
XGX_opt_restricted = pd.concat(
    [
        X,
        (G_opt @ X.loc[:, ["rm", "market_capitalization"]]).rename(
            columns=lambda s: s + "_exogenous"
        ),
    ],
    axis=1,
)
gm_sdem_restricted = ps.model.spreg.GM_Error(
    y.to_numpy(),
    XGX_opt_restricted.to_numpy(),
    name_x=XGX_opt_restricted.columns.tolist(),
    w=w,
)

print(gm_sdem_restricted.summary)

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0084609       0.0141569       0.5976523       0.5500720
#                   rm      -7.1855364       2.1808459      -3.2948391       0.0009848
#    price_to_earnings      -0.0059613       0.0026862      -2.2192336       0.0264708
# market_capitalization      -0.0000000       0.0000000      -0.6478164       0.5171037
#        profit_margin       0.0106866       0.0139421       0.7664948       0.4433819
#    price_to_research       0.0000048       0.0000034       1.3854999       0.1658997
#         rm_exogenous     -10.0784827       8.1018632      -1.2439710       0.2135102
# market_capitalization_exogenous       0.0000000       0.0000000       2.4428897       0.0145702
#               lambda      -0.8543645

# %%
XGX_opt_restricted = pd.concat(
    [
        X,
        (G_opt @ X.loc[:, ["rm", "market_capitalization"]]).rename(
            columns=lambda s: s + "_exogenous"
        ),
    ],
    axis=1,
)
gm_sdem_het_restricted = ps.model.spreg.GM_Error_Het(
    y.to_numpy(),
    XGX_opt_restricted.to_numpy(),
    name_x=XGX_opt_restricted.columns.tolist(),
    w=w,
)

print(gm_sdem_het_restricted.summary)

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0102322       0.0167660       0.6102925       0.5416681
#                   rm      -7.1559484       2.3025861      -3.1077875       0.0018849
#    price_to_earnings      -0.0060563       0.0034524      -1.7542093       0.0793946
# market_capitalization      -0.0000000       0.0000000      -0.6642087       0.5065568
#        profit_margin       0.0109518       0.0133651       0.8194323       0.4125398
#    price_to_research       0.0000048       0.0000007       7.0926047       0.0000000
#         rm_exogenous     -11.4551624       7.1779106      -1.5958909       0.1105131
# market_capitalization_exogenous       0.0000000       0.0000000       1.9440723       0.0518867
#               lambda      -0.7609893       0.5418200      -1.4045059       0.1601683
