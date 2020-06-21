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

    model = OLS(y, X.assign(alpha=1))
    results = model.fit()
    results.summary()

    tests = sms.jarque_bera(results.resid)
    print(tests)
    if tests[1] > 0.2:
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

#

# %%
# SLX : unrestricted, fixed \lambda
XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exog")], axis=1)
slx_unresticted = ps.model.spreg.OLS(
    y.to_numpy(),
    XGX.to_numpy(),
    name_x=XGX.columns.tolist(),
    w=w,
    spat_diag=True,
    moran=True,
)
print(slx_unresticted.summary)

# Adjusted R-squared  :      0.0564

# DIAGNOSTICS FOR SPATIAL DEPENDENCE
# TEST                           MI/DF       VALUE           PROB
# Lagrange Multiplier (lag)         1           0.595           0.4406
# Robust LM (lag)                   1           0.000           0.9827
# Lagrange Multiplier (error)       1           0.626           0.4289
# Robust LM (error)                 1           0.032           0.8591
# Lagrange Multiplier (SARMA)       2           0.626           0.7311


#             Variable     Coefficient       Std.Error     t-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT      -0.0231323       0.0708630      -0.3264377       0.7445987
#                   rm      -0.5914859       0.6830398      -0.8659610       0.3880479
#    price_to_earnings       0.0002494       0.0021772       0.1145532       0.9089695
# market_capitalization       0.0000000       0.0000000       1.1356064       0.2581334
#        profit_margin      -0.0373781       0.0130906      -2.8553354       0.0049788 **
#    price_to_research       0.0000034       0.0000047       0.7198249       0.4728772
#              rm_exog      -7.6433046      19.4029878      -0.3939241       0.6942586
# price_to_earnings_exog      -0.0064180       0.0301351      -0.2129749       0.8316678
# market_capitalization_exog       0.0000000       0.0000000       1.6821822       0.0948452 .
#   profit_margin_exog       0.0822042       0.1941782       0.4233442       0.6727178
# price_to_research_exog      -0.0000984       0.0002063      -0.4770157       0.6341222


# %%
# SLX : unrestricted, varying \lambda
def opt_lam(lam):
    try:
        distribution = stats.poisson(*lam.tolist())

        D = (
            (
                renamed_distances.loc[features.index, features.index]
                .replace(0, np.nan)
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
        model = ps.model.spreg.OLS(
            y.to_numpy(),
            XGX.to_numpy(),
            name_x=XGX.columns.tolist(),
            w=w,
            spat_diag=True,
            moran=True,
        )

        return -model.ar2, model, G

    except:
        print("error")
        return 0, None, None


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([samples.mean()]), tol=1e-6,)

pr2_e, slx_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(slx_unresticted_varying.summary)

# Adjusted R-squared  :      0.0923

# DIAGNOSTICS FOR SPATIAL DEPENDENCE
# TEST                           MI/DF       VALUE           PROB
# Lagrange Multiplier (lag)         1           1.288           0.2564
# Robust LM (lag)                   1           0.338           0.5608
# Lagrange Multiplier (error)       1           1.134           0.2870
# Robust LM (error)                 1           0.184           0.6682
# Lagrange Multiplier (SARMA)       2           1.472           0.4790

#             Variable     Coefficient       Std.Error     t-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT      -0.0703139       0.0485818      -1.4473313       0.1503074
#                   rm      -1.0425771       0.4763806      -2.1885379       0.0304890
#    price_to_earnings       0.0005655       0.0016768       0.3372456       0.7364973
# market_capitalization       0.0000000       0.0000000       1.7851038       0.0766697
#        profit_margin      -0.0142906       0.0099831      -1.4314832       0.1547873
#    price_to_research       0.0000051       0.0000441       0.1152248       0.9084518
#              rm_exog      -7.6982077       6.1931103      -1.2430277       0.2161842
# price_to_earnings_exog      -0.0539144       0.0389459      -1.3843392       0.1687210
# market_capitalization_exog       0.0000000       0.0000000       1.7109739       0.0895664
#   profit_margin_exog       0.4088909       0.2500903       1.6349731       0.1045705
# price_to_research_exog      -0.0002667       0.0004534      -0.5881385       0.5575005


# TEST ON NORMALITY OF ERRORS
# TEST                             DF        VALUE           PROB
# Jarque-Bera                       2           5.868           0.0532
# Accept H0 that errors are normal

# DIAGNOSTICS FOR HETEROSKEDASTICITY
# RANDOM COEFFICIENTS
# TEST                             DF        VALUE           PROB
# Breusch-Pagan test               10           5.263           0.8729
# Koenker-Bassett test             10           3.651           0.9617


# %%
XGX_opt = pd.concat([X, (G_opt @ X).rename(columns=lambda s: s + "_exogenous")], axis=1)
w_opt = ps.lib.weights.full2W(G_opt.to_numpy(), ids=G_opt.index.tolist())

sdm = ps.model.spreg.ML_Lag(
    y.to_numpy(),
    XGX_opt.to_numpy(),
    w=w_opt,
    name_x=XGX_opt.columns.tolist(),
    spat_diag=True,
)

print(sdm.summary)

OLS(sdm.e_pred, G_opt @ sdm.e_pred).fit().summary()


# %%
sdem = ps.model.spreg.ML_Error(
    y.to_numpy(),
    XGX_opt.to_numpy(),
    w=w_opt,
    name_x=XGX_opt.columns.tolist(),
    spat_diag=True,
)

print(sdem.summary)

OLS(sdem.e_filtered, G_opt @ y).fit().summary()


# %%

# kernel_lambda = 1.62635982

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT      -0.0630518       0.0461842      -1.3652236       0.1721828
#                   rm      -1.1135144       0.4528767      -2.4587585       0.0139418
#    price_to_earnings       0.0006254       0.0015912       0.3930243       0.6943016
# market_capitalization       0.0000000       0.0000000       1.9311239       0.0534677
#        profit_margin      -0.0142618       0.0094695      -1.5060829       0.1320459
#    price_to_research       0.0000043       0.0000418       0.1021696       0.9186221
#         rm_exogenous     -12.1283875       5.9343032      -2.0437762       0.0409757
# price_to_earnings_exogenous      -0.0484173       0.0370256      -1.3076723       0.1909845
# market_capitalization_exogenous       0.0000000       0.0000000       1.9034183       0.0569860
# profit_margin_exogenous       0.3832053       0.2374677       1.6137155       0.1065891
# price_to_research_exogenous      -0.0003341       0.0004337      -0.7705228       0.4409899
#            W_dep_var      -0.7128200       0.3931902      -1.8129141       0.0698451
