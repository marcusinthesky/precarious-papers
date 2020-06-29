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
features = context.catalog.load("features")
renamed_distances = context.catalog.load("renamed_distances")

samples = renamed_distances.melt().value.dropna()
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
    y = y.apply(stats.mstats.winsorize, limits=[0.05, 0.05])
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

        # Robust LM (error) seems most robust lets optimize for it
        return model.rlm_lag[-1], model, G

    except:
        print("error")
        return 0, None, None


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([average_degree]), tol=1e-6,)

pr2_e, ols_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(ols_unresticted_varying.summary)

# lam opt = 4.03330841

#             Variable     Coefficient       Std.Error     t-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0014657       0.0052182       0.2808851       0.7792863
#                   rm      -4.4774835       1.9443717      -2.3027919       0.0230276
#    price_to_earnings      -0.0017466       0.0019693      -0.8869139       0.3769142
# market_capitalization       0.0000000       0.0000000       0.0107371       0.9914512
#        profit_margin       0.0104346       0.0121419       0.8593851       0.3918560
#    price_to_research       0.0000052       0.0000026       2.0188798       0.0457474


# %%
# SER
ml_ser_fixed = ps.model.spreg.ML_Error(
    y.to_numpy(),
    X.to_numpy(),
    name_x=X.columns.tolist(),
    w=ps.lib.weights.full2W(G_opt.to_numpy(), ids=G_opt.index.tolist()),
    spat_diag=True,
)

print(ml_ser_fixed.summary)

# %%
print(sms.jarque_bera(ml_ser_fixed.e_filtered))
# Normal 0.467073

# %%
# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(ml_ser_fixed.e_filtered, X)
list(zip(name, test))
# 0.00010472361799582236
# a p-value below an appropriate threshold (e.g. p < 0.05) then the null hypothesis of homoskedasticity is rejected and heteroskedasticity assumed

## white test:
name = ["F statistic", "p-value"]
test = sms.het_white(ml_ser_fixed.e_filtered, X)
list(zip(name, test))
#  1
# null hypothesis of homoskedastic errors


# %%
# ML : unrestricted, varying \lambda
def opt_lam(lam):
    try:
        print(lam)
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

        w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())
        model = ps.model.spreg.ML_Error(
            y.to_numpy(), X.to_numpy(), name_x=X.columns.tolist(), w=w, spat_diag=True,
        )

        return model.betas[-1], model, G

    except:
        print("error")
        return 1, None, None


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([average_degree]), tol=1e-6,)

pr2_e, ml_sem_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(ml_sem_unresticted_varying.summary)

# opt lam = 3.99


# %%
print(sms.jarque_bera(ml_sem_unresticted_varying.e_filtered))
# Normal 0.46539263

# %%
# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(ml_sem_unresticted_varying.e_filtered, X)
list(zip(name, test))
# 0.00010546212393206634
# a p-value below an appropriate threshold (e.g. p < 0.05) then the null hypothesis of homoskedasticity is rejected and heteroskedasticity assumed

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
# rm	-11.0171	3.766	-2.926	0.004	-18.471	-3.563
# market_capitalization	6.416e-13	2.15e-13	2.979	0.003	2.15e-13	1.07e-12


# %%
XGX_opt_restricted = pd.concat(
    [
        X,
        (G_opt @ X.loc[:, ["rm", "market_capitalization"]]).rename(
            columns=lambda s: s + "_exogenous"
        ),
    ],
    axis=1,
).drop(columns=["market_capitalization", "profit_margin", "price_to_research"])
ml_sdem_restricted = ps.model.spreg.ML_Error(
    y.to_numpy(),
    XGX_opt_restricted.to_numpy(),
    name_x=XGX_opt_restricted.columns.tolist(),
    w=ps.lib.weights.full2W(G_opt.to_numpy(), ids=G_opt.index.tolist(),),
)

print(ml_sdem_restricted.summary)

print(sms.jarque_bera(ml_sdem_restricted.e_filtered))
# 0.68366267


# %%
# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(ml_sdem_restricted.e_filtered, X)
list(zip(name, test))
# 2.812656689440267e-05
# a p-value below an appropriate threshold (e.g. p < 0.05) then the null hypothesis of homoskedasticity is rejected and heteroskedasticity assumed

# white test:
test = sms.het_white(ml_sdem_restricted.e_filtered, X)
list(zip(name, test))
# 'p-value', 1.0
# null hypothesis of homoskedasticity


# %%
XGX_opt_restricted = pd.concat(
    [
        X,
        (G_opt @ X.loc[:, ["rm", "market_capitalization"]]).rename(
            columns=lambda s: s + "_exogenous"
        ),
    ],
    axis=1,
).drop(columns=["market_capitalization", "profit_margin", "price_to_research"])
gm_sdem_restricted = ps.model.spreg.GM_Error(
    y.to_numpy(),
    XGX_opt_restricted.to_numpy(),
    name_x=XGX_opt_restricted.columns.tolist(),
    w=w,
)

print(gm_sdem_restricted.summary)

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0332839       0.0147710       2.2533300       0.0242384
#                   rm      -5.2568759       1.7709625      -2.9683722       0.0029938
#    price_to_earnings      -0.0016499       0.0017774      -0.9282971       0.3532535
#         rm_exogenous     -24.5862776       9.0511391      -2.7163738       0.0066001
# market_capitalization_exogenous       0.0000000       0.0000000       2.5514177       0.0107286
#               lambda      -1.0000000

# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(gm_sdem_restricted.e_filtered, X)
list(zip(name, test))
# 2.4856660552697322e-05
# a p-value below an appropriate threshold (e.g. p < 0.05) then the null hypothesis of homoskedasticity is rejected and heteroskedasticity assumed

# white test:
test = sms.het_white(gm_sdem_restricted.e_filtered, X)
list(zip(name, test))
# 'p-value', 1.0
# null hypothesis of homoskedasticity

# %%
XGX_opt_restricted = pd.concat(
    [
        X,
        (
            G_opt
            @ X.drop(
                columns=["price_to_earnings", "profit_margin", "price_to_research"]
            )
        ).rename(columns=lambda s: s + "_exogenous"),
    ],
    axis=1,
)  # .drop(columns=["rm_exogenous", "market_capitalization_exogenous", "market_capitalization", "profit_margin", "price_to_earnings"])

gm_sdem_het_restricted = ps.model.spreg.GM_Error_Het(
    y.to_numpy(),
    XGX_opt_restricted.to_numpy(),
    name_x=XGX_opt_restricted.columns.tolist(),
    w=ps.lib.weights.full2W(G_opt.to_numpy(), ids=G_opt.index.tolist(),),
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

XGX_opt_restricted = pd.concat(
    [
        X,
        (
            G_opt
            @ X.drop(
                columns=[
                    "price_to_earnings",
                    "profit_margin",
                    "price_to_research",
                    "market_capitalization",
                ]
            )
        ).rename(columns=lambda s: s + "_exogenous"),
    ],
    axis=1,
).drop(columns=["market_capitalization", "profit_margin", "price_to_earnings"])
slx_restricted = ps.model.spreg.OLS(
    y.to_numpy(),
    XGX_opt_restricted.to_numpy(),
    name_x=XGX_opt_restricted.columns.tolist(),
    w=ps.lib.weights.full2W(G_opt.to_numpy(), ids=G_opt.index.tolist(),),
    robust="white",  # heteroskadastistic consistent
    spat_diag=True,
    moran=True,
)

print(slx_restricted.summary)


exclude = []
for i in range(50):
    # Winsorize
    X, y = features.drop(columns=["returns", "alpha"]), features.loc[:, ["returns"]]
    samples = renamed_distances.replace(0, np.nan).melt().value.dropna()
    # average_degree = 2.9832619059417067

    distribution = stats.poisson(best_param.x)  # average_degree)  # samples.mean())

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

    G = D.apply(lambda x: x / np.sum(x), 1)

    XGX_opt_restricted = pd.concat(
        [
            X,
            (
                G
                @ X.drop(
                    columns=[
                        "price_to_earnings",
                        "profit_margin",
                        "price_to_research",
                        "market_capitalization",
                    ]
                )
            ).rename(columns=lambda s: s + "_exogenous"),
        ],
        axis=1,
    ).drop(columns=["market_capitalization", "profit_margin", "price_to_earnings"])

    model = OLS(y, XGX_opt_restricted.assign(alpha=1), cov_type="H3")
    results = model.fit()
    results.summary()

    tests = sms.jarque_bera(results.resid)
    print(tests)
    if tests[1] > 0.05:
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


slx_varying_resticted_filtered = ps.model.spreg.OLS(
    y=y.to_numpy(),
    x=XGX_opt_restricted.to_numpy(),
    name_x=XGX_opt_restricted.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist(),),
    spat_diag=True,
    moran=True,
)

print(slx_varying_resticted_filtered.summary)
