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


# %%
samples = renamed_distances.replace(0, np.nan).melt().value.dropna()

X, y = features.drop(columns=["returns", "alpha"]), features.loc[:, ["returns"]]

distribution = stats.poisson(4.661996779388084)

D = (
    renamed_distances.loc[features.index, features.index]
    .replace(0, np.nan)
    .apply(distribution.pmf)
    .fillna(0)
)

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
# Lagrange Multiplier (lag)         1           0.219           0.6398
# Robust LM (lag)                   1           0.261           0.6092
# Lagrange Multiplier (error)       1           0.148           0.7006  # https://www.tandfonline.com/doi/abs/10.1080/03610918.2013.781626
# Robust LM (error)                 1           0.190           0.6627
# Lagrange Multiplier (SARMA)       2           0.409           0.8150

# __Reject & Conclude:__ \rho = 0 ; \lambda = 0

# %%
# SLX : unrestricted, fixed \lambda
XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exog")], axis=1)
slx_unresticted = ps.model.spreg.OLS(
    y.to_numpy(), XGX.to_numpy(), name_x=XGX.columns.tolist(), w=w, spat_diag=True
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
            renamed_distances.loc[features.index, features.index]
            .replace(0, np.nan)
            .apply(distribution.pmf)
            .fillna(0)
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
        )

        return -model.ar2, model, G

    except:
        print("error")
        return 0, None, None


best_param = minimize(
    lambda x: opt_lam(x)[0], x0=np.array([4.661996779388084]), tol=1e-6,
)

pr2_e, slx_unresticted, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(slx_unresticted.summary)

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
#             CONSTANT      -0.0964947       0.0362427      -2.6624601       0.0086999
#                   rm      -0.5810657       0.6590867      -0.8816225       0.3795480
#    price_to_earnings       0.0013110       0.0021698       0.6041827       0.5467362
# market_capitalization       0.0000000       0.0000000       0.8948871       0.3724402
#        profit_margin      -0.0296935       0.0128575      -2.3094326       0.0224374 *
#    price_to_research       0.0000025       0.0000042       0.6000757       0.5494617
#              rm_exog      -2.8261209       4.6899954      -0.6025850       0.5477957
# price_to_earnings_exog       0.0280325       0.0222323       1.2608878       0.2095240
# market_capitalization_exog       0.0000000       0.0000000       1.3398966       0.1825300
#   profit_margin_exog       0.2027940       0.1159498       1.7489813       0.0825670
# price_to_research_exog      -0.0002248       0.0001188      -1.8918424       0.0606538 .
