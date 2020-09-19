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
price = context.io.load("paradise_price")
indices = context.io.load("indices")
matched = context.catalog.load("iex_matched_entities")  # scores_matches)


# %%
returns = (
    price.loc[:, pd.IndexSlice[:, "close"]]
    .pct_change()
    .reset_index()
    .melt(id_vars="date", var_name=["symbol", "type"], value_name="returns")
    .where(lambda df: df.symbol.isin(renamed_distances.index))
    .assign(regime=lambda df: (df.date - df.date.min()).dt.days)
)

data = (
    returns.merge(
        features.drop(columns=["returns", "rm"]),
        left_on=["symbol"],
        right_index=True,
        how="left",
    )
    .merge(
        matched.drop(columns=["date"]),
        left_on=["symbol"],
        right_on=["symbol"],
        how="left",
    )
    .merge(
        indices.pct_change()
        .reset_index()
        .melt(id_vars=["Date"], var_name="exchange", value_name="rm")
        .rename(columns={"Date": "date"}),
        left_on=["date", "exchange"],
        right_on=["date", "exchange"],
        how="left",
    )
)


# %%
F = (
    data.loc[
        :,
        [
            "rm",
            "price_to_earnings",
            "market_capitalization",
            "profit_margin",
            "price_to_research",
            "regime",
            "returns",
            "symbol",
        ],
    ]
    .dropna()
    .set_index("symbol")
)
X, r, y = (
    F.loc[
        :,
        [
            "rm",
            "price_to_earnings",
            "market_capitalization",
            "profit_margin",
            "price_to_research",
        ],
    ],
    F.regime,
    F.loc[:, ["returns"]],
)
distribution = stats.poisson(2.8)

f = r.to_numpy().reshape(-1, 1)

G = (
    renamed_distances.loc[F.index, F.index]
    .replace(0, np.nan)
    .apply(distribution.pmf)
    .fillna(0)
    .multiply(np.equal(f, f.T))
    .apply(lambda x: x / x.sum(), axis=1)
)


# %%
def best_winsor(l):
    print(l)
    y_clip = y.apply(stats.mstats.winsorize, limits=[l, l])
    JB, JBpv, skew, kurtosis = sms.jarque_bera(y_clip)
    return -JBpv.item()


best_clip = minimize(best_winsor, x0=0.06, method="nelder-mead")


# %%
y = y.apply(stats.mstats.winsorize, limits=[best_clip.x, best_clip.x])


# %%
# %%

exclude = []
for i in range(50):  # 29):
    X_, y_, P = (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        pd.DataFrame(renamed_distances.loc[F.index, F.index].to_numpy()),
    )

    # Winsorize
    average_degree = 2.9832619059417067

    distribution = stats.poisson(average_degree)

    D = (
        P.loc[X_.index, X_.index]
        .replace(0, np.nan)
        .subtract(1)
        .apply(distribution.pmf)
        .fillna(0)
    )

    try:
        X_ = X_.drop(exclude)
        y_ = y_.drop(exclude)
        # G = G.drop(exclude).drop(columns=exclude)
        D = D.drop(exclude).drop(columns=exclude)
    except:
        print("skipped")
        pass

    model = OLS(y_, X_.assign(alpha=1))
    results = model.fit()
    results.summary()

    tests = sms.jarque_bera(results.resid)
    print(tests)
    if results.get_influence().influence.max() < 0.003:
        break

    # Prob(JB):	0.0825

    excluder = (
        pd.Series(results.get_influence().influence, index=y_.index)
        .nlargest(1)
        .index[0]
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
pd.Series(results.get_influence().influence, index=y_.index).nlargest(5)

# %%
# G = D
G = D.apply(lambda x: x / np.sum(x), 1)
w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())

# %%
X, y, r = X_, y_, r.iloc[y_.index]

# %%
# %%
ols = ps.model.spreg.OLS(
    y=y.to_numpy(),
    x=X.to_numpy(),
    spat_diag=True,
    moran=True,
    name_x=X.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(ols.summary)

# %%
ols_regime = ps.model.spreg.OLS_Regimes(
    y=y.to_numpy(),
    x=X.to_numpy(),
    regimes=r.to_numpy(),
    spat_diag=True,
    moran=True,
    name_x=X.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(ols_regime.summary)

# %%
for c, d in X.items():
    print(OLS(d, (G @ d).rename(c).to_frame().assign(alpha=1)).fit().summary())

X_unsaturated = X.drop(columns=["market_capitalization", "profit_margin"])


# %%
sar = ps.model.spreg.GM_Lag(
    y=y.to_numpy(),
    x=X_unsaturated.to_numpy(),
    spat_diag=True,
    w_lags=2,
    name_x=X_unsaturated.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(sar.summary)

# %%
sar = ps.model.spreg.GM_Lag_Regimes(
    y=y.to_numpy(),
    x=X_unsaturated.drop(columns=["price_to_earnings"]).to_numpy(),
    regimes=r.astype(int).astype(str).apply(lambda s: "t=" + s + "_").to_numpy(),
    name_y="r_i",
    name_regimes="event_day",
    spat_diag=True,
    w_lags=2,
    name_x=X_unsaturated.drop(columns=["price_to_earnings"]).columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(sar.summary)

# %%
ser = ps.model.spreg.GM_Error_Het(
    y=y.to_numpy(),
    x=X_unsaturated.to_numpy(),
    step1c=True,
    name_x=X_unsaturated.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(ser.summary)

# %%
ser_regimes = ps.model.spreg.GM_Error_Het_Regimes(
    y=y.to_numpy(),
    x=X_unsaturated.drop(columns=["price_to_earnings"]).to_numpy(),
    regimes=r.astype(int).astype(str).apply(lambda s: "t=" + s + "_").to_numpy(),
    name_y="r_i",
    name_regimes="event_day",
    step1c=True,
    name_x=X_unsaturated.drop(columns=["price_to_earnings"]).columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(ser_regimes.summary)

# %%
XGX = pd.concat(
    [X_unsaturated, (G @ X_unsaturated).rename(columns=lambda s: s + "_exogenous"),],
    axis=1,
)
sdm = ps.model.spreg.ML_Lag(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    spat_diag=True,
    name_x=XGX.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(sdm.summary)

# %%
sdm = ps.model.spreg.ML_Lag_Regimes(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    regimes=r.astype(int).astype(str).apply(lambda s: "t=" + s + "_").to_numpy(),
    name_y="r_i",
    name_regimes="event_day",
    spat_diag=True,
    name_x=XGX.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(sdm.summary)

# %%
slx_regimes = ps.model.spreg.OLS_Regimes(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    regimes=r.astype(int).astype(str).apply(lambda s: "t=" + s + "_").to_numpy(),
    name_y="r_i",
    name_regimes="event_day",
    spat_diag=True,
    moran=True,
    name_x=XGX.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(slx_regimes.summary)


# %%
kp = ps.model.spreg.GM_Combo_Het(
    y=y.to_numpy(),
    x=X.to_numpy(),
    step1c=False,
    w_lags=2,
    name_x=X.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(kp.summary)

# %%
kp_regimes = ps.model.spreg.GM_Combo_Het_Regimes(
    y=y.to_numpy(),
    x=X_unsaturated.drop(columns=["price_to_earnings"]).to_numpy(),
    regimes=r.astype(int).astype(str).apply(lambda s: "t=" + s + "_").to_numpy(),
    name_y="r_i",
    w_lags=2,
    name_regimes="event_day",
    step1c=True,
    name_x=X_unsaturated.drop(columns=["price_to_earnings"]).columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(kp_regimes.summary)


# %%
H = (
    G.replace(0, np.nan)
    .apply(lambda x: x == x.max(), axis=1)
    .apply(lambda x: x / x.sum(), axis=1)
    .fillna(0)
)

XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exogenous"),], axis=1,)
sdem_nn_regimes = ps.model.spreg.GM_Error_Het_Regimes(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    regimes=r.astype(int).astype(str).apply(lambda s: "t=" + s + "_").to_numpy(),
    name_y="r_i",
    name_regimes="event_day",
    step1c=True,
    name_x=XGX.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(sdem_nn_regimes.summary)

# %%
sdm_nn_regimes = ps.model.spreg.GM_Lag_Regimes(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    regimes=r.astype(int).astype(str).apply(lambda s: "t=" + s + "_").to_numpy(),
    name_y="r_i",
    name_regimes="event_day",
    spat_diag=True,
    name_x=XGX.columns.tolist(),
    w=ps.lib.weights.full2W(G.to_numpy()),
)

print(sdm_nn_regimes.summary)


# %%

kp_nn = ps.model.spreg.GM_Combo_Het(
    y=y.to_numpy(),
    x=X.to_numpy(),
    step1c=True,
    w_lags=2,
    name_x=X.columns.tolist(),
    w=ps.lib.weights.full2W(H.to_numpy()),
)

print(kp_nn.summary)


# %%
kp_regimes = ps.model.spreg.GM_Combo_Het_Regimes(
    y=y.to_numpy(),
    x=X.drop(
        columns=["profit_margin", "price_to_earnings", "market_capitalization"]
    ).to_numpy(),
    step1c=True,
    regimes=r.astype(int).astype(str).apply(lambda s: "t=" + s + "_").to_numpy(),
    name_y="r_i",
    name_regimes="event_day",
    w_lags=2,
    name_x=X.drop(
        columns=["profit_margin", "price_to_earnings", "market_capitalization"]
    ).columns.tolist(),
    w=ps.lib.weights.full2W(H.to_numpy()),
)

print(kp_regimes.summary)

# %%
sms.jarque_bera(kp_regimes.e_filtered)


# %%
kp_nn_regimes = ps.model.spreg.GM_Combo_Het_Regimes(
    y=y.to_numpy(),
    x=X.drop(columns=["market_capitalization"]).to_numpy(),
    regimes=r.to_numpy(),
    step1c=True,
    w_lags=2,
    name_x=X.drop(columns=["market_capitalization"]).columns.tolist(),
    w=ps.lib.weights.full2W(H.to_numpy()),
)

print(kp_nn_regimes.summary)
# CHOW TEST H0 all regimes sames
# REGIMES DIAGNOSTICS - CHOW TEST
#                  VARIABLE        DF        VALUE           PROB
#                  CONSTANT         2          12.396           0.0020
#     market_capitalization         2           7.013           0.0300
#         price_to_earnings         2           2.658           0.2647
#         price_to_research         2          41.856           0.0000
#             profit_margin         2          14.707           0.0006
#                        rm         2           4.666           0.0970

# %%
J = G.where(lambda x: x == 0, 1).apply(lambda x: x / x.sum(), axis=1)

kp_nn = ps.model.spreg.GM_Combo_Het(
    y=y.to_numpy(),
    x=X.to_numpy(),
    step1c=False,
    w_lags=2,
    name_x=X.columns.tolist(),
    w=ps.lib.weights.full2W(J.to_numpy()),
)

print(kp_nn.summary)
