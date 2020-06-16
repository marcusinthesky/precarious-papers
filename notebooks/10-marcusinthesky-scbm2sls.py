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

# b, loc, scale = stats.pareto.fit(samples)
# distribution = stats.pareto(b, loc, scale)


def hyperparam(lam=5):
    distribution = stats.poisson(lam)

    # k, mu = x
    # distribution = stats.poisson(k, mu)
    D = (
        renamed_distances.loc[features.index, features.index]
        .replace(0, np.nan)
        .apply(distribution.pmf)
        .fillna(0)
    )

    # G = D
    G = D.apply(lambda x: x / np.sum(x), 1)

    I = np.identity(G.shape[1])

    # bramoulle
    # step 1
    X = features.drop(columns=["alpha", "returns"])
    y = features.loc[:, ["returns"]]
    GX = (G @ X).rename(columns=lambda s: s + "_exogenous")
    Gy = (G @ y).rename(columns=lambda s: "endogenous")
    Xt = pd.concat(
        [(I - G) @ Gy, (I - G) @ X, (I - G) @ GX], axis=1
    )  # .assign(alpha=1)
    IV = (G @ G @ X).rename(columns=lambda s: s + "_instruments")
    H = pd.concat([(I - G) @ X, (I - G) @ GX, (I - G) @ IV], axis=1)
    model = IV2SLS(y, Xt, H).fit()
    model.summary()

    # step 2
    beta = model.params.endogenous
    EI_GGy = (
        G
        @ pd.DataFrame(np.linalg.inv(I - beta * G), index=G.index, columns=G.columns)
        @ ((I - G) @ (pd.concat([X, GX], axis=1) @ model.params.drop("endogenous")))
    )
    Zh = pd.concat([EI_GGy, Xt, IV], axis=1)
    model = IV2SLS(y, Xt.assign(alpha=1), Zh.assign(alpha=1))
    results = model.fit()

    return results.resid.apply(stats.norm.logpdf).apply(np.negative).sum(), results, Xt


best_param = minimize(
    lambda x: hyperparam(x)[0], x0=samples.mean(), method="Nelder-Mead", tol=1e-6,
)

r_squared_adj, results, Xt = hyperparam(best_param.x)

# summary
results.summary()


# %%
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
            for k, v in Xt.rename(
                columns=lambda s: " ".join(s.split("_")).title()
                if s.split("_")[-1] != "exogenous"
                else " ".join(s.split("_")).title() + " Social Effect"
            ).items()
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


# %%
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


# %%
# SLX
Xt = X.drop(columns=["market_capitalization", "price_to_research", "price_to_earnings"])
model = OLS(
    y, Xt.assign(alpha=1, profit_margin_exogenous=G @ X.profit_margin.to_frame())
).fit()
model.summary()

# %%
# SAR
Xt = X.drop(columns=["market_capitalization", "price_to_research", "price_to_earnings"])
model = OLS(y, Xt.assign(alpha=1, endogenous=G @ y)).fit()
model.summary()

# %%
# G2SLS
# y = y - y.mean()
S = pd.concat([(I - G) @ X, (I - G) @ G @ X, (I - G) @ G @ G @ X], axis=1)
P = S @ pd.DataFrame(np.linalg.inv(S.T @ S), index=S.columns, columns=S.columns) @ S.T
P = P.apply(lambda x: x / np.sum(x), 1)

Xt = pd.concat(
    [
        ((I - G) @ G @ y).rename(columns={"returns": "endogenous"}),
        (I - G) @ X,
        ((I - G) @ G @ X).rename(columns=lambda x: x + "_exogenous"),
    ],
    axis=1,
)

# %%
Th2sls = (
    pd.DataFrame(np.linalg.inv(Xt.T @ P @ Xt), columns=Xt.columns, index=Xt.columns)
    @ Xt.T
    @ P
    @ y
).rename(columns={"returns": "Endogenous"})
Zh = pd.concat(
    [
        (I - G) @ G @ (Xt @ Th2sls),
        (I - G) @ X,
        ((I - G) @ G @ X).rename(columns=lambda s: s + "_exogenous"),
    ],
    axis=1,
)
Thlle = (
    pd.DataFrame(np.linalg.inv(Zh.T @ Xt), columns=Zh.columns, index=Zh.columns)
    @ Zh.T
    @ y
).rename(columns={"returns": "Theta_lle"})
d = y.subtract((Zh @ Thlle).to_numpy()).pow(2)
D = pd.DataFrame(np.diag(d.to_numpy().flatten()), columns=Zh.index, index=Zh.index)
Vhlle = (
    pd.DataFrame(np.linalg.inv(Zh.T @ Xt), index=Zh.columns, columns=Zh.columns)
    @ Zh.T
    @ D
    @ Zh
    @ pd.DataFrame(np.linalg.inv(Zh.T @ Xt), index=Zh.columns, columns=Zh.columns)
)
SE_Vhlle = pd.DataFrame(np.diag(Vhlle), index=Vhlle.columns, columns=["SE"]).pow(0.5)
T_lle = (Thlle / SE_Vhlle.to_numpy()).rename(columns={"Theta_lle": "T_lle"})
p_lle = (
    T_lle.apply(np.abs)
    .apply(stats.t(df=Zh.shape[0] - Zh.shape[1]).sf)
    .rename(columns={"T_lle": "p_lle"})
)
p_lle

# %%
# A = X
X = features.drop(columns=["alpha", "returns"])
GX = (G @ X).rename(columns=lambda s: s + "_exog")
S = pd.concat([(I - G) @ X, (I - G) @ G @ X, (I - G) @ G @ GX], axis=1)
Xt = pd.concat(
    [
        ((I - G) @ G @ y).rename(columns={"returns": "endogenous"}),
        (I - G) @ X,
        (I - G) @ GX,
    ],
    axis=1,
)

theta_sls = IV2SLS(y, Xt.assign(alpha=1), S.assign(alpha=1)).fit()

beta = theta_sls.params["endogenous"]
gamma = theta_sls.params[X.columns]
sigma = theta_sls.params[GX.columns]

E = (
    G
    @ pd.DataFrame((I - beta * G).pipe(np.linalg.inv), index=G.index, columns=G.columns)
    @ ((I - G) @ (X @ gamma + GX @ sigma))
)

Zh = pd.concat(
    [
        pd.DataFrame(E, index=Xt.index, columns=["endogenous"]),
        (I - G) @ X,
        (I - G) @ GX,
    ],
    axis=1,
)
# .drop(columns=['price_to_earnings', 'market_capitalization', 'market_capitalization_exogenous',  'price_to_earnings_exogenous'])
theta_lle = IV2SLS(y, Xt.assign(alpha=1), Zh.assign(alpha=1)).fit()
theta_lle.summary()

ZhXt_inv_Zh = (
    pd.DataFrame(np.linalg.inv(Zh.T @ Xt), index=Zh.columns, columns=Xt.columns) @ Zh.T
)
Theta_hat_lle = ZhXt_inv_Zh @ y

d = y.subtract((Zh @ Theta_hat_lle).to_numpy()).pow(2)
D = pd.DataFrame(np.diag(d.to_numpy().flatten()), columns=Zh.index, index=Zh.index)

V_hat_lle = ZhXt_inv_Zh @ D @ ZhXt_inv_Zh.T

SE_Vhlle = pd.DataFrame(
    np.diag(V_hat_lle), index=V_hat_lle.columns, columns=["SE"]
).pow(0.5)
T_lle = (Theta_hat_lle / SE_Vhlle.to_numpy()).rename(columns={"Theta_lle": "T_lle"})
p_lle = (
    T_lle.apply(np.abs)
    .apply(stats.t(df=Zh.shape[0] - Zh.shape[1]).sf)
    .rename(columns={"T_lle": "p_lle"})
)
p_lle


# %%
w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())

XGX = pd.concat([X, GX], axis=1)
kelejian_prucha = ps.model.spreg.GM_Error(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    w=w,
    name_x=XGX.columns.tolist(),
    name_y=y.columns[0],
    name_ds="s-CBM",
    name_w="Row-normalized Pareto-Kernel Degree",
)
print(kelejian_prucha.summary)


# %%
H = E.to_frame()

kelejian_prucha_endog = ps.model.spreg.GM_Endog_Error(
    y=y.to_numpy(),
    x=Xt.drop(columns=["endogenous"]).to_numpy(),
    yend=Xt.loc[:, ["endogenous"]].to_numpy(),
    q=H.to_numpy(),
    w=w,
    name_x=Xt.columns.tolist()[1:],
    name_y=y.columns[0],
    name_yend=["endogenous"],
    name_ds="s-CBM",
    name_w="Row-normalized Pareto-Kernel Degree",
)
print(kelejian_prucha_endog.summary)


# %%
kelejian_prucha_endog = ps.model.spreg.GM_Endog_Error(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    yend=(G @ y).to_numpy(),
    q=(D @ np.ones_like(y)).to_numpy(),
    w=w,
    name_x=XGX.columns.tolist(),
    name_y=y.columns[0],
    name_yend=["endogenous"],
    name_ds="s-CBM",
    name_w="Row-normalized Pareto-Kernel Degree",
)
print(kelejian_prucha_endog.summary)


# %%
kelejian_prucha_endog = ps.model.spreg.GM_Endog_Error(
    y=y.to_numpy(),
    x=Xt.drop(columns=["endogenous", "alpha"]).to_numpy(),
    yend=Xt.loc[:, ["endogenous"]].to_numpy(),
    q=IV.to_numpy(),
    w=w,
    name_x=Xt.drop(columns=["endogenous", "alpha"]).columns.tolist(),
    name_y=y.columns[0],
    name_yend=["endogenous"],
    name_q=IV.columns.tolist(),
    name_ds="s-CBM",
    name_w="Row-normalized Pareto-Kernel Degree",
)
print(kelejian_prucha_endog.summary)


# %%
OLS(
    y,
    Xt.drop(columns=["endogenous"]).loc[
        :, ["alpha", "rm", "centrality", "profit_margin_exog", "profit_margin"]
    ],
).fit().summary()


# %%
X = features.loc[:, ["rm", "profit_margin"]].assign(centrality=D.sum(1))
y = features.loc[:, ["returns"]]
w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())

model = ps.model.spreg.ML_Lag(
    y.to_numpy(), X.to_numpy(), w, name_x=X.columns.tolist(), spat_diag=True
)
print(model.summary)

# %%
model = ps.model.spreg.ML_Error(
    y.to_numpy(), X.to_numpy(), w, name_x=X.columns.tolist(), spat_diag=True
)
print(model.summary)


# %%
model = ps.model.spreg.OLS(
    y.to_numpy(), X.to_numpy(), w, name_x=X.columns.tolist(), spat_diag=True
)
print(model.summary)

# %%
# SLX
GX = (G @ X).rename(columns=lambda s: s + "_exog")
XGX = pd.concat([X, GX], axis=1)
model = ps.model.spreg.OLS(
    y.to_numpy(), XGX.to_numpy(), w, name_x=XGX.columns.tolist(), spat_diag=True
)
print(model.summary)

# %%
# SAR
Gy = (G @ y).rename(columns=lambda s: "endog")
model = ps.model.spreg.OLS(
    y.to_numpy(), Gy.to_numpy(), w, name_x=["Gy"], spat_diag=True
)
print(model.summary)

# %%
# y = Xb +  Wy a  + c + e
model = ps.model.spreg.ML_Lag(
    y.to_numpy(), X.to_numpy(), w=w, name_x=X.columns.tolist(), spat_diag=True
)
print(model.summary)

# %%
# y = Xb +  c + u
# u = lW + e
model = ps.model.spreg.ML_Error(
    y.to_numpy(), X.to_numpy(), w, name_x=X.columns.tolist()
)
print(model.summary)

# %%
model = ps.model.spreg.GM_Lag(
    y.to_numpy(), X.to_numpy(), w=w, name_x=X.columns.tolist(), spat_diag=True
)
print(model.summary)

# %%
model = ps.model.spreg.GM_Error(
    y.to_numpy(), X.to_numpy(), w=w, name_x=X.columns.tolist()
)
print(model.summary)

# %%
model = ps.model.spreg.GM_Error_Het(
    y.to_numpy(), X.to_numpy(), w=w, name_x=X.columns.tolist()
)
print(model.summary)

# %%
model = ps.model.spreg.GM_Error_Hom(
    y.to_numpy(), X.to_numpy(), w=w, name_x=X.columns.tolist()
)
print(model.summary)

# %%
model = ps.model.spreg.GM_Combo(
    y.to_numpy(), X.to_numpy(), w_lags=1, w=w, name_x=X.columns.tolist()
)
print(model.summary)


# %%
# step 1
y = y - y.mean()
Xtt = X
model = ps.model.spreg.GM_Error(
    y.to_numpy(), Xtt.to_numpy(), w=w, name_x=Xtt.columns.tolist()
)
lambda_ = model.betas[-1].item()
# step 2
rho_ = OLS(model.u, G @ model.y).fit().params.item()

# step 3
Z = pd.concat([G @ Xtt, y], axis=1)
Zs = Z - rho_ * G @ Z

ys = y - rho_ * G @ y

I_lambda_G = pd.DataFrame(
    np.linalg.inv(I - lambda_ * G), index=G.index, columns=G.columns
)
H = (I - rho_ * G) @ pd.concat([Xtt, G @ I_lambda_G @ Xtt @ model.betas[1:-1]], axis=1)

IV2SLS(ys, Zs.assign(alpha=1), H.assign(alpha=1)).fit().summary()

# %%
