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
        .subtract(1)
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

# ps.model.spreg.summary_output
results.summary_output()


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


def hyperparam(lam=5):
    distribution = stats.poisson(lam)

    # k, mu = x
    # distribution = stats.poisson(k, mu)
    D = (
        renamed_distances.loc[features.index, features.index]
        .replace(0, np.nan)
        .subtract(1)
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
    Xt = pd.concat([Gy, X, GX], axis=1)  # .assign(alpha=1)
    IV = (G @ G @ X).rename(columns=lambda s: s + "_instruments")
    H = pd.concat([X, GX, IV], axis=1)
    model = IV2SLS(y, Xt, H).fit()
    model.summary()

    # step 2
    beta = model.params.endogenous
    EI_GGy = (
        G
        @ pd.DataFrame(np.linalg.inv(I - beta * G), index=G.index, columns=G.columns)
        @ ((pd.concat([X, GX], axis=1) @ model.params.drop("endogenous")))
    )
    Zh = pd.concat([EI_GGy, Xt, IV], axis=1)
    model = IV2SLS(y, Xt.assign(alpha=1), Zh.assign(alpha=1))
    results = model.fit()

    return results.resid.apply(stats.norm.logpdf).apply(np.negative).sum(), results, Xt


best_param = minimize(
    lambda x: hyperparam(x)[0], x0=samples.mean(), method="Nelder-Mead", tol=1e-6,
)

r_squared_adj, results, Xt = hyperparam(best_param.x)

# ps.model.spreg.summary_output
results.summary()


# %%
w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())
Xe = pd.concat([X], axis=1)
model = ps.model.spreg.GM_Combo_Het(
    step1c=True,
    y=(y).to_numpy(),
    x=Xe.to_numpy(),
    w_lags=2,
    w=w,
    name_x=Xe.columns.tolist(),
)
print(model.summary)


S = np.linalg.inv(I - model.betas[-1][0] * G)
effects = pd.DataFrame(
    model.betas[1:-2], index=features.columns[:-2], columns=["Coefficints"]
)
(
    effects.assign(
        direct=lambda df: df.Coefficints.apply(lambda s: np.trace(S * s) / model.n)
    )
    .assign(
        total=lambda df: df.Coefficints.apply(lambda s: (S * s).sum().sum() / model.n)
    )
    .assign(indirect=lambda df: df.total - df.direct)
)

# %%
IV2SLS(
    y, pd.concat([Gy, X], axis=1).assign(alpha=1), pd.concat([Gy, X, GX], axis=1)
).fit().summary()

# %%
import pysal as ps
import numpy as np
from numpy import linalg as la


class GM_Combo_Het(ps.model.spreg.GM_Combo_Het):
    def __init__(
        self,
        y,
        x,
        yend=None,
        q=None,
        w=None,
        w_lags=1,
        lag_q=True,
        vm=False,
        name_y=None,
        name_x=None,
        name_yend=None,
        name_q=None,
        name_w=None,
        name_ds=None,
    ):

        n = ps.model.spreg.USER.check_arrays(y, x, yend, q)
        ps.model.spreg.USER.check_y(y, n)
        ps.model.spreg.USER.check_weights(w, y, w_required=True)
        yend2, q2 = ps.model.spreg.utils.set_endog(y, x, w, yend, q, w_lags, lag_q)
        q2 = q
        print(q2.shape)
        x_constant = ps.model.spreg.USER.check_constant(x)
        ps.model.spreg.GM_Combo_Het.__bases__[0].__init__(
            self,
            y=y,
            x=x_constant,
            w=w.sparse,
            yend=yend2,
            q=q2,
            w_lags=w_lags,
            lag_q=lag_q,
        )
        self.rho = self.betas[-2]
        self.predy_e, self.e_pred, warn = ps.model.spreg.utils.sp_att(
            w, self.y, self.predy, yend2[:, -1].reshape(self.n, 1), self.rho
        )
        ps.model.spreg.utils.set_warn(self, warn)
        self.title = "SPATIALLY WEIGHTED TWO STAGE LEAST SQUARES"
        self.name_ds = ps.model.spreg.USER.set_name_ds(name_ds)
        self.name_y = ps.model.spreg.USER.set_name_y(name_y)
        self.name_x = ps.model.spreg.USER.set_name_x(name_x, x)
        self.name_yend = ps.model.spreg.USER.set_name_yend(name_yend, yend)
        self.name_yend.append(ps.model.spreg.USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append("lambda")
        self.name_q = ps.model.spreg.USER.set_name_q(name_q, q)
        self.name_q.extend(
            ps.model.spreg.USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q)
        )
        self.name_h = ps.model.spreg.USER.set_name_h(self.name_x, self.name_q)
        self.name_w = ps.model.spreg.USER.set_name_w(name_w, w)
        ps.model.spreg.summary_output.GM_Combo_Het(reg=self, w=w, vm=vm)


# %%
def maxer(rho):
    try:
        print(rho)
        XGX = pd.concat([(I - G.multiply(rho.item())) @ X], axis=1)
        GGXGX = G @ G @ pd.concat([X, GX], axis=1)
        w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())
        model_ = GM_Combo(
            y=y.to_numpy(),
            x=XGX.to_numpy(),
            w_lags=1,
            w=w,
            q=GGXGX.to_numpy(),
            name_x=XGX.columns.tolist(),
        )
        model_.summary

        return -model_.pr2_e, model_
    except:
        return 0, None


best_param = minimize(lambda x: maxer(x)[0], method="nelder-mead", x0=0.1, tol=1e-6)

r_squared_adj, model = maxer(best_param.x)

# ps.model.spreg.summary_output
print(model.summary)


# %%
XGX_ = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exog")], axis=1)
XGX = (I - G) @ XGX_
GGXGX = (I - G) @ G @ G @ pd.concat([X, GX], axis=1)
w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())
model_ = GM_Combo_Het(
    y=(y).to_numpy(),
    x=XGX.to_numpy(),
    w_lags=1,
    w=w,
    q=GGXGX.to_numpy(),
    name_x=XGX.columns.tolist(),
)

print(model_.summary)

# %%
beta = model_.betas[-2]
gamma_sigma = model_.betas[1 : (len(XGX.columns) + 1)]

G_ = G.to_numpy()
IG_G_y = (
    G_ @ np.linalg.inv(I - beta.item() * G_) @ ((I - G_) @ XGX_ @ gamma_sigma)
).rename(lambda s: "endogenous")

model_ = GM_Combo_Het(
    y=(y).to_numpy(),
    x=XGX.to_numpy(),
    w_lags=1,
    w=w,
    q=IG_G_y.to_numpy(),
    name_x=XGX.columns.tolist(),
)

print(model_.summary)

# %%
print(sms.jarque_bera(ml_sdem_fixed.e_filtered))

# normal
