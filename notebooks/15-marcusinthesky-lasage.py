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
# SDM : unrestricted, fixed \lambda
XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exog")], axis=1)

sdm_saturated_restricted = ps.model.spreg.ML_Lag(
    y=y.to_numpy(), x=XGX.to_numpy(), w=w, name_x=XGX.columns.tolist(), spat_diag=True
)
print(sdm_saturated_restricted.summary)

# Spatial Pseudo R-squared:  0.1171

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0040255       0.0675309       0.0596092       0.9524669
#                   rm      -0.6043740       0.6476418      -0.9331916       0.3507210
#    price_to_earnings       0.0004543       0.0020652       0.2199943       0.8258756
# market_capitalization       0.0000000       0.0000000       1.3522543       0.1762940
#        profit_margin      -0.0397208       0.0124162      -3.1991017       0.0013786 **
#    price_to_research       0.0000034       0.0000044       0.7742937       0.4387571
#              rm_exog     -11.9051932      18.3929835      -0.6472682       0.5174584
# price_to_earnings_exog       0.0137423       0.0288805       0.4758334       0.6341931
# market_capitalization_exog       0.0000000       0.0000000       1.9814154       0.0475447 *
#   profit_margin_exog      -0.0713462       0.1904880      -0.3745442       0.7079995
# price_to_research_exog      -0.0000828       0.0001969      -0.4208524       0.6738629
#            W_dep_var      -0.9182708       0.3998068      -2.2967863       0.0216310 *

# SDM : unrestricted, varying \lambda
def opt_lam(lam):
    try:
        print(lam)
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


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([samples.mean()]), tol=1e-6,)

pr2_e, slx_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(slx_unresticted_varying.summary)

# \lambda: 4.76413852

# Spatial Pseudo R-squared:  0.1177

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT      -0.0052500       0.0680120      -0.0771922       0.9384706
#                   rm      -0.6233628       0.6473428      -0.9629562       0.3355695
#    price_to_earnings       0.0004563       0.0020625       0.2212387       0.8249066
# market_capitalization       0.0000000       0.0000000       1.4397062       0.1499506
#        profit_margin      -0.0390569       0.0124319      -3.1416830       0.0016798
#    price_to_research       0.0000033       0.0000044       0.7479413       0.4544956
#              rm_exog     -11.3505751      18.2830481      -0.6208251       0.5347147
# price_to_earnings_exog       0.0128691       0.0292583       0.4398463       0.6600484
# market_capitalization_exog       0.0000000       0.0000000       2.0945089       0.0362147
#   profit_margin_exog      -0.0455119       0.1942418      -0.2343053       0.8147480
# price_to_research_exog      -0.0000972       0.0001972      -0.4926832       0.6222364
#            W_dep_var      -0.9615484       0.4048675      -2.3749706       0.0175503


# %%
# test of the likelihood ratio on the common factor hypothesis (θ = −ρβ )
n = slx_unresticted_varying.n  # (scalar) number of observations

# constained
constrained = slx_unresticted_varying.betas.copy()
constrained[6:-1, :] = -constrained[-1, :] * constrained[1:6, :]
utu = (
    pd.concat([y.pow(0), XGX, G @ y], axis=1)
    .dot(constrained)
    .subtract(y.to_numpy())
    .pow(2)
    .sum()
)
constrained_ll_result = -0.5 * (
    n * (np.log(2 * np.pi)) + n * np.log(utu / n) + (utu / (utu / n))
)

# lambda_lr
lam_lr = -2 * (constrained_ll_result - slx_unresticted_varying.logll)
stats.chi.sf(lam_lr, 5)

# can reject (θ = −ρβ )

# %%
# GM_SDM : unrestricted, fixed \lambda

import pysal as ps
import numpy as np
from numpy import linalg as la


class BaseGM_Lag(ps.model.spreg.twosls.BaseTSLS):
    def __init__(
        self,
        y,
        x,
        yend=None,
        q=None,
        w=None,
        w_lags=1,
        lag_q=True,
        robust=None,
        gwk=None,
        sig2n_k=False,
    ):

        ps.model.spreg.twosls.BaseTSLS.__init__(
            self, y=y, x=x, yend=yend, q=q, robust=robust, gwk=gwk, sig2n_k=sig2n_k
        )


class GM_Lag(BaseGM_Lag):
    def __init__(
        self,
        y,
        x,
        yend=None,
        q=None,
        w=None,
        w_lags=1,
        lag_q=True,
        robust=None,
        gwk=None,
        sig2n_k=False,
        spat_diag=False,
        vm=False,
        name_y=None,
        name_x=None,
        name_yend=None,
        name_q=None,
        name_w=None,
        name_gwk=None,
        name_ds=None,
    ):

        n = ps.model.spreg.USER.check_arrays(x, yend, q)
        ps.model.spreg.USER.check_y(y, n)
        ps.model.spreg.USER.check_weights(w, y, w_required=True)
        ps.model.spreg.USER.check_robust(robust, gwk)
        yend2, q2 = ps.model.spreg.utils.set_endog(y, x, w, yend, q, w_lags, lag_q)
        q2 = G.to_numpy() @ q2
        x_constant = ps.model.spreg.USER.check_constant(x)
        BaseGM_Lag.__init__(
            self,
            y=y,
            x=x_constant,
            w=w.sparse,
            yend=yend2,
            q=q2,
            w_lags=w_lags,
            robust=robust,
            gwk=gwk,
            lag_q=lag_q,
            sig2n_k=sig2n_k,
        )
        self.rho = self.betas[-1]
        self.predy_e, self.e_pred, warn = ps.model.spreg.utils.sp_att(
            w, self.y, self.predy, yend2[:, -1].reshape(self.n, 1), self.rho
        )
        ps.model.spreg.utils.set_warn(self, warn)
        self.title = "SPATIAL TWO STAGE LEAST SQUARES"
        self.name_ds = ps.model.spreg.USER.set_name_ds(name_ds)
        self.name_y = ps.model.spreg.USER.set_name_y(name_y)
        self.name_x = ps.model.spreg.USER.set_name_x(name_x, x)
        self.name_yend = ps.model.spreg.USER.set_name_yend(name_yend, yend)
        self.name_yend.append(ps.model.spreg.USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_q = ps.model.spreg.USER.set_name_q(name_q, q)
        self.name_q.extend(
            ps.model.spreg.USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q)
        )
        self.name_h = ps.model.spreg.USER.set_name_h(self.name_x, self.name_q)
        self.robust = ps.model.spreg.USER.set_robust(robust)
        self.name_w = ps.model.spreg.USER.set_name_w(name_w, w)
        self.name_gwk = ps.model.spreg.USER.set_name_w(name_gwk, gwk)
        ps.model.spreg.summary_output.GM_Lag(reg=self, w=w, vm=vm, spat_diag=spat_diag)


XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exog")], axis=1)

gm_sdm_saturated_restricted = GM_Lag(
    y=y.to_numpy(), x=XGX.to_numpy(), w=w, name_x=XGX.columns.tolist(), spat_diag=True
)
print(gm_sdm_saturated_restricted.summary)


# SDM : unrestricted, varying \lambda
def opt_lam(lam):
    try:
        print(lam)
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


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([samples.mean()]), tol=1e-6,)

pr2_e, slx_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(slx_unresticted_varying.summary)


# %%
