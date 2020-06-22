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


# %%
samples = renamed_distances.replace(0, np.nan).subtract(1).melt().value.dropna()

X, y = features.drop(columns=["returns", "alpha"]), features.loc[:, ["returns"]]
exclude = [
    "DNN",
    "GLBS",
    "EXEL",
    "SHIP",
    "GLNG",
    "TAT",
    "CCF",
    "HNW",
    "ESEA",
    "HES",
    "CVGI",
    "MXL",
    "MASI",
    "CL",
    "TWIN",
    "SBLK",
    "MRVL",
    "TXN",
    "FGEN",
    "PFPT",
    "ROST",
    "LVS",
    "QCOM",
    "MSFT",
    "INTC",
    "JNJ",
    "COST",
    "AMGN",
    "ARCC",
]

X = X.drop(exclude)
y = y.drop(exclude)
average_degree = 2.9832619059417067
distribution = stats.poisson(average_degree)  # samples.mean())

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


# %%
# SDM : unrestricted, fixed \lambda
XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exog")], axis=1)

sdm_saturated_restricted = ps.model.spreg.ML_Lag(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    w=w,
    name_x=XGX.columns.tolist(),
    spat_diag=True,
    method="ORD",
)
print(sdm_saturated_restricted.summary)

# Spatial Pseudo R-squared:  0.1977

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0166323       0.0321272       0.5177026       0.6046658
#                   rm      -6.9318562       2.2394697      -3.0953115       0.0019661
#    price_to_earnings      -0.0058256       0.0026851      -2.1696131       0.0300362
# market_capitalization      -0.0000000       0.0000000      -0.5352747       0.5924599
#        profit_margin       0.0100430       0.0141086       0.7118384       0.4765649
#    price_to_research       0.0000049       0.0000036       1.3528516       0.1761030
#              rm_exog     -17.2746194       9.9612154      -1.7341879       0.0828847
# price_to_earnings_exog       0.0142663       0.0327990       0.4349612       0.6635906
# market_capitalization_exog       0.0000000       0.0000000       1.8744611       0.0608669
#   profit_margin_exog      -0.0818302       0.1606903      -0.5092415       0.6105830
# price_to_research_exog       0.0000057       0.0001059       0.0539087       0.9570079
#            W_dep_var      -0.9685720       0.4240386      -2.2841602       0.0223621

# %%
# test normality
tests = sms.jarque_bera(sdm_saturated_restricted.e_pred)
print(tests)

# %%
OLS(
    sdm_saturated_restricted.e_pred, G @ sdm_saturated_restricted.e_pred
).fit().summary()


# SDM : unrestricted, varying \lambda
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
            method="ORD",
        )

        return -model.pr2_e, model, G

    except:
        print("error")
        return 0, None, None


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([average_degree]), tol=1e-6,)

pr2_e, sdm_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(sdm_unresticted_varying.summary)

# \lambda: 3.06158744

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0161220       0.0324023       0.4975577       0.6187958
#                   rm      -6.9636693       2.2350744      -3.1156320       0.0018355
#    price_to_earnings      -0.0058182       0.0026815      -2.1697665       0.0300245
# market_capitalization      -0.0000000       0.0000000      -0.5174786       0.6048221
#        profit_margin       0.0100912       0.0140856       0.7164215       0.4737311
#    price_to_research       0.0000048       0.0000036       1.3373167       0.1811193
#              rm_exog     -17.6096813      10.0388484      -1.7541535       0.0794042
# price_to_earnings_exog       0.0133833       0.0327424       0.4087441       0.6827275
# market_capitalization_exog       0.0000000       0.0000000       1.8850697       0.0594204
#   profit_margin_exog      -0.0771270       0.1609443      -0.4792153       0.6317854
# price_to_research_exog       0.0000016       0.0001075       0.0151220       0.9879349
#            W_dep_var      -0.9934981       0.4299800      -2.3105679       0.0208567

# can reject \rho = 0 with p-value = 0.02


# %%
# test normality
tests = sms.jarque_bera(sdm_unresticted_varying.e_pred)
print(tests)

# reject normality of u array([0.01773414])

# %%
# test of the likelihood ratio on the common factor hypothesis (θ = −ρβ )
n = sdm_unresticted_varying.n  # (scalar) number of observations

# constained
constrained = sdm_unresticted_varying.betas.copy()
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
lam_lr = np.exp(constrained_ll_result) / np.exp(sdm_unresticted_varying.logll)
#  0.015151

chi = -2 * (constrained_ll_result - sdm_unresticted_varying.logll)
stats.chi.sf(chi, 5)
# can reject (θ = −ρβ ) 9.2481578e-14

# %%
# est of the likelihood ratio on θ = 0
n = sdm_unresticted_varying.n  # (scalar) number of observations

# constained
constrained_theta = sdm_unresticted_varying.betas.copy()
constrained_theta[6:-1, :] = 0
utu_theta = (
    pd.concat([y.pow(0), XGX, G @ y], axis=1)
    .dot(constrained_theta)
    .subtract(y.to_numpy())
    .pow(2)
    .sum()
)
constrained_theta_ll_result = -0.5 * (
    n * (np.log(2 * np.pi)) + n * np.log(utu_theta / n) + (utu_theta / (utu_theta / n))
)

# lambda_lr
lam_lr_theta = np.exp(constrained_theta_ll_result) / np.exp(
    sdm_unresticted_varying.logll
)
lam_lr_theta
#  4.575699e-16

chi_theta = -2 * (constrained_theta_ll_result - sdm_unresticted_varying.logll)
stats.chi.sf(chi_theta, 5)

# reject H0: \all θ = 0

# %%
# est of the likelihood ratio on θ = 0
n = sdm_unresticted_varying.n  # (scalar) number of observations

# constained
constrained_rho = sdm_unresticted_varying.betas.copy()
constrained_rho[-1, :] = 0
utu_rho = (
    pd.concat([y.pow(0), XGX, G @ y], axis=1)
    .dot(constrained_rho)
    .subtract(y.to_numpy())
    .pow(2)
    .sum()
)
constrained_rho_ll_result = -0.5 * (
    n * (np.log(2 * np.pi)) + n * np.log(utu_rho / n) + (utu_rho / (utu_rho / n))
)

# lambda_lr
lam_lr_rho = np.exp(constrained_rho_ll_result) / np.exp(sdm_unresticted_varying.logll)
lam_lr_rho
#  0.014855

chi_rho = -2 * (constrained_rho_ll_result - sdm_unresticted_varying.logll)
stats.chi.sf(chi_rho, 1)

# reject H0: \all θ = 0 at 0

# %%
# \lambda = 0
OLS(sdm_unresticted_varying.e_pred, G @ sdm_unresticted_varying.e_pred).fit().summary()
# p-value = 0.003


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

# Spatial Pseudo R-squared:  0.1990
# W_dep_var      -0.7799806       0.9346993      -0.8344722       0.4040149

# TEST                           MI/DF       VALUE           PROB
# Anselin-Kelejian Test             1           0.338          0.5611
#  null hypothesis that there is no spatial autocorrelation in the residuals of your model.
# Therefore u = \lambda u + \epsilon
# Testing For Spatial Error Autocorrelation In The Presence Of Endogenous Regressors ", by Anselin and Kelejian.


# %%
# GM SDM : unrestricted, varying \lambda
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
        model = GM_Lag(
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

pr2_e, gm_sdm_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(gm_sdm_unresticted_varying.summary)

# DIAGNOSTICS FOR SPATIAL DEPENDENCE
# TEST                           MI/DF       VALUE           PROB
# Anselin-Kelejian Test             1           0.915          0.3389

#  null hypothesis that there is no spatial autocorrelation in the residuals of your model.
# Therefore u = \lambda u + \epsilon
# Testing For Spatial Error Autocorrelation In The Presence Of Endogenous Regressors ", by Anselin and Kelejian.


# %%
# %%
# SLX : unrestricted, fixed \lambda
XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_exog")], axis=1)
slx_unresticted_fixed = ps.model.spreg.OLS(
    y.to_numpy(),
    XGX.to_numpy(),
    name_x=XGX.columns.tolist(),
    w=w,
    spat_diag=True,
    moran=True,
)
print(slx_unresticted_fixed.summary)

# %%
# SLX : unrestricted, varying \lambda
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


best_param = minimize(lambda x: opt_lam(x)[0], x0=np.array([average_degree]), tol=1e-6,)

pr2_e, slx_unresticted_varying, G_opt = opt_lam(best_param.x)
print(best_param.x)
print(slx_unresticted_varying.summary)


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
XGX_opt = pd.concat([X, (G_opt @ X).rename(columns=lambda s: s + "_exog")], axis=1)

slx_unresticted_restults = OLS(y, XGX_opt.assign(alpha=1)).fit()

slx_unresticted_restults.summary()

slx_unresticted_restults.f_test(
    "rm_exog = 0, price_to_earnings_exog = 0, market_capitalization_exog = 0, profit_margin_exog=0, price_to_research_exog=0"
)

# cannot reject hypothesis all zero

slx_resticted_restults = ps.model.spreg.OLS(
    y=y.drop(["GES", "ERA", "QCOM", "GLMD"]).to_numpy(),
    x=XGX.loc[:, ["rm", "rm_exog"]].drop(["GES", "ERA", "QCOM", "GLMD"]).to_numpy(),
    w=ps.lib.weights.full2W(
        G.drop(["GES", "ERA", "QCOM", "GLMD"])
        .drop(columns=["GES", "ERA", "QCOM", "GLMD"])
        .to_numpy()
    ),
    name_x=XGX.loc[:, ["rm", "rm_exog"]].columns.tolist(),
    spat_diag=True,
)
print(slx_resticted_restults.summary)


#############################

# %%
ml_sdem_fixed = ps.model.spreg.ML_Error(
    y=y.to_numpy(), x=XGX.to_numpy(), w=w, name_x=XGX.columns.tolist()
)
print(ml_sdem_fixed.summary)
# lambda significant

# %%
print(sms.jarque_bera(ml_sdem_fixed.e_filtered))
# not normal

# %%
gm_sdem_fixed = ps.model.spreg.GM_Error(
    y=y.to_numpy(), x=XGX.to_numpy(), w=w, name_x=XGX.columns.tolist()
)
print(gm_sdem_fixed.summary)

# %%
gm_sdem_fixed = ps.model.spreg.GM_Error_Het(
    y=y.to_numpy(), x=XGX.to_numpy(), w=w, name_x=XGX.columns.tolist()
)
print(gm_sdem_fixed.summary)

# # lambda insignificant

# %%
ml_lag_fixed = ps.model.spreg.ML_Error(
    y=y.to_numpy(), x=X.to_numpy(), w=w, name_x=X.columns.tolist()
)
print(ml_lag_fixed.summary)
# lambda significant at 0.04


# %%
print(sms.jarque_bera(ml_lag_fixed.e_filtered))
# normal

# %%
# gm_lag_fixed = ps.model.spreg.GM_Error(
#     y=y.to_numpy(), x=X.to_numpy(), w=w, name_x=X.columns.tolist()
# )
# print(gm_lag_fixed.summary)

# # %%
# # %%
# gm_sdem_fixed = ps.model.spreg.GM_Error_Het(
#     y=y.to_numpy(), x=X.to_numpy(), w=w, name_x=X.columns.tolist()
# )
# print(gm_sdem_fixed.summary)

r1 = np.square(ml_lag_fixed.e_filtered).sum()
r2 = np.square(ml_sdem_fixed.e_filtered).sum()

k1 = ml_lag_fixed.betas.shape[0]
k2 = ml_sdem_fixed.betas.shape[0]

n = ml_sdem_fixed.e_filtered.shape[0]

dfn = k2 - k1
dfd = n - k2
F = ((r1 - r2) / (k2 - k1)) / (r2 / (n - k2))

stats.f(dfn=dfn, dfd=dfd).sf(F)
# p-value = 0.1605726555082848


#
lambda_lr = -2 * (ml_lag_fixed.logll - ml_sdem_fixed.logll)
lambda_lr
# 8.242926298307339

stats.chi(5).sf(lambda_lr)
# 2.7400304247748863e-13

# %%
# %%
XGX_opt = pd.concat([X, (G_opt @ X).rename(columns=lambda s: s + "_exogenous")], axis=1)
w_opt = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())


ml_sdem_varying = ps.model.spreg.ML_Error(
    y=y.to_numpy(), x=XGX_opt.to_numpy(), w=w_opt, name_x=XGX_opt.columns.tolist()
)
print(ml_sdem_varying.summary)
# lambda significant 0.0132470

# %%
print(sms.jarque_bera(ml_sdem_varying.e_filtered))
# Normal at 0.064 %


# %%
ml_lag_varying = ps.model.spreg.ML_Error(
    y=y.to_numpy(), x=X.to_numpy(), w=w_opt, name_x=X.columns.tolist()
)
print(ml_lag_varying.summary)
# lambda significant at 0.0454188


# %%
print(sms.jarque_bera(ml_lag_varying.e_filtered))
# normal p-value = 0.1976079


r1 = np.square(ml_lag_varying.e_filtered).sum()
r2 = np.square(ml_sdem_varying.e_filtered).sum()

k1 = ml_lag_varying.betas.shape[0]
k2 = ml_sdem_varying.betas.shape[0]

n = ml_sdem_varying.e_filtered.shape[0]

dfn = k2 - k1
dfd = n - k2
F = ((r1 - r2) / (k2 - k1)) / (r2 / (n - k2))

stats.f(dfn=dfn, dfd=dfd).sf(F)
# p-value = 0.04999816226741284

# %%
lambda_lr = -2 * (ml_lag_varying.logll - ml_sdem_varying.logll)
lambda_lr
# 8.242926298307339

# %%
stats.chi(5).sf(lambda_lr)
# 0

# %%
alpha_lr = np.exp(ml_lag_varying.logll) / np.exp(ml_sdem_varying.logll)
alpha_lr
# 0.00334157506545991
