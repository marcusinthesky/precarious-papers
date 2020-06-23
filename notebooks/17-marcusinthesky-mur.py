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

    distribution = stats.poisson(0.8)  # average_degree)  # samples.mean())

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
    if tests[1] > 0.1:
        break

    # Prob(JB):	0.0825

    # %%
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

#############
y = y.drop(["GES", "ERA"])
X = X.drop(["GES", "ERA"])
G = (
    G.drop(["GES", "ERA"])
    .drop(columns=["GES", "ERA"])
    .apply(lambda x: x / np.sum(x), 1)
)

# %%
XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_endog")], axis=1)
ml_sdem_saturated_restricted = ps.model.spreg.ML_Error(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    w=ps.lib.weights.full2W(G.to_numpy()),
    name_x=XGX.columns.tolist(),
    spat_diag=True,
)
print(ml_sdem_saturated_restricted.summary)

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0145946       0.0270940       0.5386640       0.5901187
#                   rm      -5.4836955       1.9487001      -2.8140274       0.0048925
#    price_to_earnings      -0.0063200       0.0025687      -2.4603718       0.0138793
#        profit_margin       0.0157316       0.0127184       1.2369181       0.2161175
#    price_to_research       0.0000054       0.0000026       2.0622207       0.0391867
#             rm_endog     -21.7774523      10.2503404      -2.1245589       0.0336234
# price_to_earnings_endog      -0.0301682       0.0210504      -1.4331401       0.1518178
# market_capitalization_endog      -0.0000000       0.0000000      -0.3212339       0.7480331
#  profit_margin_endog       0.1402799       0.1165547       1.2035545       0.2287618
# price_to_research_endog       0.0000327       0.0000352       0.9270421       0.3539047
#               lambda      -1.0000000       0.3650585      -2.7392871       0.0061573
# %%
# normality
sms.jarque_bera(ml_sdem_saturated_restricted.e_filtered)
# p-value \wo exlusion 0.21748992

# %%
# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(ml_sdem_saturated_restricted.e_filtered, XGX)
list(zip(name, test))
# 1.2331696707213434e-06
# a p-value below an appropriate threshold (e.g. p < 0.05) then the null hypothesis of homoskedasticity is rejected and heteroskedasticity assumed


# white test:
test = sms.het_white(ml_sdem_saturated_restricted.e_filtered, X)
list(zip(name, test))
# 'p-value', 1.0
# null hypothesis of homoskedasticity


# %%
OLS(np.square(ml_sdem_saturated_restricted.e_filtered), G @ y).fit().summary()

# %%
OLS(np.square(ml_sdem_saturated_restricted.e_filtered), y).fit().summary()

# %%
# heteroskedasticity plots
(
    reduce(
        add,
        [
            v.to_frame()
            .assign(Residuals=ml_sdem_saturated_restricted.e_filtered)
            .hvplot.scatter(x=k, y="Residuals")
            for k, v in X.items()
        ],
    )
    * hv.HLine(0).opts(color="black")
).cols(1)

# %%
XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_endog")], axis=1)
XGX_unsaturated = XGX.drop(
    columns=[
        "market_capitalization",
        "market_capitalization_endog",
        "price_to_research_endog",
        "profit_margin",
    ]
)
ml_sdem_unsaturated_restricted = ps.model.spreg.ML_Error(
    y=y.to_numpy(),
    x=XGX_unsaturated.to_numpy(),
    w=ps.lib.weights.full2W(G.to_numpy()),
    name_x=XGX_unsaturated.columns.tolist(),
    spat_diag=True,
)
print(ml_sdem_unsaturated_restricted.summary)

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0014016       0.0211017       0.0664196       0.9470438
#                   rm      -5.3589347       1.9357944      -2.7683388       0.0056343
#    price_to_earnings      -0.0055857       0.0025337      -2.2046123       0.0274813
#    price_to_research       0.0000052       0.0000027       1.9601814       0.0499746
#             rm_endog     -21.8865059       9.5855754      -2.2832751       0.0224142
# price_to_earnings_endog      -0.0449007       0.0159773      -2.8102792       0.0049499
#  profit_margin_endog       0.2422901       0.0749886       3.2310272       0.0012335
#               lambda      -0.9804466       0.3650528      -2.6857669       0.0072364

# %%
# normality
sms.jarque_bera(ml_sdem_unsaturated_restricted.e_filtered)
# p-value \wo exlusion 0.1217

# %%
# Heteroskedasticity tests¶
## Breush-Pagan test:
import statsmodels.stats.api as sms

name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(ml_sdem_unsaturated_restricted.e_filtered, XGX_unsaturated)
list(zip(name, test))
# 9.685e-08
# a p-value below an appropriate threshold (e.g. p < 0.05) then the null hypothesis of homoskedasticity is rejected and heteroskedasticity assumed


try:
    # white test:
    test = sms.het_white(ml_sdem_unsaturated_restricted.e_filtered, XGX_unsaturated)
    list(zip(name, test))
    # 'p-value', 1.0
    # null hypothesis of homoskedasticity
except:
    pass


# %%
y.assign(errors=ml_sdem_unsaturated_restricted.e_filtered).hvplot.scatter(
    x="returns", y="errors"
) * y.assign(true=y.returns).sort_values("returns").hvplot.line(
    x="returns", predicted="true"
)


# %%
# heteroskedasticity plots
(
    reduce(
        add,
        [
            v.to_frame()
            .assign(Residuals=ml_sdem_saturated_restricted.e_filtered)
            .hvplot.scatter(x=k, y="Residuals")
            for k, v in XGX_unsaturated.items()
        ],
    )
    * hv.HLine(0).opts(color="black")
).cols(1)


#####################
def opt(lam):
    print(lam)
    distribution_star = stats.poisson(lam)  # average_degree)  # samples.mean())
    D_star = (
        renamed_distances.loc[features.index, features.index]
        .replace(0, np.nan)
        .subtract(1)
        .apply(distribution_star.pmf)
        .fillna(0)
    )
    D_star = D_star.drop(exclude).drop(columns=exclude)
    G_star = D_star.apply(lambda x: x / np.sum(x), 1)
    G_star = (
        G_star.drop(["GES", "ERA", "AZO", "JLL"])
        .drop(columns=["GES", "ERA", "AZO", "JLL"])
        .apply(lambda x: x / np.sum(x), 1)
    )

    X_unsaturated = X.drop(
        columns=["market_capitalization", "profit_margin", "price_to_earnings"]
    )
    ml_kp_unsaturated_restricted = ps.model.spreg.GM_Combo_Het(
        y=y.drop(["AZO", "JLL"]).to_numpy(),
        x=X_unsaturated.drop(["AZO", "JLL"]).to_numpy(),
        w=ps.lib.weights.full2W(G_star.to_numpy()),
        w_lags=3,
        name_x=X_unsaturated.columns.tolist(),
        step1c=True,
    )
    ml_kp_unsaturated_restricted.z_stat[-1][-1]

    try:
        return -ml_kp_unsaturated_restricted.pr2_e, ml_kp_unsaturated_restricted, G_star
    except:
        return 0, None, None


best_param = minimize(lambda x: opt(x)[0], x0=0.7, tol=1e-6)

r_squared_adj, ml_kp_unsaturated_restricted_opt, G_star_opt = opt(best_param.x)

np.linalg.matrix_rank(G_star_opt)
np.linalg.matrix_rank(G_star_opt @ G_star_opt)
np.linalg.matrix_rank(G_star_opt @ G_star_opt @ G_star_opt)
# is of full rank

# ps.model.spreg.summary_output
print(ml_kp_unsaturated_restricted_opt.summary)

# opt_sr2 lam = 0.7451337

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0032878       0.0032159       1.0223641       0.3066086
#                   rm      -4.5239588       2.2698082      -1.9931018       0.0462503
#    price_to_research       0.0000046       0.0000005      10.1192730       0.0000000
#            W_dep_var       0.9999969       0.3623707       2.7595963       0.0057873
#               lambda      -0.9731263       0.5477732      -1.7765133       0.0756484


# opt lam = average_degree

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0044581       0.0035291       1.2632212       0.2065097
#                   rm      -5.0685092       2.3589524      -2.1486272       0.0316640
#    price_to_research       0.0000045       0.0000005       8.9667348       0.0000000
#            W_dep_var       0.8632708       0.3192055       2.7044358       0.0068420
#               lambda      -1.0000000       0.6704572      -1.4915196       0.1358251


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
        step1c=False,
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
            step1c=step1c,
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
XGX = pd.concat([X, (G @ X).rename(columns=lambda s: s + "_endog")], axis=1)
XGX_unsaturated = XGX.drop(
    columns=["market_capitalization", "market_capitalization_endog"]
)
ml_kp_exo_unsaturated_restricted = GM_Combo_Het(
    y=y.to_numpy(),
    x=XGX_unsaturated.to_numpy(),
    w=ps.lib.weights.full2W(G.to_numpy()),
    q=(G @ G @ XGX_unsaturated).to_numpy(),
    name_x=XGX_unsaturated.columns.tolist(),
    step1c=True,
)
print(ml_kp_exo_unsaturated_restricted.summary)


###############


# %%
gm_sdem_saturated_restricted_het = ps.model.spreg.GM_Error_Het(
    y=y.to_numpy(),
    x=XGX.to_numpy(),
    w=ps.lib.weights.full2W(G.to_numpy()),
    name_x=XGX.columns.tolist(),
    # step1c=True,
)
print(gm_sdem_saturated_restricted_het.summary)

#             Variable     Coefficient       Std.Error     z-Statistic     Probability
# ------------------------------------------------------------------------------------
#             CONSTANT       0.0125233       0.0282947       0.4426009       0.6580545
#                   rm      -5.4663139       2.0949114      -2.6093294       0.0090720
#    price_to_earnings      -0.0067767       0.0033308      -2.0345395       0.0418972
# market_capitalization       0.0000000       0.0000000       1.3310107       0.1831855
#        profit_margin       0.0137293       0.0128114       1.0716454       0.2838793
#    price_to_research       0.0000055       0.0000008       6.5935493       0.0000000
#             rm_endog     -20.6717060       9.5466293      -2.1653408       0.0303616
# price_to_earnings_endog      -0.0279674       0.0235296      -1.1886033       0.2345958
# market_capitalization_endog      -0.0000000       0.0000000      -0.4830111       0.6290879
#  profit_margin_endog       0.1372920       0.1303382       1.0533518       0.2921798
# price_to_research_endog       0.0000316       0.0000385       0.8197729       0.4123456
#               lambda      -1.0000000       0.7222363      -1.3845884       0.1661784


# %%
OLS(np.square(gm_sdem_saturated_restricted_het.e_filtered), G @ y).fit().summary()


# %%
OLS(np.square(gm_sdem_saturated_restricted_het.e_filtered), y).fit().summary()


# %%
gm_kp_saturated_restricted_het = ps.model.spreg.GM_Combo_Het(
    y=y.to_numpy(),
    x=X.to_numpy(),
    w=ps.lib.weights.full2W(G.to_numpy()),
    name_x=X.columns.tolist(),
    step1c=True,
)
print(gm_kp_saturated_restricted_het.summary)


# %%
OLS(np.square(gm_kp_saturated_restricted_het.e_filtered), G @ y).fit().summary()


# %%
OLS(np.square(gm_kp_saturated_restricted_het.e_filtered), y).fit().summary()


# %%
slx = ps.model.spreg.OLS(
    y=y.to_numpy(),
    x=X.to_numpy(),
    w=ps.lib.weights.full2W(G.to_numpy()),
    name_x=X.columns.tolist(),
    white_test=True,
    spat_diag=True,
    moran=True,
)

print(slx.summary)

slx_error = y.subtract(slx.predy)

# %%
OLS(slx_error, y).fit().summary()
# correlated with errors OVB

# %%
OLS(slx_error, G @ y).fit().summary()
