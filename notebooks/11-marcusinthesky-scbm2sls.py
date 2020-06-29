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
#%%
features = context.catalog.load("features")
renamed_distances = context.catalog.load("renamed_distances")


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
