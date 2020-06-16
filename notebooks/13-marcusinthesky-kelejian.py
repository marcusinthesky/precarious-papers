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


def opt_lam(lam):
    try:
        print(lam)
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

        w = ps.lib.weights.full2W(G.to_numpy(), ids=G.index.tolist())
        model = ps.model.spreg.GM_Combo_Het(
            step1c=True,
            y=(y).to_numpy(),
            x=X.to_numpy(),
            w_lags=2,
            w=w,
            name_x=X.columns.tolist(),
        )

        return -model.pr2_e, model, G
    except:
        return 0, None, None


best_param = minimize(lambda x: opt_lam(x)[0], x0=2, method="Nelder-Mead", tol=1e-6,)

best_param

# %%
pr2_e, model, G = opt_lam(best_param.x)

I = np.identity(G.shape[1])

print(model.summary)


# %%
features = [
    "α",
    "rₘ - rf",
    "Price-to-earnings",
    "Market capitalization",
    "Profit margin",
    "Price-to-research",
    "Wy",
    "λ",
]
S = np.linalg.inv(I - model.betas[-2][0] * G)
effects = pd.DataFrame(model.betas[1:-2], index=features[1:-2], columns=["Coefficient"])
(
    effects.assign(
        Direct=lambda df: df.Coefficient.apply(lambda s: np.diag(S * s).mean())
    )
    .assign(
        Total=lambda df: df.Coefficient.apply(lambda s: (S * s).sum().sum() / model.n)
    )
    .assign(Indirect=lambda df: df.Total - df.Direct)
    .loc[:, ["Coefficient", "Direct", "Indirect", "Total"]]
)


# %%
b = model.betas[1:-2]
rho = model.rho
btot = b / (float(1) - rho)
bind = btot - b

full_eff = pd.DataFrame(
    np.hstack([b, bind, btot]),
    index=features[1:-2],
    columns=["Direct", "Indirect", "Total"],
)
full_eff

# %%
pd.DataFrame(
    {
        "Estimate": model.betas.flatten(),
        "SE": model.std_err.flatten(),
        "t": [z for z, p in model.z_stat],
        "p-value": [p.round(3) for z, p in model.z_stat],
    },
    index=features,
)


# %%
# probablity plot
osm_osr, slope_intercept_r = stats.probplot(model.e_filtered.flatten(), dist="norm")

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

# %%
stats.ttest_1samp(model.e_filtered, 0, axis=0)

# %%
# heteroskedasticity plots
(
    reduce(
        add,
        [
            v.to_frame()
            .assign(Residuals=model.e_filtered)
            .hvplot.scatter(x=k, y="Residuals")
            for k, v in X.items()
        ],
    )
    * hv.HLine(0).opts(color="black")
).cols(1)
