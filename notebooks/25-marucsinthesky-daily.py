# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: precariouspapers
#     language: python
#     name: python3
# ---

# %%
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

# %%
cost = price.loc[:, pd.IndexSlice[:, "close"]]
cost.columns = cost.columns.droplevel(1)
cap = cost * factors.set_index('symbol').commonStock
weight = cap.apply(lambda df: df / df.sum(), axis=1)

# %%
p = price.loc[:, pd.IndexSlice[:, "close"]].pct_change().dropna(how='all')
p.columns = p.columns.droplevel(1)
r = (p.reset_index()
     .melt(id_vars='date', var_name='symbol', value_name='ri')
     .groupby(['date','symbol']).head(1)
     .assign(rf = rf)
     .merge(features.loc[:, ['price_to_earnings', 'market_capitalization', 'profit_margin', 'price_to_research']], left_on='symbol', right_index=True)
     .merge(matched.groupby("symbol").head(1)
             .loc[:, ['symbol', 'exchange']]
             .merge(indices
                    .pct_change()
                    .dropna(how='all')
                    .reset_index()
                    .melt(id_vars='Date', var_name='exchange', value_name='rm')
                    .rename(columns={'Date':'date'}), on='exchange', how='outer')
             .drop(columns=['exchange']),
           on=['date', 'symbol'], how='left') 
     .set_index(['date', 'symbol'])
     .assign(rm_rf = lambda df: df.rm - df.rf,
             ri_rf = lambda df: df.ri - df.rf)
     .drop(columns=['rm','rf', 'ri'])
            )

# %%
sample_index = (weight * p).sum(1)

# %%
from scipy.stats.mstats import winsorize

# %%
X = (r
     .drop(columns=['ri_rf'])
     .rename(columns={'price_to_earnings': 'Price_to_Earnings', 'market_capitalization': 'Market_Capitalization', 'profit_margin': 'Profit_Margin', 'price_to_research':'Price_to_Research'}))
y = pd.Series(winsorize(r.ri_rf, [0.0325, 0.0515]), index=r.index, name='ri_rf').to_frame()

# %%
from statsmodels.stats.stattools import jarque_bera
'JB: {} JBpv: {}, skew: {}, kurtosis: {}'.format(*jarque_bera(y))

# %%
y.hvplot.hist(title='Histogram of Winsorized Excess Returns', height=400, wight=800) *\
pd.Series(np.random.normal(y.mean(), y.std(), 10000)).hvplot.kde(fill_alpha=0, label='Samples from Moment Matching Normal Distribution')

# %%
ols = OLS(y, X.assign(alpha=1)).fit()
ols.summary()

# %%
# weibull
D = (
    distances.loc[p.columns, p.columns]
    .replace(0, np.nan)
    .pipe(lambda df: df.apply(stats.weibull_min(*stats.weibull_min.fit(df.melt().value.dropna(), floc=0, f0=1)).pdf))
    .fillna(0)
    .apply(lambda df: df/df.sum(), axis=1)
    .pipe(lambda df: df.where(lambda df: df.sum(1) != 0).fillna((df.shape[0]-1)**-1))
    .pipe(lambda df: df * (1 - np.eye(df.shape[0])))
    .apply(lambda df: df/df.sum(), axis=1)
)
assert np.all(np.diag(D) == 0)
assert D.sum(1).apply(np.testing.assert_almost_equal, desired=1).all()
assert (D.index == p.columns).all()
assert (D.columns == p.columns).all()
assert D.var().nunique() > 1

# %%
y

# %%
fama = context.catalog.load('fama_french')
centralty = pd.read_parquet('centrality.parquet')
merged_centralities = matched.loc[:, ['node_id', 'symbol']].merge(centralty, left_on='node_id', right_index=True, how='inner').groupby('symbol').Centrality.mean().rename('ICIJ Centrality')
F = X.join(fama.pipe(lambda df: df - df + pd.np.random.normal(scale=5, size=df.shape)), on=['date']).loc[:, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']].join(merged_centralities, on='symbol')
R = y.add(rf).ri_rf.subtract(F.RF).to_frame('ri_rf')
F = F.drop(columns=['RF'])

# %%
selector = y.index.get_level_values('symbol')
W = D.loc[selector, selector].apply(lambda df: df/df.sum(), axis=1).pipe(lambda df: pd.DataFrame(df.to_numpy(), index=y.index, columns=y.index))
w = ps.lib.weights.full2W(W.to_numpy())

# %%
Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization'])
spatial_ols = ps.model.spreg.OLS(y.to_numpy(), Z.to_numpy(), w=w, moran=True, spat_diag=True, white_test=True, robust='white',
                                 name_w='Weibull Kernel over Shortest Path Length', name_ds='Paradise Papers Metadata', name_y=y.columns.tolist(), name_x=Z.columns.tolist())
print(spatial_ols.summary)

# %%
Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization']).pipe(lambda df: df.join((W@df).rename(columns=lambda s: 'W_'+s))).drop(columns=['W_rm_rf'])
spatial_ols = ps.model.spreg.OLS(y.to_numpy(), Z.to_numpy(), w=w, moran=True, spat_diag=True, white_test=True, robust='white',
                                 name_w='Weibull Kernel over Shortest Path Length', name_ds='Paradise Papers Metadata', name_y=y.columns.tolist(), name_x=Z.columns.tolist())
print(spatial_ols.summary)

# %%
ols = OLS(R, F.assign(alpha=1).drop(columns=['ICIJ Centrality'])).fit()
ols.summary()

# %%
sample_index.to_frame('Sample Market Capitalization Weighted Returns').join(fama).apply(lambda df: df / df.std(), axis=0).hvplot.line(height=400, width=1000, colormap='Category10', title='Comparison to Fama-French 5-Factors')#x='Sample Market Capitalization Weighted Index', y='CMA')

# %%
sample_index.to_frame('Sample Market Capitalization Weighted Returns').join(fama).drop(columns=['RF'.corr('kendall')

# %%
G = F.drop(columns=['ICIJ Centrality', 'SMB', 'RMW', 'HML'])#.pipe(lambda df: df.join((W@df).rename(columns=lambda s: 'W_'+s))).drop(columns=['W_Mkt-RF'])

spatial_ols = ps.model.spreg.OLS(R.to_numpy(), G.to_numpy(), w=w, moran=True, spat_diag=True, white_test=True, robust='white',
                                 name_w='Weibull Kernel over Shortest Path Length', name_ds='Paradise Papers Metadata', name_y=y.columns.tolist(), name_x=G.columns.tolist())
print(spatial_ols.summary)

# %%
Z = X.drop(columns=['Market_Capitalization', 'Price_to_Research',])
spatial_error = ps.model.spreg.ML_Lag(y=y.to_numpy(), x=Z.to_numpy(), w=w, spat_diag=True,
                                 name_w='Weibull Kernel over Shortest Path Length', name_ds='Paradise Papers Metadata', name_y=y.columns.tolist()[0], name_x=Z.columns.tolist())
print(spatial_error.summary)

# %%
Z = X.drop(columns=['Market_Capitalization', 'Price_to_Research',])
spatial_error = ps.model.spreg.ML_Error(y=y.to_numpy(), x=Z.to_numpy(), w=w, spat_diag=True,
                                 name_w='Weibull Kernel over Shortest Path Length', name_ds='Paradise Papers Metadata', name_y=y.columns.tolist()[0], name_x=Z.columns.tolist())
print(spatial_error.summary)

# %%
Z = X.drop(columns=['Market_Capitalization', 'Price_to_Earnings',])
spatial_error = ps.model.spreg.GM_Error_Het(y=y.to_numpy(), x=Z.to_numpy(), w=w, #w_lags=2, 
                                 name_w='Weibull Kernel over Shortest Path Length', name_ds='Paradise Papers Metadata', name_y=y.columns.tolist()[0], name_x=Z.columns.tolist())
print(spatial_error.summary)

# %%
# Z = X.drop(columns=['Market_Capitalization', 'Price_to_Earnings', ])
# spatial_error = ps.model.spreg.GM_Combo_Het(y=y.to_numpy(), x=Z.to_numpy(), w=w, w_lags=2, step1c=True,
#                                  name_w='Weibull Kernel over Shortest Path Length', name_ds='Paradise Papers Metadata', name_y=y.columns.tolist()[0], name_x=Z.columns.tolist())
# print(spatial_error.summary)

# %%
# Z = X.drop(columns=['Market_Capitalization', 'Price_to_Earnings'])
# spatial_error = ps.model.spreg.GM_Combo_Hom(y=y.to_numpy(), x=Z.to_numpy(), w=w, w_lags=2, 
#                                  name_w='Weibull Kernel over Shortest Path Length', name_ds='Paradise Papers Metadata', name_y=y.columns.tolist()[0], name_x=Z.columns.tolist())
# print(spatial_error.summary)

# %%
