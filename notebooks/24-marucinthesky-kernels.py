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
import warnings
from functools import partial

import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.networkx as hvnx
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from sklearn.manifold import Isomap

tqdm().pandas()
hv.extension("bokeh")

#%%
paradise_edges = context.io.load("paradise_edges")

# %%
paradise_graph = nx.convert_matrix.from_pandas_edgelist(
    df=paradise_edges,
    source="START_ID",
    target="END_ID",
    edge_attr=paradise_edges.columns.drop(["START_ID", "END_ID"]).tolist(),
)


# %%
try:
    centralty = pd.read_parquet("centrality.parquet")
except:
    centrality = pd.Series(
        nx.eigenvector_centrality(paradise_graph, tol=1e-7, max_iter=5000)
    ).to_frame("Centrality")
    centrality.to_parquet("centrality.parquet")

# %%
prob = paradise_graph.number_of_edges() / (paradise_graph.number_of_nodes() ** 2)
pd.Series(stats.geom(prob).rvs(1000)).hvplot.hist()

# %%
D = (
    renamed_distances.loc[features.index, features.index]
    .replace(0, np.nan)
    .pipe(
        lambda df: df.apply(
            stats.weibull_min(
                *stats.weibull_min.fit(df.melt().value.dropna(), floc=0, f0=1)
            ).pdf
        )
    )  # stats.geom(prob).pmf))#
    .fillna(0)
)
graph = nx.from_pandas_adjacency(D)
# graph_centrality = pd.Series(nx.eigenvector_centrality(graph), name='Matched_Centrality')

# %%
dist = (
    renamed_distances.loc[features.index, features.index]
    .replace(0, np.nan)
    .pipe(lambda df: stats.weibull_min.fit(df.melt().value.dropna(), floc=0, f0=1.0))
)
dist

# %%
pd.Series(stats.weibull_min(*dist).rvs(1000)).hvplot.hist()

# %%
merged_centralities = (
    matched.loc[:, ["node_id", "symbol"]]
    .merge(centralty, left_on="node_id", right_index=True, how="inner")
    .groupby("symbol")
    .Centrality.mean()
)
graph_centrality = pd.Series(
    nx.eigenvector_centrality(graph), name="Matched_Centrality"
)
features_centrality = features.join(
    merged_centralities.rename("ICIJ_Centrality"), how="left"
)

# %%
from scipy.stats.mstats import winsorize

# %%
X, y = (
    features_centrality.assign(
        ri_rf=lambda df: winsorize(
            df.returns, [0.028 - 0.0, 0.062 + 0.02]
        ),  # [0.028, 0.062]
        rm_rf=lambda df: df.rm,
    )
    .drop(columns=["returns", "rm"])
    #         .join(pd.Series(nx.eigenvector_centrality(graph), name='Matched_Centrality'))
    .rename(
        columns={
            "price_to_earnings": "Price_to_Earnings",
            "market_capitalization": "Market_Capitalization",
            "profit_margin": "Profit_Margin",
            "price_to_research": "Price_to_Research",
        }
    )
    .pipe(lambda df: (df.drop(columns=["ri_rf", "alpha"]), df.loc[:, ["ri_rf"]]))
)

# %%
from statsmodels.stats.stattools import jarque_bera

"JB: {} JBpv: {}, skew: {}, kurtosis: {}".format(*jarque_bera(y))

# %%
stats.kstest(
    features_centrality.returns,
    stats.norm(
        features_centrality.returns.mean(), features_centrality.returns.std()
    ).cdf,
)

# %%

# %%
y.hvplot.hist(
    title="Histogram of Winsorized Excess Returns", height=400, wight=800
) * pd.Series(np.random.normal(y.mean(), y.std(), 10000)).hvplot.kde(
    fill_alpha=0,
    xlabel="Excess Returns",
    label="Samples from Moment Matching Normal Distribution",
)

# %%

# %%
ols = OLS(
    y,
    X.assign(alpha=1).drop(
        columns=["Price_to_Earnings", "Market_Capitalization", "ICIJ_Centrality"]
    ),
).fit()
ols.summary()

# %%
from statsmodels.graphics.regressionplots import influence_plot

fig, ax = plt.subplots(figsize=(20, 10))
fig = influence_plot(ols, ax=ax)
fig.tight_layout(pad=1.0)
plt.show()

# %%
outlier_results = ols.get_influence()
pd.Series(outlier_results.cooks_distance[0], y.index).sort_values(
    ascending=False
).hvplot.bar(width=1200, title="Cooks Distance").opts(xrotation=90)

# %%
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# %%
H = pairwise_distances(StandardScaler().fit_transform(outlier_results.dfbeta)).sum(1)

# %%
pd.Series(H, y.index).sort_values(ascending=False).hvplot.bar(
    width=1200, title="Leave-one-out pairwise standardized dfbeta distances"
).opts(xrotation=90)

# %%
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RANSACRegressor, HuberRegressor, TheilSenRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# %%
# exclude = {'FGEN', 'DNN', 'GLBS', 'EXEL', 'MASI', 'CL', 'GLMD'}
# X, y = X.drop(exclude), y.drop(exclude)

# %%
degree = np.mean(
    np.array(list(dict(paradise_graph.degree()).values()), dtype="float64")
)

# %%
prob_proj = graph.number_of_edges() / (graph.number_of_nodes() ** 2)


# %%
# weibull
W = (
    distances.loc[y.index, y.index]
    .replace(0, np.nan)
    .pipe(
        lambda df: df.apply(
            stats.weibull_min(
                *stats.weibull_min.fit(df.melt().value.dropna(), floc=0, f0=1)
            ).pdf
        )
    )
    .fillna(0)
    .apply(lambda df: df / df.sum(), axis=1)
    .pipe(
        lambda df: df.where(lambda df: df.sum(1) != 0).fillna((df.shape[0] - 1) ** -1)
    )
    .pipe(lambda df: df * (1 - np.eye(df.shape[0])))
    .apply(lambda df: df / df.sum(), axis=1)
)
assert np.all(np.diag(W) == 0)
assert W.sum(1).apply(np.testing.assert_almost_equal, desired=1).all()
assert (W.index == y.index).all()
assert (W.columns == y.index).all()
assert W.var().nunique() > 1

huber = Pipeline(
    [("scaler", StandardScaler()), ("model", HuberRegressor(epsilon=2.25, alpha=0.0))]
)  # 2.028
huber.fit(X.assign(Wy=W @ y), y)
huber_outliers = huber.named_steps["model"].outliers_
print("# Outliers: ", huber_outliers.sum())

# %%
huber_outliers_set = set(
    y.loc[huber_outliers, :].index.tolist()
)  # | {'DNN', 'FGEN', 'EXEL', 'SMMF', 'MASI', 'TAT', 'CL', 'GLBS', 'THO', 'SHIP', 'HNW'}
huber_outliers_set

# %%
# try:
#     exclude = set(y.loc[(huber_outliers_set), :].index.tolist())# | {'DNN', } #| { 'HL', 'MSFT', 'LVS', 'JNJ', 'EXEL', 'CCF', 'DNN', 'TAT', 'HNW', 'GLNG'}
#     print('Excluded: ',len(exclude))
#     X, y = X.drop(exclude), y.drop(exclude)
# except:
#     print('Error')
#     pass

# %%

# %%
pos = nx.kamada_kawai_layout(graph)  # positions for all nodes
# hvnx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4, width=600, height=600) *\

component = hvnx.draw(
    graph,
    pos,
    edge_color="weight",
    edge_cmap="greys",
    edge_width=hv.dim("weight") * 25,
    node_size=hv.dim("size") * 20,
).opts(title="Cooks Distance", colorbar=True) * hvnx.draw_networkx_nodes(
    graph,
    pos,  # nodelist=list(p.keys()),
    node_cmap="turbo",
    node_size=80,
    node_color=H,  # .cooks_distance[0],
    cmap="turbo",
    colorbar=True,
    height=600,
    width=1000,
    logz=True,
)

# %%
hvnx.draw_kamada_kawai(
    graph,
    node_cmap="turbo",
    node_size=150,
    node_color=outlier_results.cooks_distance[0],
    label="Cooks Distance over Graph",
    cmap="turbo",
    edge_color=hv.dim("weight", np.exp),
    edge_alpha=0.25,
    edge_cmap="greys",
    edge_width=hv.dim("weight", pd.np.exp) * 2.5,
    colorbar=True,
    #                            title='Cooks Distance',
    height=600,
    width=1000,
    logz=True,
)

# %%

w = ps.lib.weights.full2W(W.to_numpy())
Z = X.drop(columns=["Price_to_Earnings", "Market_Capitalization", "ICIJ_Centrality"])
spatial_ols = ps.model.spreg.OLS(
    y.to_numpy(),
    Z.to_numpy(),
    w=w,
    moran=True,
    spat_diag=True,
    white_test=True,
    robust=None,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist(),
    name_x=Z.columns.tolist(),
)
print(spatial_ols.summary)

# %%
from scipy.sparse.linalg import eigs

val, vec = eigs(D.to_numpy(), k=2, which="LM", sigma=0)

pd.DataFrame(vec.real, columns=["Component 1", "Component 2"], index=W.index).assign(
    ticker=W.index
).hvplot.scatter("Component 1", "Component 2", color="ticker")

# %%
V = pairwise_distances(np.abs(vec))
V = stats.norm(0, V.flatten().std()).pdf(V)
pd.np.fill_diagonal(V, 0)

# %%
J = pd.DataFrame(V, index=W.index, columns=W.index).apply(
    lambda df: df / df.sum(), axis=1
)
j = ps.lib.weights.full2W(J.to_numpy())

# %%
m = ps.explore.esda.Moran(y, j)

# %%
(m.I, m.p_sim)

# %%
# c  = ps.explore.esda.Geary(y, j)
# c.C, c.p_sim

# %%
g = ps.explore.esda.G(y, j)
g.G, g.p_sim

# %%
w = ps.lib.weights.full2W(W.to_numpy())

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# %%
pca = Pipeline([("scale", StandardScaler()), ("pca", PCA())])
F = X.pipe(lambda df: df.join((W @ df).rename(columns=lambda s: "W_" + s)))
P = pd.DataFrame(pca.fit_transform(F), index=X.index).rename(
    columns=lambda s: "Component " + str(s + 1)
)

# %%
pca.named_steps["pca"].singular_values_.min()

# %%
(pca.named_steps["pca"].singular_values_ >= 1).sum()

# %%
pd.Series(pca.named_steps["pca"].explained_variance_ratio_).hvplot.bar(
    title="Explained Variance Ratio"
)

# %%
pca_object = pca.named_steps["pca"]

# %%
first, second = map(
    lambda s: "(" + str(s) + "%)",
    np.round(pca.named_steps["pca"].explained_variance_ratio_[:2] * 100, 2),
)

# %%
pd.DataFrame(pca_object.components_, index=F.columns.str.replace("_", " ")).rename(
    columns=lambda s: "Component " + str(s + 1)
).reset_index().hvplot.labels(
    x="Component 1",
    xlabel="Component 1 " + first,
    y="Component 1",
    ylabel="Component 2 " + second,
    text="index",
    text_baseline="top",
    width=800,
    height=600,
    title="Spatial Lag Explanatory Biplot",
    hover=False,
)

# %%
OLS(y, P.assign(alpha=1)).fit().summary()

# %%
# # geometric
# W = (
#     renamed_distances.loc[y.index, y.index]
#     .replace(0, np.nan)
#     .pipe(lambda df: df.applymap(stats.geom(df.melt().value.dropna().mean()**-1).pmf))
#     .fillna(0)
#     .apply(lambda df: df/df.sum(), axis=1)
#     .pipe(lambda df: df.where(lambda df: df.sum(1) != 0).fillna((df.shape[0]-1)**-1))
#     .pipe(lambda df: df * (1 - np.eye(df.shape[0])))
#     .apply(lambda df: df/df.sum(), axis=1)
# )
# assert np.all(np.diag(W) == 0)
# assert W.sum(1).apply(np.testing.assert_almost_equal, desired=1).all()
# assert (W.index == y.index).all()
# assert (W.columns == y.index).all()
# assert W.var().nunique() > 1

# %%
# # poisson
# W = (
#     renamed_distances.loc[y.index, y.index]
#     .replace(0, np.nan)
#     .pipe(lambda df: df.applymap(stats.poisson(df.melt().value.dropna().mean()).pmf))
#     .fillna(0)
#     .apply(lambda df: df/df.sum(), axis=1)
#     .pipe(lambda df: df.where(lambda df: df.sum(1) != 0).fillna((df.shape[0]-1)**-1))
#     .pipe(lambda df: df * (1 - np.eye(df.shape[0])))
#     .apply(lambda df: df/df.sum(), axis=1)
# )
# assert np.all(np.diag(W) == 0)
# assert W.sum(1).apply(np.testing.assert_almost_equal, desired=1).all()
# assert (W.index == y.index).all()
# assert (W.columns == y.index).all()
# assert W.var().nunique() > 1

# %%

w = ps.lib.weights.full2W(W.to_numpy())
Z = X.drop(columns=["Price_to_Earnings", "Market_Capitalization", "ICIJ_Centrality"])
spatial_ols = ps.model.spreg.OLS(
    y.to_numpy(),
    Z.to_numpy(),
    w=w,
    moran=True,
    spat_diag=True,
    white_test=True,
    robust=None,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist(),
    name_x=Z.columns.tolist(),
)
print(spatial_ols.summary)

# %%

w = ps.lib.weights.full2W(W.to_numpy())
Z = (
    X.drop(columns=["Price_to_Earnings", "Market_Capitalization", "ICIJ_Centrality"])
    .pipe(lambda df: df.join((W @ df).rename(columns=lambda s: "W_" + s)))
    .drop(columns=["W_rm_rf"])
)
spatial_slx = ps.model.spreg.OLS(
    y.to_numpy(),
    Z.to_numpy(),
    w=w,
    moran=True,
    spat_diag=True,
    white_test=True,
    robust=None,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist(),
    name_x=Z.columns.tolist(),
)
print(spatial_slx.summary)

# %%
from sklearn.decomposition import FactorAnalysis

# %%
Z.head()

# %%
U = pd.DataFrame(
    Pipeline([("scale", StandardScaler()), ("pca", PCA(2))]).fit_transform(Z),
    index=Z.index,
)


# %%
slx = OLS(y, U.assign(alpha=1)).fit()
slx.summary()

# %%
slx = OLS(y, Z.assign(alpha=1)).fit()
slx.summary()

# %%
from statsmodels.graphics.regressionplots import influence_plot

fig, ax = plt.subplots(figsize=(20, 10))
fig = influence_plot(slx, ax=ax)
fig.tight_layout(pad=1.0)
plt.savefig("slx_leverage.svg")

# %%
slx_outlier_results = ols.get_influence()
pd.Series(slx_outlier_results.cooks_distance[0], y.index).sort_values(
    ascending=False
).hvplot.bar(width=1200, title="Cooks Distance SLX Model").opts(xrotation=90)

# %%
hvnx.draw_kamada_kawai(
    graph,
    node_cmap="turbo",
    node_size=150,
    node_color=slx_outlier_results.cooks_distance[0],
    label="Cooks Distance of SLX Model over Graph",
    cmap="turbo",
    edge_color=hv.dim("weight", np.exp),
    edge_alpha=0.25,
    edge_cmap="greys",
    edge_width=hv.dim("weight", pd.np.exp) * 2.5,
    colorbar=True,
    #                            title='Cooks Distance',
    height=600,
    width=1000,
    logz=True,
)

# %%
outlier_pca = Pipeline([("scale", StandardScaler()), ("pca", PCA())])

# %%
U = pd.DataFrame(outlier_pca.fit_transform(Z)[:, :2], index=Z.index).rename(
    columns=lambda s: "Component " + str(s + 1)
)
top_cooks = pd.Series(outlier_results.cooks_distance[0], index=y.index).nlargest(3)

# %%
first, seconds = map(
    lambda s: "(" + str(s) + "%)",
    np.round(outlier_pca.named_steps["pca"].explained_variance_ratio_[:2] * 100, 2),
)

# %%
U_cooks = U.assign(**{"Cooks Distance": outlier_results.cooks_distance[0]})

U_cooks.hvplot.scatter(
    x="Component 1",
    y="Component 2",
    xlabel="Component 1 " + first,
    ylabel="Component 2 " + second,
    color="Cooks Distance",
    cmap="turbo",
    logz=False,
    width=900,
    height=500,
    title="Principal Components of our SLX Model with Cooks Outliers",
) * U_cooks.loc[top_cooks.index, :].reset_index().hvplot.labels(
    x="Component 1",
    y="Component 2",
    text="symbol",
    text_baseline="bottom",
)

# %%
pd.Series(outlier_pca.named_steps["pca"].explained_variance_ratio_).hvplot.bar(
    title="Variance Explained Ratio of Principal Components of SLX Explanatory Variables"
)

# %%
from sklearn.cluster import MeanShift

clusters = MeanShift().fit_predict(Z.iloc[:, :3])  # U.iloc[:, :4])

# %%

# %%
U_cooks.assign(Clusters=clusters.astype(str)).hvplot.scatter(
    x="Component 1",
    y="Component 2",
    xlabel="Component 1 " + first,
    ylabel="Component 2 " + second,
    color="Clusters",
    #                        cmap='turbo',
    logz=False,
    width=900,
    height=500,
    title="Principal Components of our SLX Model with Mean Shift Clustering",
)

# %%
hvnx.draw_kamada_kawai(
    graph,  # node_cmap='turbo',
    node_size=150,
    node_color=[str(s) for s in clusters.astype(str)],
    label="Mean Shift Clustering of Principal Components of SLX Model over Graph",
    cmap="Category10",
    edge_color=hv.dim("weight", np.exp),
    edge_alpha=0.25,
    edge_cmap="greys",
    edge_width=hv.dim("weight", pd.np.exp) * 2.5,
    #                          colorbar=True,
    #                            title='Cooks Distance',
    height=600,
    width=1000,
    #                          logz=True
)

# %%
# Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization', 'ICIJ_Centrality', 'Price_to_Research'])

# %%
from sklearn.metrics import pairwise_distances

# %%
pearson_corr = (
    pd.DataFrame(
        pairwise_distances((W @ X).T, metric=lambda x, y: stats.pearsonr(x, y)[0]),
        index=X.columns,
        columns=X.columns,
    )
    .round(4)
    .astype(str)
)
pearson_p = (
    pd.DataFrame(
        pairwise_distances((W @ X).T, metric=lambda x, y: stats.pearsonr(x, y)[1]),
        index=X.columns,
        columns=X.columns,
    )
    .round(4)
    .astype(str)
)

# %%
(pearson_corr + pearson_p.applymap(lambda s: f"\n({s})")).to_clipboard()

# %%
stats.pearsonr()

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.Series(
    [variance_inflation_factor(X.values, i) for i in range(Z.shape[1])],
    index=Z.columns,
    name="VIF",
).round(4).to_clipboard()

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.Series(
    [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])],
    index=Z.columns,
    name="VIF",
).round(
    4
)  # .to_clipboard()

# %%

w = ps.lib.weights.full2W(W.to_numpy())
# Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization', 'ICIJ_Centrality']).pipe(lambda df: df.join((W@df).rename(columns=lambda s: 'W_'+s))).drop(columns=[ 'W_rm_rf'])
spatial_ols = ps.model.spreg.ML_Error(
    y.to_numpy(),
    Z.to_numpy(),
    w=w,
    spat_diag=True,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist(),
    name_x=Z.columns.tolist(),
)
print(spatial_ols.summary)

# %%
# Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization', 'ICIJ_Centrality'])
spatial_error = ps.model.spreg.ML_Lag(
    y=y.to_numpy(),
    x=Z.to_numpy(),
    w=w,
    spat_diag=True,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist()[0],
    name_x=Z.columns.tolist(),
)
print(spatial_error.summary)

# %%
# Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization', 'ICIJ_Centrality']1
spatial_lag = ps.model.spreg.GM_Lag(
    y=y.to_numpy(),
    x=Z.to_numpy(),
    w=w,
    spat_diag=True,
    w_lags=2,
    robust="white",
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist()[0],
    name_x=Z.columns.tolist(),
)
print(spatial_lag.summary)

# %%
# Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization', 'ICIJ_Centrality'])
spatial_error = ps.model.spreg.ML_Error(
    y=y.to_numpy(),
    x=Z.to_numpy(),
    w=w,
    spat_diag=True,
    epsilon=1e-10,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist()[0],
    name_x=Z.columns.tolist(),
)
print(spatial_error.summary)

# %%
# Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization', 'ICIJ_Centrality'])
spatial_error = ps.model.spreg.GM_Error(
    y=y.to_numpy(),
    x=Z.to_numpy(),
    w=w,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist()[0],
    name_x=Z.columns.tolist(),
)
print(spatial_error.summary)

# %%
# Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization', 'ICIJ_Centrality'])
spatial_error = ps.model.spreg.GM_Error_Het(
    y=y.to_numpy(),
    x=Z.to_numpy(),
    w=w,
    max_iter=10,
    step1c=True,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist()[0],
    name_x=Z.columns.tolist(),
)
print(spatial_error.summary)

# %%
# Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization', 'ICIJ_Centrality'])
spatial_error = ps.model.spreg.GM_Combo(
    y=y.to_numpy(),
    x=Z.to_numpy(),
    w=w,
    w_lags=2,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist()[0],
    name_x=Z.columns.tolist(),
)
print(spatial_error.summary)

# %%
# Z = X.drop(columns=['Price_to_Earnings', 'Market_Capitalization', 'ICIJ_Centrality'])
spatial_error = ps.model.spreg.GM_Combo_Het(
    y=y.to_numpy(),
    x=Z.to_numpy(),
    w=w,
    w_lags=2,
    step1c=False,
    inv_method="power_exp",
    max_iter=1000,
    epsilon=1e-10,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist()[0],
    name_x=Z.columns.tolist(),
)
print(spatial_error.summary)

# %%
ps.model.spreg.diagnostics.likratiotest(spatial_slx, spatial_error)

# %%
Z = X.drop(columns=["Price_to_Earnings", "Market_Capitalization", "ICIJ_Centrality"])
spatial_error = ps.model.spreg.GM_Combo_Hom(
    y=y.to_numpy(),
    x=Z.to_numpy(),
    w=w,
    w_lags=2,
    max_iter=100,
    name_w="Weibull Kernel over Shortest Path Length",
    name_ds="Paradise Papers Metadata",
    name_y=y.columns.tolist()[0],
    name_x=Z.columns.tolist(),
)
print(spatial_error.summary)
