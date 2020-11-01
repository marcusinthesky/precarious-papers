# -*- coding: utf-8 -*-
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
R = features.returns

# %%
A = (
    renamed_distances.loc[R.index, R.index]
    .replace(0, np.nan)
#     .clip(0, 4)
#     .apply(dist.pdf)
#     .fillna(0)
#     .multiply(np.equal(f, f.T))
)

# %%
d = A.melt().value.dropna()

# %%
dist = stats.weibull_min(*stats.weibull_min.fit(A.melt().value.dropna(), floc=0, f0=1))

# %%
# dist = stats.poisson(d.mean())

# %%
dist.mean()

# %%
dist.std()

# %%
D = (A.apply(dist.pdf)
    .fillna(0))

# %%
D.sum().sum()

# %%
D.replace(0, np.nan).melt().value.hvplot.hist()

# %%
d.mean()

# %%
pd.Series(stats.poisson(d.mean()).rvs(d.shape[0]), name='v').value_counts()

# %%
pd.Series(stats.exponweib(*stats.exponweib.fit(d, floc=0, f0=1)).rvs(d.shape[0]), name='v').round(0).where(lambda df: df < d.max()).value_counts().sort_index().hvplot.bar('k', 'v', label='Estimated Weibull Distribution', xlabel='Shortest Path Length', ylabel='Probability Density')

# %%

from scipy import stats
import matplotlib.pyplot as plt
pd.Series(stats.exponweib(*stats.exponweib.fit(d, floc=0, f0=1)).rvs(d.shape[0]), name='v').round(0).where(lambda df: df < d.max()).value_counts().sort_index().hvplot.bar('k', 'v', label='Estimated Weibull Distribution', xlabel='Shortest Path Length', ylabel='Count', alpha=0.5) *\
d.value_counts().sort_index().hvplot.bar(xlabel='Shortest Path Length', title="Distribution of Shortest Path Lengths", label='Observed', alpha=0.5) *\
pd.Series(stats.poisson(d.mean()).rvs(d.shape[0]), name='v').where(lambda df: df < d.max()).value_counts().sort_index().hvplot.bar('k', 'v', label='Estimated Poisson Distribution', xlabel='Shortest Path Length', ylabel='Count', alpha=0.5)
# _ = plt.hist(d, bins=np.linspace(0, 16, 33), normed=True, alpha=0.5);
# plt.show()

# %%
A.melt().value.value_counts().sort_index().hvplot.bar(xlabel='Shortest Path Length', title="Distribution of Shortest Path Lengths")

# %%
W = D

# %%
W.sum(1)

# %%
L = np.diag(W.sum(0)) - W

# %%
vals, vec = np.linalg.eigh(L)

# %%
y = R

# %%
factors

# %%
centere
factors.excess.pipe(lambda df: df.fillna(df.mean())).pipe(lambda df: df - df.mean())

# %%
F = np.conjugate(vec) @ (centered)

# %%
C = pd.DataFrame({'Magnitude':F}).applymap(np.abs).assign(**{'Eigenvalues (Frequency)': vals.real})

# %%
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting

# %%
G = graphs.Graph(W.to_numpy(), coords=factors.returns.to_numpy())

# %%
G.compute_fourier_basis()

# %%
G.plot_signal(factors.returns.to_numpy())

# %%
g = factors.returns.to_numpy()

# %%
fig, ax = plt.subplots()
g = filters.Filter(G, lambda x: x)
g.plot(plot_eigenvalues=True, ax=ax)
_ = ax.set_title('Filter frequency response')

# %%
C.groupby('Eigenvalues (Frequency)').Magnitude.sum().reset_index().hvplot.scatter(x='Eigenvalues (Frequency)', y='Magnitude', title='Returns Graph Fourier Series of the Weibull Kernel Laplacian')

# %%
C.hvplot.scatter(x='Eigenvalues (Frequency)', y='Magnitude', title='Centered Returns Graph Fourier Series of the Weibull Kernel Laplacian')

# %%
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

# %%
L

# %%
v, Z = eigsh(L.to_numpy(), 2, which='SM',maxiter=1e10)

# %%
import networkx as nx

# %%
pd.Series(nx.eigenvector_centrality(nx.from_numpy_matrix((A <= 4)
    .fillna(0).to_numpy()))).hvplot.hist(bins=100)

# %%
import hvplot.networkx as hvnx

# %%
hvnx.draw_kamada_kawai(nx.from_numpy_matrix((A < 4)
    .fillna(0).to_numpy()), edge_color='weight',  edge_width=hv.dim('weight')*1, node_size=1)

# %%
graph = nx.from_numpy_matrix(W.to_numpy())#where(lambda df: df > df.melt().value.quantile(0.5)).fillna(0).to_numpy())

# %%
centrality = pd.Series(nx.eigenvector_centrality(graph), name='Eigenvector Centrality')
centrality.hvplot.hist(bins=100, xlabel='Eigenvector Centrality', title='Histrogram of Eigenvector Centrality')

# %%
pos = nx.kamada_kawai_layout(graph)  # positions for all nodes
# hvnx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4, width=600, height=600) *\

hvnx.draw(graph, pos, edge_color='weight', edge_cmap='greys',
          edge_width=hv.dim('weight')*25, node_size=hv.dim('size')*20).opts(title='Eigenvector Centrality on Weibull Weighted Graph', colorbar=True) *\
hvnx.draw_networkx_nodes(graph, pos, #nodelist=list(p.keys()),
                         node_cmap='turbo',
                         node_size=80,
                         node_color=centrality.tolist(),
                         cmap='turbo',
                         colorbar=True,
                         logz=True).opts(colorbar=True, tools=['hover'])

# %%
A.melt().value.value_counts()

# %%
((A == 4) | (A == 6)).to_numpy().sum()

# %%
s = nx.from_numpy_matrix((A <= 6)
    .fillna(0).to_numpy())

# %%
# stats.pearsonr(pd.Series(nx.katz_centrality(graph, alpha=-1.5)), features.returns)

# %%
stats.pearsonr(pd.Series(nx.eigenvector_centrality(graph)), features.returns)

# %%
stats.spearmanr(pd.Series(nx.eigenvector_centrality(graph)), features.returns)

# %%
import hvplot.networkx as hvnx


# %%
communities = nx.algorithms.community.modularity_max.greedy_modularity_communities(graph)


# %%
c = list(communities)

# %%
tags = pd.Series({i: list(j) for  i, j in enumerate(communities)}).explode().sort_values()#.index.astype(str).tolist()

# %%
R.to_frame('returns').assign(community = tags.index).groupby('community').agg(['min', 'max', 'count'])

# %%
pos = nx.kamada_kawai_layout(graph)  # positions for all nodes
# hvnx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4, width=600, height=600) *\

hvnx.draw(graph, pos, edge_color='weight', edge_cmap='greys',
          edge_width=hv.dim('weight')*25, node_size=hv.dim('size')*20).opts(title='Communities on Weibull Weighted Graph', colorbar=True) *\
hvnx.draw_networkx_nodes(graph, pos, #nodelist=list(p.keys()),
                         node_cmap='Category20',
                         node_size=80,
                         node_color=tags.index.astype(str).tolist(),
                         cmap='Category20',
                         colorbr=True,
                         logz=True)

# %%

# %%
pos = nx.kamada_kawai_layout(graph)  # positions for all nodes
# hvnx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4, width=600, height=600) *\

hvnx.draw(graph, pos, edge_color='weight', edge_cmap='greys',
          edge_width=hv.dim('weight')*25, node_size=hv.dim('size')*20).opts(title='Returns on Weibull Weighted Graph', colorbar=True) *\
hvnx.draw_networkx_nodes(graph, pos, #nodelist=list(p.keys()),
                         node_cmap='turbo',
                         node_size=80,
                         node_color=R.tolist(),
                         cmap='turbo',
                         colorbar=True,
                         logz=True)

# %%
(F>0).mean()


# %%
def mask(Q, m = None):
    if m is not None:
        z = np.zeros_like(Q)
        z[m] = 1
        return z
    else:
        return np.ones_like(Q)


# %%
from operator import add
from functools import reduce

plots = []
sorted_values = np.argsort(np.abs(F))[::-1]
S = vec.T * F.reshape((-1, 1))

for k in range(6):
    k += 6

    m = sorted_values[k]

    top_component = pd.Series((vec.T @ (F * mask(F, m)).reshape((-1, 1))).flatten())#S[:, [m]].sum(1))

    pos = nx.kamada_kawai_layout(graph)  # positions for all nodes
    # hvnx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4, width=600, height=600) *\

    component = (hvnx.draw(graph, pos, edge_color='weight', edge_cmap='greys',
              edge_width=hv.dim('weight')*25, node_size=hv.dim('size')*20).opts(title=f'Eigenvector λ={np.round(vals[m], 3)} on Weibull Weighted Graph', colorbar=True) *\
    hvnx.draw_networkx_nodes(graph, pos, #nodelist=list(p.keys()),
                             node_cmap='turbo',
                             node_size=80,
                             node_color=top_component.tolist(),
                             cmap='turbo',
                             colorbar=True,
                             logz=True))

    plots.append(component)
    
reduce(add, plots).cols(2)

# %%
sorted_values[:10]

# %%
top_component = pd.Series((vec.T @ (F * mask(F, sorted_values[:10].tolist())).reshape((-1, 1))).flatten())

pos = nx.kamada_kawai_layout(graph)  # positions for all nodes
# hvnx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4, width=600, height=600) *\

component = (hvnx.draw(graph, pos, edge_color='weight', edge_cmap='greys',
          edge_width=hv.dim('weight')*25, node_size=hv.dim('size')*20).opts(title=f'Top ten Graph Fourier components', colorbar=True) *\
hvnx.draw_networkx_nodes(graph, pos, #nodelist=list(p.keys()),
                         node_cmap='turbo',
                         node_size=80,
                         node_color=top_component.tolist(),
                         cmap='turbo',
                         colorbar=True,
                         logz=True))

component

# %%
plots = []
sorted_values = np.argsort(np.abs(vals))
# S = vec.T * F.reshape((-1, 1))

for k in range(6):
    m = sorted_values[k]

    top_component = pd.Series((vec.T @ (F * mask(F, m)).reshape((-1, 1))).flatten())#S[:, [m]].sum(1))

    pos = nx.kamada_kawai_layout(graph)  # positions for all nodes
    # hvnx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4, width=600, height=600) *\

    component = (hvnx.draw(graph, pos, edge_color='weight', edge_cmap='greys',
              edge_width=hv.dim('weight')*25, node_size=hv.dim('size')*20).opts(title=f'Eigenvector λ={np.round(vals[m], 3)} on Weibull Weighted Graph', colorbar=True) *\
    hvnx.draw_networkx_nodes(graph, pos, #nodelist=list(p.keys()),
                             node_cmap='turbo',
                             node_size=80,
                             node_color=top_component.tolist(),
                             cmap='turbo',
                             colorbar=True,
                             logz=True))

    plots.append(component)
    
reduce(add, plots).cols(2)

# %%
pd.DataFrame(Z[:, :2], columns=['Component 1', 'Component 2'], index=factors.index).assign(returns=factors.returns).hvplot.scatter(x='Component 1', y='Component 2', color='returns', cmap='turbo', title='Returns over Eigenvectors of the Weibull Kernel Weighted Laplacian')#.opts(xrotation=90)

# %%
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

# %%
v

# %%
pd.Series(W.to_numpy().flatten()).hvplot.hist()

# %%
Q = pairwise_distances(Z[:, [1]])
Q.std()

# %%
Q[take, :][:, take].std()

# %%
E = Q.reshape(-1, 1)

# %%
pos = E[E > 0.]
pos.sort()
limits = pos[-5]

# %%
limits

# %%
take = ~ (Q > limits).any(1)

# %%
K = stats.norm(scale=0.002).pdf(pairwise_distances(Z[:, :2]))
np.fill_diagonal(K, 0)

# %%
K = (K / K.sum(1)).T

# %%
from pysal import explore
import pysal as ps

# %%
u = np.nan_to_num(K, 0)[take, :][:, take]
u[(u.sum(1) == 0), :] = (u.shape[0]-1)**-1
u = pd.DataFrame(u).apply(lambda df: df / df.sum(), axis=1).to_numpy()
np.fill_diagonal(u, 0)

# %%
from statsmodels.regression.linear_model import OLS

# %%
wy =  stats.mstats.winsorize(y, [0.028, 0.062])

# %%
m = OLS(wy[take], pd.DataFrame(u @ wy[take], index=y.iloc[take].index)).fit()

# %%
m.summary()

# %%
w = ps.lib.weights.full2W(u)


# %%
mi = explore.esda.Moran(wy[take], w, permutations=9999)
mi.I, mi.p_sim

# %%
y.loc[~take]

# %%
gc = explore.esda.Geary(wy[take], w, permutations=9999)
gc.C, gc.p_sim

# %%
g = explore.esda.G(wy[take], w, permutations=999)
g.G, g.p_sim

# %%
j = features.drop(columns=['alpha', 'returns'])

# %%

# %%
print(j.apply(lambda s: stats.pearsonr(np.array(W@s), s)[0]).rename('Pearson Correlation Coefficient').to_frame().join(j.apply(lambda s: stats.pearsonr(np.array(wy), s)[1]).rename('p-value')).to_latex())

# %%
f = features.drop(columns=['alpha', 'returns',  'price_to_earnings', 'price_to_research', 'market_capitalization'])#.to_numpy()

# %%
sm = ps.model.spreg.OLS(np.array(wy[take].reshape(-1, 1)), f.loc[take, :].to_numpy(), w=w, name_x=f.columns.tolist(), spat_diag=True, robust='white', white_test=True)

# %%
print(sm.summary)

# %%
K.sum(1)

# %%
pd.Series(K.flatten()).value_counts()

# %%
lag = ps.model.spreg.GM_Lag(np.array(wy[take].reshape(-1, 1)), f.loc[take, :].to_numpy(), w=w, name_x=f.columns.tolist())#, spat_diag=True)
print(lag.summary)

# %%

lag = ps.model.spreg.ML_Error(np.array(wy[take].reshape(-1, 1)), f.loc[take, :].to_numpy(), w=w, name_x=f.columns.tolist())#, spat_diag=True)
print(lag.summary)

# %%
lag = ps.model.spreg.GM_Error(np.array(wy[take].reshape(-1, 1)), f.loc[take, :].to_numpy(), w=w, name_x=f.columns.tolist())#, spat_diag=True)
print(lag.summary)

# %%
