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
# %%
import warnings
from functools import partial

import holoviews as hv
import hvplot.pandas  # noqa
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from sklearn.manifold import Isomap

tqdm().pandas()
hv.extension("bokeh")

#%%
# paradise_entities = context.io.load("paradise_nodes_entity")
# paradise_nodes_address = context.io.load("paradise_nodes_address")
# paradise_nodes_intermediary = context.io.load("paradise_nodes_intermediary")
# paradise_nodes_officer = context.io.load("paradise_nodes_officer")
# paradise_nodes_other = context.io.load("paradise_nodes_other")
paradise_edges = context.io.load("paradise_edges")

# %%
paradise_graph = nx.convert_matrix.from_pandas_edgelist(
    df=paradise_edges,
    source="START_ID",
    target="END_ID",
    edge_attr=paradise_edges.columns.drop(["START_ID", "END_ID"]).tolist(),
)


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
n = np.random.choice(np.array(paradise_graph.nodes), 100000)
m = np.random.choice(np.array(paradise_graph.nodes), 100000)

# %%
l = []
for i, j in zip(m, n):
    try:
        l.append(nx.shortest_path_length(paradise_graph, i, j))
    except:
        pass

# %%
np.mean(l)

# %%
np.var(l)

# %%
pd.Series(l).to_csv('random_shortest_lengths.csv')

# %%
o, d  = list(zip(*[(node, val) for (node, val) in paradise_graph.degree()]))

# %%
p = np.random.choice(np.array(o), size=10000, p = d/np.sum(d))
q = np.random.choice(np.array(o), size=10000, p = d/np.sum(d))

# %%
k = []
for i, j in zip(p, q):
    try:
        k.append(nx.shortest_path_length(paradise_graph, i, j))
    except:
        pass

# %%
np.mean(k)

# %%
np.var(k)

# %%
pd.Series(k).to_csv('non_random_shortest_lengths.csv')

# %%
from scipy import stats

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
e = A.melt().value.dropna()

# %%
pd.Series(k,  name='value').hvplot.hist(label='Degree-sampled ICIJ Graph', title='Histogram of Shortest Path Lengths', xlabel='Shortest Path Lengths', ylabel='Count', bins=35, alpha=0.5)

# %%
e.shape

# %%
( e.hvplot.hist(label='Matched List Companies', xlabel='Shortest Path Lengths', ylabel='Count', alpha=0.5) *\
    pd.Series(k,  name='value').sample(e.shape[0], replace=True).hvplot.hist(label='Degree-sampled ICIJ Graph', color='green', title='Histogram of Shortest Path Lengths', xlabel='Shortest Path Lengths', ylabel='Count', bins=35, alpha=0.5) *\
 pd.Series(l,  name='value').sample(e.shape[0], replace=True).hvplot.hist(label='Randomly-sampled ICIJ Graph', color='orange', title='Histogram of Shortest Path Lengths', xlabel='Shortest Path Lengths', ylabel='Count', bins=35, alpha=0.5)
 ).opts(show_legend=True)

# %%
d = A.melt().value.dropna()

# %%
dist = stats.weibull_min(*stats.weibull_min.fit(e))

# %%
true_dist = stats.weibull_min(*stats.weibull_min.fit(l))

# %%
non_true_dist = stats.weibull_min(*stats.weibull_min.fit(k, scale=np.var(k)))

# %%
stats.weibull_min.fit(k, scale=np.var(k))

# %%
stats.weibull_min.fit(k)

# %%
pd.DataFrame({'k': e, 'v': dist.pdf(e)}).sort_values('k').hvplot.line('k', 'v', label='Matched List Companies', title='Estimated Weibull Distribution', xlabel='Shortest Path Length', ylabel='Probability Density') *\
pd.DataFrame({'k': e, 'v': non_true_dist.pdf(e)}).sort_values('k').hvplot.line('k', 'v', label='Degree-sampled ICIJ Graph', color='green', title='Estimated Weibull Distribution', xlabel='Shortest Path Length', ylabel='Probability Density') *\
pd.DataFrame({'k': e, 'v': true_dist.pdf(e)}).sort_values('k').hvplot.line('k', 'v', label='Randomly-sampled ICIJ Graph', color='orange', title='Estimated Weibull Distribution', xlabel='Shortest Path Length', ylabel='Probability Density')

# %%
centrality = nx.eigenvector_centrality(paradise_graph, tol=1e-6, max_iter=1500)

# %%
degree = pd.Series({k: v for k,v in paradise_graph.degree()})

# %%

# %%
degree.to_frame('Degree').assign(**{'Matched Listed Entities': lambda df: df.index.isin(matched.node_id)}).dropna().where(lambda df: df.Degree < 500).dropna().hvplot.hist(title='Degree', by='Matched Listed Entities', alpha=0.5, bins=100)

# %%
print(degree.loc[matched.node_id].replace(0, np.nan).describe().to_frame('Matched Listed Companies').assign(**{'non-Matched Listed Companies': degree.drop(index=matched.node_id).replace(0, np.nan).describe()}).to_latex())

# %%
c = pd.Series(centrality)

# %%
c.quantile(0.1)

# %%
c.to_frame('Eigenvector Centrality').assign(**{'Matched Listed Entities': lambda df: df.index.isin(matched.node_id)}).dropna().hvplot.hist(title='Eigenvector Centrality', by='Matched Listed Entities', alpha=0.5, bins=100)

# %%
c.loc[matched.node_id].replace(0, np.nan).dropna().hvplot.hist(label='Matched Listed Entities', alpha=0.5, bins=100) * c.drop(matched.node_id).replace(0, np.nan).dropna().hvplot.hist(label='non-Matched Listed Entities', title='Eigenvector Centrality', alpha=0.5, bins=100)

# %%
print(c.loc[matched.node_id].describe().to_frame('Matched Listed Companies').assign(**{'non-Matched Listed Companies':c.drop(index=matched.node_id).describe()}).to_latex())

# %%
c.loc[matched.node_id].median()

# %%
c.drop(index=matched.node_id).median()

# %% [markdown]
# ___

# %%
A = nx.to_scipy_sparse_matrix(paradise_graph).asfptype()

# %%
1 / (A.sum(0).mean()/A.shape[0])

# %%
pd.Series(np.array(A.sum(0)).flatten()).clip(0, 100).value_counts()

# %%
observed = pd.Series(np.array(A.sum(0)).flatten()).clip(0, 100).hvplot.hist(label='Observed Histogram of Node Degree', title='Histogram comparing of Observed and Log-Normal Distributed of Node Degree', xlabel='Node Degree', bins=100)
observed

# %%
d = stats.lognorm(*stats.lognorm.fit(pd.Series(np.array(A.sum(0)).flatten()).clip(0, 100)))

# %%
k = pd.Series(np.array(A.sum(0)).flatten()).clip(0, 100).mean()

# %%
from scipy import stats

# %%
theoretical = pd.Series((d.rvs(A.shape[0]))).where(lambda df: df < 100).hvplot.hist(label='Theoretical Histogram of Node Degree', xlabel='Node Degree', bins=100, color='orange')

# %%
theoretical

# %%
observed * theoretical

# %%
observed + theoretical.opts(color='orange')

# %%
membership.sum()

# %%
from scipy.sparse import diags
D = diags(np.array(A.sum(0)).flatten(), 0)
D.shape

# %%
L = (D - A).astype('f')
L.dtype

# %%
# del A
# del D
# # del paradise_entities
# # del paradise_nodes_address
# # del paradise_nodes_intermediary
# # del paradise_nodes_officer
# # del paradise_nodes_other
# del paradise_edges

# %%
from scipy.sparse.linalg import eigsh
vals, vecs = eigsh(L, k=600)#, sigma=0)#, maxiter=1)#500, tol=1e-3)#which='SM')

vecs.shape

# %%
# del L

# %%
U, e = np.conjugate(vecs), vals

# %%
U.shape

# %%
matched_entities = context.catalog.load("iex_matched_entities")

# %%
membership = np.array([1 if f in matched_entities.node_id.tolist() else 0 for f in list(paradise_graph.nodes)]).reshape((-1, 1))

# %%
gft = np.tensordot(U, membership, ([0], [0]))
true = pd.DataFrame({'Eigenvalues (Frequency)': vals.real, 'Magnitudes': np.abs(gft).flatten()})

# %%
true.sort_values('Magnitudes', ascending=False)

# %%
hv.extension('bokeh')

# %%
true.hvplot.scatter(x='Eigenvalues (Frequency)', y='Magnitudes', height=600, width=1200, logx=True, color='orange')

# %%
signal = membership.copy()

samples = 1000
R = np.ones((samples, U.shape[1]))
for i in range(10):
    pd.np.random.shuffle(signal)
    rgft = np.tensordot(U, signal, ([0], [0]))
    R[i, :] =  np.abs(rgft).flatten()

# %%
I = pd.DataFrame(R.T).assign(**{'Eigenvalues (Frequency)': vals.real})

# %%
I

# %%
I.max(0).median()

# %%
I.set_index('Eigenvalues (Frequency)').idxmax(0).std()

# %%
true.sort_values('Magnitudes', ascending=False)

# %%
pd.DataFrame(R.T).assign(**{'Eigenvalues (Frequency)': vals.real}).melt(id_vars='Eigenvalues (Frequency)', value_name='Magnitudes').hvplot.scatter(x='Eigenvalues (Frequency)', y='Magnitudes',  color='blue', label='Simualted', height=600, width=1200, alpha=0.25, size=1)*\
true.hvplot.scatter(x='Eigenvalues (Frequency)', y='Magnitudes', height=600, width=1200, logx=True, alpha=0.5, size=5, label='Observed')

# %%
true.hvplot.scatter(x='Eigenvalues (Frequency)', y='Magnitudes', height=600, width=1200, color='orange', logx=True)


# %%
def phase(z): # Calculates the phase of a complex number
    return (z / np.abs(z)).real


# %%
pd.Series(phase(np.array(gft).flatten())).hvplot.line(title='Phase')

# %%
pd.Series(vals.flatten().real).hvplot.line(title='Eigen Values')

# %%
pd.Series(np.abs(gft).flatten()).hvplot.line(title='Magnitude')

# %%
pd.Series(np.angle(gft).flatten()).hvplot.line(title='Angle')
