# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from functools import partial, reduce
from operator import mul, add
from toolz.curried import pipe, accumulate, map, take, compose_left, sliding_window
import numpy as np
from scipy import stats
import hvplot.pandas  # noqa
import holoviews as hv
import networkx as nx
import hvplot.networkx as hvnx
import numba


hv.extension("bokeh")

# %%
from tqdm.auto import tqdm

tqdm.pandas()

# %%
T = 1000
N = 100
W = 100
P = 0.05

S = 1000


# %%
# def graph(t=T, n=N):
#     t +=1
#     a = np.random.uniform(0., 1., size=(t, n, n)).cumsum(0)
#     b = np.clip(a, 0, 1) # slip where the cum average exceeded the bounds
#     c = np.diff(a, n=1, axis=0, prepend=np.zeros(shape=(1, n, n))) # this gives us the values in excess of the clip
#     d = (b - c) # now it looks like it gets stuck of bounded off the bounds
#     e = d < P
#     return np.nan_to_num(e / e.sum(1).reshape(t, 1, n))[1:, :, :]

# %%
# def random_renyi_matrix():
#     return pipe(random_renyi(N, 0.1),
#                 nx.to_numpy_matrix,
#                 partial(np.expand_dims, axis=0))

# def moving_average(x, w=10):
#     return np.convolve(x, np.ones(w), 'valid') / w

# def random_renyi(n, p):
#     G = nx.fast_gnp_random_graph(n, p)
#     for (u,v,w) in G.edges(data=True):
#         w['weight'] = np.random.uniform()
#     return G

# %%
@numba.jit
def gaps(t=T, w=W):
    return np.diff(
        np.sort(
            np.append(0, np.append(np.random.choice(np.arange(1, t), t // w - 1), t))
        )
    )


@numba.jit
def get_mat(n=N):
    a = np.tril(np.random.uniform(0.0, 1.0, size=(n, n)), -1)
    return np.expand_dims(a + a.T, 0)


def smooth(x, y, w=W):
    return np.linspace(start=x, stop=y, num=w, axis=1)[0]


def graph(t=T, n=N):
    return (
        np.concatenate(
            [
                smooth(a, b, w)
                for (a, b), w in zip(
                    sliding_window(2, [get_mat(n) for _ in range(t)]), gaps(T, W)
                )
            ]
        )
        < P
    )


# %%
# %%timeit
graph()

# %%
# G = compose_left(nx.fast_gnp_random_graph,
#                  nx.to_numpy_matrix,
#                  lambda x: np.random.uniform(0, x, size=x.shape),
#                  partial(np.expand_dims, axis=0)
#         )

# D = partial(G, n=N, p=P)

# def smooth(x, y, num=W):
#     return np.linspace(start=x, stop=y, num=num, axis=1)[0]

# def graph():
#     G_ =  pipe([D() for _ in range((T // W)+1)],
#                  sliding_window(2),
#                  map(lambda z: smooth(*z)),
#                  list,
#                  np.concatenate,
#                  np.round)


#     return np.nan_to_num(G_ / G_.sum(1).reshape(T, 1, N))

# %%
def sample(t=T, n=N):
    e = np.random.normal(size=(t, n))
    rm_rf = np.random.normal(size=(t, 1))
    b = np.random.normal(loc=1, size=(1, n))
    ri = rm_rf @ b + e

    return ri


# %%
# %%timeit
sample()

# %%
hv.output(max_frames=1000)

graphs = pipe(
    graph()[:100, :, :],
    map(nx.from_numpy_matrix),
    map(partial(hvnx.draw_spring, node_size=10, iterations=1)),
    list,
    enumerate,
    dict,
)

hmap = hv.HoloMap(graphs, kdims="time")
hv.save(hmap, filename="tmp.gif", holomap="gif", fps=6)
hmap


# %%
def spatial_corr(ri):
    G = graph()
    ri_g = np.matmul(ri.reshape(T, 1, N), G).reshape(T, N) + ri

    return ri_g


# %%
# %%timeit
spatial_corr(sample())

# %%
samples_ = (
    pd.DataFrame(np.ones((S, 1)), columns=["ri"])
    .progress_applymap(lambda df: sample())
    .assign(ri_g=lambda df: df.progress_applymap(spatial_corr))
    .progress_applymap(pd.DataFrame)
)

# %%
samples_

# %%
(
    samples_.progress_applymap(lambda x: x.sum(1).mean()).hvplot.hist(
        title="Mean under Evolving Graph Spatial Correlation", xlabel="Mean"
    )
)

# %%
(
    samples_.progress_applymap(lambda x: x.sum(1).std()).hvplot.hist(
        title="Std under Evolving Graph Spatial Correlation", xlabel="Std"
    )
)

# %%
(
    samples_.progress_applymap(lambda x: x.sum(1).skew()).hvplot.hist(
        title="Skewness under Evolving Graph Spatial Correlation", xlabel="Skewness"
    )
)

# %%
(
    samples_.progress_applymap(lambda x: x.sum(1).kurt()).hvplot.hist(
        title="Kurtosis under Evolving Graph Spatial Correlation", xlabel="Kurtosis"
    )
)

# %%
s = samples_.sample()

(
    s.iloc[0, 0]
    .loc[:, 32]
    .rename("r_i")
    .to_frame()
    .join(s.iloc[0, 1].loc[:, 32].rename("r_i_G"))
    .reset_index()
    .rename(columns={"index": "Time"})
    .melt(id_vars="Time", var_name="type", value_name="Return")
    .hvplot.line(
        x="Time", by="type", title="Sample Graph Spatially Correlated Share", alpha=0.5
    )
)

# %%
from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift
from sklearn.mixture import BayesianGaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE


class TSNETransfomer(TSNE):
    def transform(self, X):
        return self.fit_transform(X)


class DBSCANTransformer(DBSCAN):
    def transform(self, X):
        return X


def clusters(z, algorithm):
    return len(np.unique(algorithm.fit_predict(z)))


# %%
meanshift = MeanShift(n_jobs=-1)

# %%
states_ = samples_.head(10).progress_applymap(partial(clusters, algorithm=meanshift))
states_.hvplot.hist(
    title="Number of MeanShift Market States", xlabel="Count of Discovered States"
)

# %%
