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
paradise_entities = context.io.load("paradise_nodes_entity")
paradise_nodes_address = context.io.load("paradise_nodes_address")
paradise_nodes_intermediary = context.io.load("paradise_nodes_intermediary")
paradise_nodes_officer = context.io.load("paradise_nodes_officer")
paradise_nodes_other = context.io.load("paradise_nodes_other")
paradise_edges = context.io.load("paradise_edges")

# %%
paradise_graph = nx.convert_matrix.from_pandas_edgelist(
    df=paradise_edges,
    source="START_ID",
    target="END_ID",
    edge_attr=paradise_edges.columns.drop(["START_ID", "END_ID"]).tolist(),
)

# density
nx.classes.function.density(paradise_graph)

# degree
(
    hv.Curve(nx.classes.function.degree_histogram(paradise_graph)).opts(
        title="Degree Histogram", ylabel="Count", xlabel="Degree"
    )
)

average_degree = float(sum(dict(paradise_graph.degree()).values())) / float(
    paradise_graph.number_of_nodes()
)


#%%
context.catalog.save("paradise_graph", paradise_graph)
# %%
iex_matched_entities = context.catalog.load("iex_matched_entities")
filtered_iex_matched_entities = iex_matched_entities.where(
    lambda df: df.score == 100
).dropna(how="all")

matched_entities = paradise_entities.merge(
    filtered_iex_matched_entities.drop(columns=["name"]),
    left_on="name",
    right_on="entities",
    how="inner",
)

# %%
def find_path_length(source, target, G):
    try:
        return nx.shortest_path_length(G, source.item(), target.item())
    except nx.exception.NetworkXNoPath:
        warnings.warn("No path found")
        return np.nan


# %%
D = pairwise_distances(
    X=(matched_entities.node_id.to_numpy().reshape(-1, 1)),
    metric=find_path_length,
    n_jobs=-1,
    G=paradise_graph,
)

pairwise_distances = (
    pd.DataFrame(D, columns=matched_entities.node_id, index=matched_entities.node_id)
    .T.drop_duplicates()
    .T.drop_duplicates()
)

context.catalog.save("paradise_distances", pairwise_distances)


# %%
Z = Isomap(2, metric="precomputed").fit_transform(
    pairwise_distances.fillna(pairwise_distances.max().max())
)
components = ["Component 1", "Component 2"]
isomap_plot = (
    pd.DataFrame(Z, columns=components)
    .assign(name=named_entities.name.to_numpy())
    .hvplot.scatter(
        x=components[0],
        y=components[1],
        color="name",
        width=800,
        height=600,
        legend=False,
        title="IsoMap",
    )
    .opts(tools=["hover"])
)

isomap_plot_text = hv.Labels(
    pd.DataFrame(Z, columns=components).assign(name=named_entities.name.to_numpy()),
    kdims=components,
    vdims="name",
).opts(text_font_size="6pt", width=800, height=600, title="IsoMap")

context.catalog.save("paradise_isomap", isomap_plot)
