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
        title="Degree Histogram", ylabel="Count", xlabel="Degree", logx=True
    )
)

average_degree = float(sum(dict(paradise_graph.degree()).values())) / float(
    paradise_graph.number_of_nodes()
)
average_degree
# 2.9832619059417067


#%%
context.catalog.save("paradise_graph", paradise_graph)
# %%
matched_entities = context.catalog.load("iex_matched_entities")

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

distances = (
    pd.DataFrame(D, columns=matched_entities.symbol, index=matched_entities.symbol)
    .T.drop_duplicates()
    .T.drop_duplicates()
    .reset_index()
    .groupby("symbol")
    .min()  # .apply(lambda df: df.sample(1)).set_index('symbol')
    .T.reset_index()
    .groupby("symbol")
    .min()  # .apply(lambda df: df.sample(1)).set_index('symbol')
)

context.catalog.save("paradise_distances", distances)


# %%
Z = Isomap(2, metric="precomputed").fit_transform(
    distances.fillna(distances.max().max())
)
entity_names = (
    matched_entities.groupby("symbol")
    .apply(lambda df: df.sample(1))
    .set_index("symbol")
    .loc[distances.index, ["name", "exchange"]]
)
components = ["Component 1", "Component 2"]
isomap_plot = (
    pd.DataFrame(Z, columns=components, index=entity_names.index)
    .join(entity_names)
    .hvplot.scatter(
        x=components[0],
        y=components[1],
        color="exchange",
        width=800,
        height=600,
        legend=True,
        title="IsoMap",
    )
    .opts(tools=["hover"])
)

isomap_plot_text = hv.Labels(
    pd.DataFrame(Z, columns=components, index=entity_names.index).join(entity_names),
    kdims=components,
    vdims=["name"],
).opts(text_font_size="6pt", width=800, height=600, title="IsoMap")

context.catalog.save("paradise_isomap", isomap_plot)


# %%
