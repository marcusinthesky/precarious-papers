#%%
import holoviews as hv
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysal as ps
import seaborn as sns
from pysal import explore
from scipy.stats import norm, poisson
from scipy import stats
from statsmodels.regression.linear_model import OLS

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

graph = context.catalog.load("paradise_graph")
nodes = pd.concat(
    [
        context.io.load("paradise_nodes_entity"),
        context.io.load("paradise_nodes_address"),
        context.io.load("paradise_nodes_intermediary"),
        context.io.load("paradise_nodes_officer"),
        context.io.load("paradise_nodes_other"),
    ],
    axis=0,
).set_index("node_id")


target_degree = 4

double_dutch_paths = (
    distances.reset_index()
    .rename(columns={"symbol": "from"})
    .melt(id_vars=["from"], var_name="to", value_name="degree")
    .where(lambda s: s.degree == target_degree)
    .dropna(how="all")
    .replace(matched.set_index("symbol").node_id)
)

import networkx as nx


def shortest_path(G, source, target):
    try:
        return nx.shortest_path(G, source, target)
    except:
        return pd.np.nan


paths = double_dutch_paths.assign(
    path=lambda df: df.apply(
        lambda s: shortest_path(graph, source=s["from"], target=s["to"]), "columns"
    )
)

country_steps = (
    paths.path.dropna()
    .apply(lambda s: [int(i) for i in s])
    .apply(lambda s: nodes.loc[s, "countries"].reset_index(drop=True))
)


counters = []
for i in range(target_degree):
    counters.append(
        country_steps.iloc[:, [i, i + 1]]
        .rename(columns={i: "start", (i + 1): "end"})
        .assign(
            start=lambda s: s.start.apply(lambda x: f"{x} [{i}]"),
            end=lambda s: s.end.apply(lambda x: f"{x} [{i+1}]"),
        )
        .assign(count=1)
        .groupby(["start", "end"])
        .count()
    )
links = pd.concat(counters).reset_index()

hv.Sankey(links).opts(title=f"{target_degree} Degree Country Paths")

entity_steps = (
    paths.path.dropna()
    .apply(lambda s: [int(i) for i in s])
    .apply(lambda s: nodes.loc[s, "name"].reset_index(drop=True))
)
