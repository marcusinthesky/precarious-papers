#%%
import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
import holoviews as hv
import matplotlib.pyplot as plt
import seaborn as sns


hv.extension("bokeh")

#%%
# matched entities
paradise_entities = context.io.load("paradise_nodes_entity")
iex_matched_entities = context.catalog.load("iex_matched_entities")
distances = context.io.load("paradise_distances")
price = context.io.load("paradise_price")


# %%
symbols = (
    paradise_entities.merge(
        iex_matched_entities.drop(columns=["name"]), left_on="name", right_on="entities"
    )
    .set_index("node_id")
    .drop_duplicates()
    .symbol
)

renamed_distances = (
    distances.rename(index=symbols, columns=symbols)
    .reset_index()
    .melt(id_vars="node_id", var_name="node_id_col", value_name="distance")
    .groupby(["node_id", "node_id_col"])
    .mean()
    .reset_index()
    .pivot(index="node_id", columns="node_id_col", values="distance")
)

# %%
returns = price.loc[:, pd.IndexSlice[:, "close"]].pct_change().cumprod().tail(1)

returns.columns = returns.columns.droplevel(1)
returns = returns.T.rename(columns=lambda x: "returns").dropna()

# %%
import pysal as ps
from pysal import explore

filled_distances = (
    renamed_distances.drop_duplicates()
    .T.drop_duplicates()
    .loc[returns.index, returns.index]
    .pipe(lambda df: df.fillna(df.max().max()))
).to_numpy()
w = ps.lib.weights.full2W(filled_distances)
mi = explore.esda.Moran(returns["returns"], w)

mi.I, mi.p_sim

sns.kdeplot(mi.sim, shade=True)
plt.vlines(mi.sim, 0, 0.5)
plt.vlines(mi.I, 0, 10, "r")
plt.xlim([-0.15, 0.15])


# %%
js = returns
js["returns_std"] = (js["returns"] - js["returns"].mean()) / js["returns"].std()
js["w_returns_std"] = ps.model.spreg.lag_spatial(w, js["returns_std"].to_numpy())


# %%
mi_std = explore.esda.Moran(returns["returns_std"], w)
mi_std.p_sim

# %%
# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot values
sns.regplot(x="returns_std", y="w_returns_std", data=js)
# Add vertical and horizontal lines
plt.axvline(0, c="k", alpha=0.5)
plt.axhline(0, c="k", alpha=0.5)
ax.set_xlim(-2, 7)
ax.set_ylim(-2.5, 2.5)
plt.text(3, 1.5, "HH", fontsize=25)
plt.text(3, -1.5, "HL", fontsize=25)
plt.text(-1, 1.5, "LH", fontsize=25)
plt.text(-1, -1.5, "LL", fontsize=25)
# Display
plt.show()


# %%
gy = explore.esda.Geary(returns["returns_std"], w)
gy.p_sim

# %%
g = explore.esda.G(returns["returns_std"], w)
g.p_sim

# %%
gamma = explore.esda.Gamma(returns["returns_std"], w)
gamma.p_sim

# %%
