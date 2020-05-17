#%%
import holoviews as hv
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysal as ps
import seaborn as sns
from pysal import explore
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS

hv.extension("bokeh")

#%%
# matched entities
matched = context.catalog.load("iex_matched_entities")  # scores_matches)

distances = context.io.load("paradise_distances")
price = context.io.load("paradise_price")
indices = context.io.load("indices")

# %%
index = (
    matched.groupby("symbol")
    .apply(lambda df: df.sample(1))
    .set_index("symbol")
    .exchange
)

rf_rates = (
    index.rename("index")
    .to_frame()
    .merge(
        indices.pct_change()
        .add(1)
        .cumprod()
        .tail(1)
        .subtract(1)
        .T.rename(columns=lambda x: "rf"),
        left_on="index",
        right_index=True,
    )
    .rf
)

# %%
returns = (
    price.loc[:, pd.IndexSlice[:, "close"]]
    .pct_change()
    .add(1)
    .cumprod()
    .tail(1)
    .subtract(1)
)

returns.columns = returns.columns.droplevel(1)
returns = returns.T.rename(columns=lambda x: "returns").dropna()
returns["excess"] = returns["returns"].subtract(rf_rates.loc[returns.index])
returns["rf"] = rf_rates.loc[returns.index].to_frame().to_numpy()

inner_index = returns.join(distances, how="inner").index

#%%
renamed_distances = distances.loc[inner_index, inner_index]
returns = returns.loc[inner_index, :]

# Binary connectedness
#%%
filled_distances = (~renamed_distances.isna()).to_numpy().astype(np.float)

np.fill_diagonal(filled_distances, 0)
w = ps.lib.weights.full2W(filled_distances)

# %%
mi = explore.esda.Moran(returns["excess"], w, transformation="B")
mi.I, mi.p_sim

#%%
sns.kdeplot(mi.sim, shade=True)
plt.vlines(mi.sim, 0, 0.5)
plt.vlines(mi.I, 0, 10, "r")
plt.xlim([-0.15, 0.15])
plt.show()

# %%
gc = explore.esda.Geary(returns["excess"], w, transformation="B")
gc.C, gc.p_sim

# %%
lm = explore.esda.Moran_Local(returns["excess"], w, transformation="B", permutations=99)
lm.p_z_sim
lm.q

#%%
returns["w_excess"] = ps.model.spreg.lag_spatial(w, returns["excess"].to_numpy())

# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot values
sns.regplot(x="excess", y="w_excess", data=returns)
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
m1 = ps.model.spreg.OLS(
    returns.loc[:, ["returns"]].to_numpy(),
    returns.loc[:, ["rf"]].to_numpy(),
    w=w,
    spat_diag=True,
    name_x=["beta"],
    name_y="r-rf",
    moran=True,
)

print(m1.summary)

# continues connectedness
# %%
filled_distances = (renamed_distances.fillna(0)).to_numpy()
w = ps.lib.weights.full2W(filled_distances)

#%%
mi = explore.esda.Moran(returns["excess"], w, transformation="r")
mi.I, mi.p_sim

# %%
gc = explore.esda.Geary(returns["excess"], w, transformation="r")
gc.C, gc.p_sim


#%%
sns.kdeplot(mi.sim, shade=True)
plt.vlines(mi.sim, 0, 0.5)
plt.vlines(mi.I, 0, 10, "r")
plt.xlim([-0.15, 0.15])
plt.show()

# %%
returns["w_excess"] = ps.model.spreg.lag_spatial(w, returns["excess"].to_numpy())

# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot values
sns.regplot(x="excess", y="w_excess", data=returns)
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
m1 = ps.model.spreg.OLS(
    returns.loc[:, ["returns"]].to_numpy(),
    returns.loc[:, ["rf"]].to_numpy(),
    w=w,
    spat_diag=True,
    name_x=["beta"],
    name_y="r-rf",
)

print(m1.summary)
