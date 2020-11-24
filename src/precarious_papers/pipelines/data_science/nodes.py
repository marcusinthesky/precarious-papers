# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name
from functools import reduce
from operator import add
from typing import Dict, List, Optional, Tuple

import holoviews as hv
import hvplot.networkx as hvnx
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pysal as ps
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.regression.linear_model import OLS, OLSResults

hv.extension("bokeh")


def get_spatial_statistics(spatial_model: ps.model.spreg.ols.OLS) -> pd.Series:
    """Extracts and formals estimates and statistics from spatial model

    :param spatial_model: Estimated pysal non-spatial ols model
    :type spatial_model: ps.model.spreg.ols.OLS
    :return: Model estimates and statistics
    :rtype: pd.Series
    """
    statistics: pd.Series = pd.Series(
        {
            "F-statistic": "{} ({})".format(
                round(spatial_model.f_stat[0], 3), round(spatial_model.f_stat[1], 3)
            ),
            "Jarque-Bera": "{} ({})".format(
                round(spatial_model.jarque_bera["jb"], 3),
                round(spatial_model.jarque_bera["pvalue"], 3),
            ),
            "Breusch-Pagan": "{} ({})".format(
                round(spatial_model.breusch_pagan["bp"], 3),
                round(spatial_model.breusch_pagan["pvalue"], 3),
            ),
            "Koenker-Basset": "{} ({})".format(
                round(spatial_model.koenker_bassett["kb"], 3),
                round(spatial_model.koenker_bassett["pvalue"], 3),
            ),
            "White": "{} ({})".format(
                round(spatial_model.white["wh"], 3),
                round(spatial_model.white["pvalue"], 3),
            ),
            "Moran’s I (error)": "{} ({})".format(
                round(spatial_model.moran_res[-2], 3),
                round(spatial_model.moran_res[-1], 3),
            ),
            "Robust LM (lag)": "{} ({})".format(
                round(spatial_model.rlm_error[-2], 3),
                round(spatial_model.rlm_error[-1], 3),
            ),
            "Robust LM (error)": "{} ({})".format(
                round(spatial_model.rlm_lag[-2], 3), round(spatial_model.rlm_lag[-1], 3)
            ),
        }
    )

    return statistics


def backwards_selection(
    X: pd.DataFrame, y: pd.DataFrame, W: pd.DataFrame
) -> pd.DataFrame:
    """Formulate model estimates as tabular results


    :param X: Firm characteristics
    :type X: pd.DataFrame
    :param y: Firm returns over event window
    :type y: pd.DataFrame
    :param W: spatial weighting matrix
    :type W: pd.DataFrame
    :return: Estimates and statistics from backwards selection procedure
    :rtype: pd.DataFrame
    """
    w: ps.lib.weights.weights.W = ps.lib.weights.full2W(W.to_numpy())

    coefficients: List = []
    statistics: List = []
    for i in range(X.shape[1]):

        spatial_model: ps.model.spreg.ols.OLS = ps.model.spreg.OLS(
            y.to_numpy(),
            X.to_numpy(),
            w=w,
            moran=True,
            spat_diag=True,
            white_test=True,
            robust=None,
            name_w="Weibull Kernel over Shortest Path Length",
            name_ds="Paradise Papers Metadata",
            name_y=y.columns.tolist(),
            name_x=X.columns.tolist(),
        )

        coefficients.append(
            pd.DataFrame(
                {
                    i: [
                        f"{np.format_float_scientific(c, precision=3, exp_digits=2)}\
                            ({round(p, 3)})"
                        for c, p in zip(
                            spatial_model.betas.flatten(),
                            np.array(spatial_model.t_stat)[:, 1],
                        )
                    ]
                },
                index=spatial_model.name_x,
            )
        )

        statistics.append(get_spatial_statistics(spatial_model).to_frame(i))

        significances: np.ndarray = np.array(spatial_model.t_stat)[2:, 1]

        if len(significances) > 0:
            arg_least_significant = significances.argmax()
            least_significant = spatial_model.name_x[2:][arg_least_significant]
        else:
            least_significant = spatial_model.name_x[1]

        X = X.drop(columns=[least_significant])

    c: pd.DataFrame = pd.concat(coefficients, axis=1).fillna("")
    c: pd.DataFrame = c.loc[c.eq("").sum(1).sort_values(ascending=False).index, :]

    s: pd.DataFrame = pd.concat(statistics, axis=1).fillna("")

    return (
        pd.concat([c, s]).reset_index().rename(columns={"index": "Estimate"}).fillna("")
    )


def get_spatial_weights(
    X: pd.DataFrame, y: pd.DataFrame, D: pd.DataFrame
) -> pd.DataFrame:
    """estimate weighting matrix and estimate


    :param X: Firm characteristics
    :type X: pd.DataFrame
    :param y: Market returns
    :type y: pd.DataFrame
    :param D: Shortest path distances between firms
    :type D: pd.DataFrame
    :return: Spatial weighting matrix using weibull distribution
    :rtype: pd.DataFrame
    """
    # weibull
    W: pd.DataFrame = (
        D.loc[y.index, y.index]
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
            lambda df: df.where(lambda df: df.sum(1) != 0).fillna(
                (df.shape[0] - 1) ** -1
            )
        )
        .pipe(lambda df: df * (1 - np.eye(df.shape[0])))
        .apply(lambda df: df / df.sum(), axis=1)
    )
    assert np.all(np.diag(W) == 0)
    assert W.sum(1).apply(np.testing.assert_almost_equal, desired=1).all()
    assert (W.index == y.index).all()
    assert (W.columns == y.index).all()
    assert W.var().nunique() > 1

    return W


def get_slx_basis(
    X: pd.DataFrame,
    W: pd.DataFrame,
    drop_features: List = ["price_to_earnings", "market_capitalization", "centrality"],
) -> pd.DataFrame:
    """estimate weighting matrix and estimate


    :param X: Firm characteristics
    :type X: pd.DataFrame
    :param W: Spatial weighting matrix
    :type W: pd.DataFrame
    :param drop_features: Features to remove based on significance,
        defaults to ["price_to_earnings", "market_capitalization", "centrality"]
    :type drop_features: List, optional
    :return: Model parameters estimates and statistical tests
    :rtype: pd.DataFrame
    """
    Z: pd.DataFrame = X.drop(columns=drop_features).pipe(
        lambda df: df.join((W @ df).rename(columns=lambda s: "W_" + s))
    )
    return Z


def pearson_corr(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation coefficients

    :param X: Features for analysis
    :type X: pd.DataFrame
    :param Y: Features for analysis
    :type Y: pd.DataFrame
    :return: Pearson correlation coefficients with p-values
    :rtype: pd.DataFrame
    """
    pearson_corr: pd.DataFrame = (
        pd.DataFrame(
            pairwise_distances(X.T, Y.T, metric=lambda x, y: stats.pearsonr(x, y)[0]),
            index=X.columns,
            columns=Y.columns,
        )
        .round(4)
        .astype(str)
    )
    pearson_p: pd.DataFrame = (
        pd.DataFrame(
            pairwise_distances(X.T, Y.T, metric=lambda x, y: stats.pearsonr(x, y)[1]),
            index=X.columns,
            columns=Y.columns,
        )
        .round(4)
        .astype(str)
    )

    return pearson_corr + pearson_p.applymap(lambda s: f"\n({s})")


def biplots(X: pd.DataFrame, WX: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Biplots of data

    :param X: Basis
    :type X: pd.DataFrame
    :param WX: Spatially Lagged Basis
    :type WX: Optional[pd.DataFrame]
    :return: Biplot and variance explained bar plot
    :rtype: pd.DataFrame
    """
    if WX is not None:
        F: pd.DataFrame = pd.concat([X, WX], axis=1)
    else:
        F: pd.DataFrame = X

    pca: Pipeline = Pipeline([("scale", StandardScaler()), ("pca", PCA())])
    pca.fit(F)

    explained_variance: pd.Series = pd.Series(
        pca.named_steps["pca"].explained_variance_ratio_
    ).hvplot.bar(title="Explained Variance Ratio")

    first, second = map(
        lambda s: "(" + str(s) + "%)",
        np.round(pca.named_steps["pca"].explained_variance_ratio_[:2] * 100, 2),
    )

    biplot: pd.DataFrame = (
        pd.DataFrame(
            pca.named_steps["pca"].components_, index=F.columns.str.replace("_", " ")
        )
        .rename(columns=lambda s: "Component " + str(s + 1))
        .reset_index()
        .hvplot.labels(
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
    )

    return biplot, explained_variance


def mask(Q: np.ndarray, m: int = None) -> np.ndarray:
    if m is not None:
        z = np.zeros_like(Q)
        z[m] = 1
        return z
    else:
        return np.ones_like(Q)


def get_graph(
    graph: nx.Graph, vecs: np.ndarray, vals: np.ndarray, gft: np.ndarray, k: int = 0
):
    top_component: pd.Series = pd.Series((vecs @ (gft * mask(gft, k))).flatten())
    pos: Dict = nx.kamada_kawai_layout(graph)

    component: hv.Graph = hvnx.draw(
        graph, pos, edge_width=hv.dim("weight") * 1, node_size=hv.dim("size") * 20
    ).opts(
        title=f"Eigenvector λ={np.round(vals[k], 3)} on Weibull Weighted Graph",
        colorbar=True,
    ) * hvnx.draw_networkx_nodes(
        graph,
        pos,
        node_cmap="turbo",
        node_size=80,
        node_color=top_component.tolist(),
        cmap="turbo",
        colorbar=True,
        logz=True,
    )
    return component


def returns_weibull_gft(
    W: pd.DataFrame, y: pd.DataFrame
) -> Tuple[hv.element.chart.Scatter, hv.core.layout.Layout, hv.core.layout.Layout]:

    D: np.ndarray = np.diag(np.array(W.sum(0)).flatten(), 0)
    L: np.ndarray = D - W

    vals, vecs = np.linalg.eigh(L.to_numpy())

    U, e = np.conjugate(vecs), vals  # noqa
    gft: np.ndarray = np.tensordot(U, y, ([0], [0]))

    true: pd.DataFrame = pd.DataFrame(
        {"Eigenvalues (Frequency)": vals.real, "Magnitudes": np.abs(gft).flatten()}
    )

    returns_weibull_gft_plot: hv.element.chart.Scatter = true.hvplot.scatter(
        x="Eigenvalues (Frequency)",
        y="Magnitudes",
        height=350,
        width=750,
        size=5,
        logx=False,
        color="darkblue",
    )

    graph: nx.Graph = nx.from_pandas_adjacency(
        W ** 0.125
    )  # this is only to scale the layout

    lowest = []
    for k in range(6):
        lowest.append(get_graph(graph, vecs, vals, gft, k))
    lowest_components = reduce(add, lowest).cols(2)

    arg_mags = np.argsort(np.abs(gft.flatten()))[::-1][:6]
    highest = []
    for k in arg_mags:
        highest.append(get_graph(graph, vecs, vals, gft, k))
    highest_components = reduce(add, highest).cols(2)

    return returns_weibull_gft_plot, lowest_components, highest_components


def get_regression_diagnostics(
    X: pd.DataFrame,
    y: pd.DataFrame,
    W: pd.DataFrame,
    title: str = "OLS",
    drop_features: Optional[List[str]] = None,
) -> hv.Graph:
    if drop_features is not None:
        X: pd.DataFrame = X.drop(columns=drop_features)

    graph: nx.Graph = nx.from_pandas_adjacency(
        W ** 0.125
    )  # this is only to scale the layout

    ols: OLSResults = OLS(
        y,
        X.assign(alpha=1),
    ).fit()

    fig, ax = plt.subplots(figsize=(20, 10))
    leverage: plt.Figure = influence_plot(ols, ax=ax)

    outlier_results: Tuple[np.ndarray] = ols.get_influence().cooks_distance
    cooks_distance = (
        pd.Series(outlier_results[0], y.index)
        .sort_values(ascending=False)
        .hvplot.bar(width=1200, title=f"Cooks Distance {title} Model")
        .opts(xrotation=90)
    )

    cooks_graph: hv.Graph = hvnx.draw_kamada_kawai(
        graph,
        node_cmap="turbo",
        node_size=150,
        node_color=outlier_results[0],
        label=f"Cooks Distance of {title} Model over Graph",
        cmap="turbo",
        edge_width=hv.dim("weight") * 2.5,
        colorbar=True,
        #                            title='Cooks Distance',
        height=600,
        width=1000,
        logz=True,
    )

    outlier_pca: Pipeline = Pipeline([("scale", StandardScaler()), ("pca", PCA())])
    Z: pd.DataFrame = pd.DataFrame(outlier_pca.fit_transform(X), index=X.index)

    pca_explained: pd.Series = pd.Series(
        outlier_pca.named_steps["pca"].explained_variance_ratio_
    ).hvplot.bar(
        title=f"Variance Explained Ratio of Principal Components of {title} Explanatory Variables"
    )

    U: pd.DataFrame = pd.DataFrame(
        outlier_pca.fit_transform(Z)[:, :2], index=X.index
    ).rename(columns=lambda s: "Component " + str(s + 1))

    first, second = map(
        lambda s: "(" + str(s) + "%)",
        np.round(outlier_pca.named_steps["pca"].explained_variance_ratio_[:2] * 100, 2),
    )

    U_cooks: pd.DataFrame = U.assign(**{"Cooks Distance": outlier_results[0]})

    top_cooks: pd.Series = pd.Series(outlier_results[0], index=y.index).nlargest(3)

    pca_cooks: hv.Graph = U_cooks.hvplot.scatter(
        x="Component 1",
        y="Component 2",
        xlabel="Component 1 " + first,
        ylabel="Component 2 " + second,
        color="Cooks Distance",
        cmap="turbo",
        logz=True,
        width=900,
        height=500,
        title=f"Principal Components of our {title} Model with Cooks Outliers",
    ) * U_cooks.loc[top_cooks.index, :].reset_index().hvplot.labels(
        x="Component 1",
        y="Component 2",
        text="symbol",
        text_baseline="bottom",
    )

    return leverage, cooks_distance, cooks_graph, pca_cooks, pca_explained
