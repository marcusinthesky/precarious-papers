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

import numpy as np
import pandas as pd
from scipy import stats
import pysal as ps


def backwards_selection(
    X: pd.DataFrame, y: pd.DataFrame, w: ps.lib.weights.weights.W
) -> pd.DataFrame:
    """
    Formulate model estimates as tabular results
    """
    results = []
    for i in range(X.shape[1]):

        spatial_model = ps.model.spreg.OLS(
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

        results.append(
            pd.DataFrame(
                {
                    i: [
                        f"{round(c, 3)} ({round(p, 3)})"
                        for c, p in zip(
                            spatial_model.betas.flatten(),
                            np.array(spatial_model.t_stat)[:, 1],
                        )
                    ]
                },
                index=spatial_model.name_x,
            )
        )

        arg_least_significant = np.array(spatial_model.t_stat)[1::, 1].argmax()
        least_significant = spatial_model.name_x[1:][arg_least_significant]

        X = X.drop(columns=[least_significant])

    r = pd.concat(results, axis=1).fillna("")

    return r.loc[r.isna().sum(1).sort_values(ascending=False).index, :]


def get_non_spatial_results(
    X: pd.DataFrame, y: pd.DataFrame, D: pd.DataFrame
) -> pd.DataFrame:
    """
    estimate weighting matrix and estimate
    """
    # weibull
    W = (
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

    w = ps.lib.weights.full2W(W.to_numpy())

    return backwards_selection(X, y, w)
