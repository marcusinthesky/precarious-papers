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

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    get_spatial_weights,
    get_slx_basis,
    backwards_selection,
    pearson_corr,
    biplots,
    returns_weibull_gft,
    get_regression_diagnostics,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                pearson_corr,
                ["X", "X"],
                "nonspatialpearson",
                tags=["pearson", "local"],
            ),
            node(
                get_spatial_weights,
                ["X", "y", "D"],
                "W",
                tags=["spatialweights", "local"],
            ),
            node(
                returns_weibull_gft,
                ["W", "y"],
                [
                    "returns_weibull_gft",
                    "lowest_frequencies",
                    "top_magnitude_frequencies",
                ],
                tags=["gft", "local"],
            ),
            node(
                backwards_selection,
                ["X", "y", "W"],
                "nonspatialresults",
                tags=["nonspatialmodel", "local"],
            ),
            node(
                get_regression_diagnostics,
                ["X", "y", "W", "params:ols", "params:drop_features"],
                [
                    "ols_leverage",
                    "ols_cooks_distance",
                    "ols_cooks_graph",
                    "ols_pca_cooks",
                    "ols_pca_explained",
                    "ols_pca_clustered",
                    "ols_clustered_graph",
                    "ols_cluster_model",
                    "ols_pca_model",
                ],
                tags=["nonspatialmodel", "nonspatialdiagnostics", "local"],
            ),
            node(
                get_slx_basis,
                ["X", "W", "params:drop_features"],
                "WX",
                tags=["slx", "local"],
            ),
            node(
                pearson_corr,
                ["X", "WX"],
                "nonspatialspatialpearson",
                tags=["pearson", "local"],
            ),
            node(
                pearson_corr,
                ["WX", "WX"],
                "spatialpearson",
                tags=["pearson", "local"],
            ),
            node(
                biplots,
                ["X", "WX"],
                ["biplot", "explained_variance"],
                tags=["biplot", "local"],
            ),
            node(
                backwards_selection,
                ["WX", "y", "W"],
                "spatialresults",
                tags=["spatialmodel", "local"],
            ),
            node(
                get_regression_diagnostics,
                ["WX", "y", "W", "params:slx"],
                [
                    "slx_leverage",
                    "slx_cooks_distance",
                    "slx_cooks_graph",
                    "slx_pca_cooks",
                    "slx_pca_explained",
                    "slx_pca_clustered",
                    "slx_clustered_graph",
                    "slx_cluster_model",
                    "slx_pca_model",
                ],
                tags=["spatialmodel", "spatialdiagnostics", "local"],
            ),
        ]
    )
