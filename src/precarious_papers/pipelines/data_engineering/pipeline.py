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

from kedro.pipeline import Pipeline, node

from .nodes import (
    get_iex_symbols,
    get_entities,
    get_factor_data,
    get_graph,
    compute_paradise_distances,
    get_iex_matched_entities,
    get_price_data,
    get_basis,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(get_iex_symbols, ["secrets"], "symbols", tags=["symbols"]),
            node(
                get_entities,
                [
                    "paradise_nodes_entity",
                    "paradise_nodes_intermediary",
                    "paradise_nodes_officer",
                    "paradise_nodes_other",
                ],
                "entities",
                tags=["entities"],
            ),
            node(
                get_iex_matched_entities,
                ["entities", "symbols"],
                "iex_matched_entities",
                tags=["matched"],
            ),
            node(
                get_factor_data,
                ["iex_matched_entities", "secrets"],
                ["balancesheet", "income_statement", "market_cap_data"],
                tags=["factors"],
            ),
            node(get_graph, ["paradise_edges"], "paradise_graph", tags=["graph"]),
            node(
                compute_paradise_distances,
                ["iex_matched_entities", "paradise_graph"],
                "paradise_distances",
                tags=["distances"],
            ),
            node(
                get_price_data,
                ["iex_matched_entities", "params:release", "params:window", "secret"],
                "paradise_price",
                tags=["price"],
            ),
            node(
                get_basis,
                dict(
                    matched="iex_matched_entities",
                    indices="indices",
                    price="paradise_price",
                    balancesheet="balance_sheet",
                    income="income_statement",
                    distances="paradise_distances",
                    release="params:release",
                ),
                ["X", "y", "D"],
                tags=["basis"],
            ),
        ]
    )
