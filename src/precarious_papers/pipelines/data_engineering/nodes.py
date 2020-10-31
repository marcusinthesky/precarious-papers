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
import string
import warnings
import json
from typing import Dict

import holoviews as hv
import hvplot.pandas  # noqa
import networkx as nx
import numpy as np
import pandas as pd
from iexfinance.stocks import get_historical_data
import requests
from iexfinance.refdata import get_symbols
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from scipy.stats.mstats import winsorize


tqdm().pandas()
hv.extension("bokeh")


def get_iex_symbols(secret: str) -> pd.DataFrame:
    """
    Calls IEXCloud API to get symbols for all securities.
    """
    token = secret["iex"]
    symbols = get_symbols(output_format="pandas", token=token)

    return symbols


def get_entities(
    paradise_nodes_entity: pd.DataFrame,
    paradise_nodes_intermediary: pd.DataFrame,
    paradise_nodes_officer: pd.DataFrame,
    paradise_nodes_other: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges all entity types across files and defines index
    """

    entities = (
        pd.concat(
            [
                paradise_nodes_entity,
                paradise_nodes_intermediary,
                paradise_nodes_officer,
                paradise_nodes_other,
            ],
            axis=0,
        )
        .reset_index(drop=True)
        .rename(columns={"index": "source_index"})
    )

    return entities


def get_iex_matched_entities(
    entities: pd.DataFrame, symbols: pd.DataFrame
) -> pd.DataFrame:
    """
    Merged IEXCloud symbols and metadata with leak entities
    """
    iex_matched_entities = (
        entities.assign(
            lower_string=lambda df: df.name.str.lower().str.replace(
                "[{}]".format(string.punctuation), ""
            )
        )
        .assign(entities=lambda df: df.name)
        .merge(
            symbols.assign(
                lower_string=lambda df: df.name.str.lower().str.replace(
                    "[{}]".format(string.punctuation), ""
                )
            ).rename(columns={"name": "match"}),
            on="lower_string",
            how="inner",
        )
        .assign(score=100)
    )

    return iex_matched_entities


def get_graph(paradise_edges: pd.DataFrame) -> nx.Graph:
    """
    Uses edge list to build graph
    """
    paradise_graph = nx.convert_matrix.from_pandas_edgelist(
        df=paradise_edges,
        source="START_ID",
        target="END_ID",
        edge_attr=paradise_edges.columns.drop(["START_ID", "END_ID"]).tolist(),
    )

    return paradise_graph


def find_path_length(source: str, target: str, G: nx.Graph) -> int:
    try:
        return nx.shortest_path_length(G, source.item(), target.item())
    except nx.exception.NetworkXNoPath:
        warnings.warn("No path found")
        return np.nan


def compute_paradise_distances(
    iex_matched_entities: pd.DataFrame, paradise_graph: nx.Graph
) -> pd.DataFrame:
    D = pairwise_distances(
        X=(iex_matched_entities.node_id.to_numpy().reshape(-1, 1)),
        metric=find_path_length,
        n_jobs=-1,
        G=paradise_graph,
    )

    paradise_distances = (
        pd.DataFrame(
            D, columns=iex_matched_entities.symbol, index=iex_matched_entities.symbol
        )
        .T.drop_duplicates()
        .T.drop_duplicates()
        .reset_index()
        .groupby("symbol")
        .min()
        .T.reset_index()
        .groupby("symbol")
        .min()
    )

    return paradise_distances


def get_balance_sheet(ticker: str, token: str) -> Dict:
    """
    Makes call to iexcloud for balance sheet data
    """
    try:
        data_set = requests.get(
            url=f"https://cloud.iexapis.com/stable/stock/{ticker}/balance-sheet",
            params={"period": "annual", "last": 4, "token": token},
        )
        return data_set.json()

    except requests.ConnectionError or json.decoder.JSONDecodeError:
        return {}


def balancesheet_to_frame(d: Dict) -> pd.DataFrame:
    """
    Formats balance sheet data
    """
    if "balancesheet" in d.keys() and "symbol" in d.keys():
        return pd.DataFrame(d["balancesheet"]).assign(symbol=d["symbol"])
    else:
        return pd.DataFrame()


def get_income_statement(ticker, token) -> Dict:
    """
    Makes call to iexcloud for income statement data
    """
    try:
        data_set = requests.get(
            url=f"https://cloud.iexapis.com/stable/stock/{ticker}/income",
            params={"period": "annual", "last": 4, "token": token},
        )
        return data_set.json()
    except requests.ConnectionError or json.decoder.JSONDecodeError:
        return {}


def income_statement_to_frame(d: Dict) -> pd.DataFrame:
    """
    Formats income statement data
    """
    if "income" in d.keys() and "symbol" in d.keys():
        return pd.DataFrame(d["income"]).assign(symbol=d["symbol"])
    else:
        return pd.DataFrame()


def get_market_cap(ticker: str, token: str) -> Dict:
    """
    Makes call to iexcloud for marketcap data
    """
    try:
        data_set = requests.get(
            url=f"https://cloud.iexapis.com/stable/stock/{ticker}/stats/marketcap",
            params={"period": "annual", "last": 4, "token": token},
        )
        return data_set.json()

    except requests.ConnectionError or json.decoder.JSONDecodeError:
        return {}


def get_factor_data(iex_matched_entities: pd.DataFrame, token: str) -> pd.DataFrame:
    """
    Pulls, formats and merges firm characteristic data from IEXCloud
    """

    balance_sheet_data = (
        iex_matched_entities.loc[:, ["symbol"]]
        .drop_duplicates()
        .assign(
            balance_sheet=lambda df: df.symbol.apply(get_balance_sheet, token=token)
        )
    )

    balancesheet = pd.concat(
        balance_sheet_data.balance_sheet.apply(balancesheet_to_frame).tolist()
    )

    income_statement_data = (
        iex_matched_entities.loc[:, ["symbol"]]
        .drop_duplicates()
        .assign(
            income_statement=lambda df: df.symbol.apply(
                get_income_statement, token=token
            )
        )
    )

    income_statement = pd.concat(
        income_statement_data.income_statement.apply(income_statement_to_frame).tolist()
    )

    market_cap_data = (
        iex_matched_entities.loc[:, ["symbol"]]
        .drop_duplicates()
        .assign(market_cap=lambda df: df.symbol.apply(get_market_cap, token=token))
    )

    return balancesheet, income_statement, market_cap_data


def get_price_data(
    iex_matched_entities: pd.DataFrame, release: Dict, window: Dict, secret: str
) -> pd.DataFrame:
    token = secret["iex"]
    release = pd.to_datetime(release["paradise_papers"])

    start = (release + pd.tseries.offsets.BDay(window["start"])).to_pydatetime()

    end = (release + pd.tseries.offsets.BDay(window["end"])).to_pydatetime()

    unique_tickers = iex_matched_entities.symbol.drop_duplicates().tolist()

    historical_prices = []
    chunks = np.array_split(unique_tickers, (len(unique_tickers)) // 100 + 1)
    for c in chunks:
        historical_prices.append(
            get_historical_data(
                symbols=c.tolist(),
                start=start,
                end=end,
                close_only=True,
                token=token,
                output_format="pandas",
            )
        )

    paradise_price = pd.concat(historical_prices, axis=1)

    return paradise_price


def get_basis(
    matched: pd.DataFrame,
    indices: pd.DataFrame,
    price: pd.DataFrame,
    balancesheet: pd.DataFrame,
    income: pd.DataFrame,
    distances: pd.DataFrame,
    release: str,
) -> pd.DataFrame:
    """
    Constructs the Basis matrix and pairwise distance matrix for our experiments

    """
    index = (
        matched.groupby("symbol")
        .apply(lambda df: df.sample(1))
        .set_index("symbol")
        .exchange
    )

    rf = ((1 + 3.18e-05) ** 5) - 1

    rm_rates = (
        index.rename("index")
        .to_frame()
        .merge(
            indices.pct_change()
            .add(1)
            .cumprod()
            .tail(1)
            .subtract(1)
            .T.rename(columns=lambda x: "rm"),
            left_on="index",
            right_index=True,
        )
        .rm.subtract(rf)
    )

    returns = (
        price.loc[:, pd.IndexSlice[:, "close"]]
        .pct_change()
        .add(1)
        .cumprod()
        .tail(1)
        .subtract(1)
        .subtract(rf)
    )

    returns.columns = returns.columns.droplevel(1)
    returns = returns.T.rename(columns=lambda x: "returns").dropna()

    returns = returns.assign(
        excess=returns.returns.subtract(rm_rates).dropna(), rm=rm_rates
    ).dropna()

    # remove islands
    inner_index = returns.join(
        (
            distances.where(lambda df: df.sum() > 0)
            .dropna(how="all")
            .T.where(lambda df: df.sum() > 0)
            .dropna(how="all")
        ),
        how="inner",
    ).index

    renamed_distances = distances.loc[inner_index, inner_index]
    returns = returns.loc[inner_index, :]

    communities = (
        pd.get_dummies(
            renamed_distances.where(
                lambda df: df.isna(), lambda df: df.apply(lambda x: x.index)
            )
            .fillna("")
            .apply(lambda df: hash(df.str.cat()))
        )
        .T.reset_index(drop=True)
        .T
    )
    communities = communities.loc[:, communities.sum() > 1]

    factors = pd.merge_asof(
        returns.reset_index()
        .rename(columns={"index": "symbol"})
        .assign(reportDate=pd.to_datetime(release["paradise_papers"]))
        .sort_values("reportDate"),
        balancesheet.merge(
            income.drop(columns=["minorityInterest", "fiscalDate", "currency"]),
            on=["reportDate", "symbol"],
        )
        .assign(reportDate=lambda df: pd.to_datetime(df.reportDate))
        .sort_values("reportDate"),
        on="reportDate",
        by="symbol",
        direction="backward",
    )

    average_price = (
        price.loc[:, pd.IndexSlice[:, "close"]]
        .mean()
        .reset_index()
        .rename(columns={"level_0": "symbol", 0: "price"})
        .set_index("symbol")
        .price
    )
    features = (
        factors.set_index("symbol")
        .fillna(factors.mean())
        .assign(alpha=1)
        .select_dtypes(include="number")
        .join(average_price)
        .assign(
            price_to_earnings=lambda df: df.price / (df.totalAssets / df.commonStock),
            market_capitalization=lambda df: df.price * df.commonStock,
            profit_margin=lambda df: df.grossProfit / df.totalRevenue,
            price_to_research=lambda df: (df.price * df.commonStock)
            / df.researchAndDevelopment,
        )
        .loc[
            :,
            [
                "rm",
                "price_to_earnings",
                "market_capitalization",
                "profit_margin",
                "price_to_research",
                "alpha",
                "returns",
            ],
        ]
    )

    X, y, D = (
        features.drop(columns=["returns", "alpha"]),
        features.returns.to_frame().apply(
            winsorize, limits=[0.028 - 0.0, 0.062 + 0.02]
        ),
        renamed_distances.loc[features.index, features.index],
    )

    return X, y, D
