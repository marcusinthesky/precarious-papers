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
import datetime
from typing import Dict, List, Set, Tuple

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
    """Calls IEXCloud API to get symbols for all securities.

    :param secret: Dict with {'iex': YOUR_TOKEN}
    :type secret: str
    :return:  Availible IEXCloud companies
    :rtype: pd.DataFrame
    """
    token: str = secret["iex"]
    symbols: pd.DataFrame = get_symbols(output_format="pandas", token=token)

    return symbols


def get_entities(
    paradise_nodes_entity: pd.DataFrame,
    paradise_nodes_intermediary: pd.DataFrame,
    paradise_nodes_officer: pd.DataFrame,
    paradise_nodes_other: pd.DataFrame,
) -> pd.DataFrame:
    """Merges all entity types across files and defines index

    :param paradise_nodes_entity: [description]
    :type paradise_nodes_entity: pd.DataFrame
    :param paradise_nodes_intermediary: [description]
    :type paradise_nodes_intermediary: pd.DataFrame
    :param paradise_nodes_officer: [description]
    :type paradise_nodes_officer: pd.DataFrame
    :param paradise_nodes_other: [description]
    :type paradise_nodes_other: pd.DataFrame
    :return: Nodes and nodes attributed for graph
    :rtype: pd.DataFrame
    """

    entities: pd.DataFrame = (
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
    """Merged IEXCloud symbols and metadata with leak entities

    :param entities: [description]
    :type entities: pd.DataFrame
    :param symbols: [description]
    :type symbols: pd.DataFrame
    :return: IEXCloud companies joined on ICIJ Graph Node metadata
    :rtype: pd.DataFrame
    """
    iex_matched_entities: pd.DataFrame = (
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
    """Uses edge list to build graph

    :param paradise_edges: [description]
    :type paradise_edges: pd.DataFrame
    :return: NetworkX graph of ICIJ node edges
    :rtype: nx.Graph
    """
    paradise_graph: nx.Graph = nx.convert_matrix.from_pandas_edgelist(
        df=paradise_edges,
        source="START_ID",
        target="END_ID",
        edge_attr=paradise_edges.columns.drop(["START_ID", "END_ID"]).tolist(),
    )

    return paradise_graph


def find_path_length(source: str, target: str, G: nx.Graph) -> int:
    """Find the shortest path between nodes given a graph

    :param source: [description]
    :type source: str
    :param target: [description]
    :type target: str
    :param G: [description]
    :type G: nx.Graph
    :return: Shortest path length between nodes
    :rtype: int
    """
    try:
        return nx.shortest_path_length(G, source.item(), target.item())
    except nx.exception.NetworkXNoPath:
        warnings.warn("No path found")
        return np.nan


def compute_paradise_distances(
    iex_matched_entities: pd.DataFrame, paradise_graph: nx.Graph
) -> pd.DataFrame:
    """Using Dijskas algorithm to find the shortest path between entities

    :param iex_matched_entities: Joined IEXCloud firms and ICIJ graph entities[description]
    :type iex_matched_entities: pd.DataFrame
    :param paradise_graph: [description]
    :type paradise_graph: nx.Graph
    :return: Shortest path length pairwise distance matrix
    :rtype: pd.DataFrame
    """
    D: np.ndarray = pairwise_distances(
        X=(iex_matched_entities.node_id.to_numpy().reshape(-1, 1)),
        metric=find_path_length,
        n_jobs=-1,
        G=paradise_graph,
    )

    paradise_distances: pd.DataFrame = (
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
    """Makes call to iexcloud for balance sheet data

    :param ticker: [description]
    :type ticker: str
    :param token: [description]
    :type token: str
    :return: Balance sheet information for particular firm
    :rtype: Dict
    """
    try:
        data_set: requests.Response = requests.get(
            url=f"https://cloud.iexapis.com/stable/stock/{ticker}/balance-sheet",
            params={"period": "annual", "last": 4, "token": token},
        )
        return data_set.json()

    except requests.ConnectionError or json.decoder.JSONDecodeError:
        return {}


def balancesheet_to_frame(d: Dict) -> pd.DataFrame:
    """Formats balance sheet data

    :param d: Balance sheet information for particular firm
    :type d: Dict
    :return: Reformatted Balance sheet information
    :rtype: pd.DataFrame
    """
    if "balancesheet" in d.keys() and "symbol" in d.keys():
        return pd.DataFrame(d["balancesheet"]).assign(symbol=d["symbol"])
    else:
        return pd.DataFrame()


def get_income_statement(ticker: str, token: str) -> Dict:
    """Makes call to iexcloud for income statement data

    :param ticker: Stock ticker eg AAPL for Apple Computers
    :type ticker: str
    :param token: [description]
    :type token: str
    :return: Firm income statement information
    :rtype: Dict
    """
    try:
        data_set: requests.Response = requests.get(
            url=f"https://cloud.iexapis.com/stable/stock/{ticker}/income",
            params={"period": "annual", "last": 4, "token": token},
        )
        return data_set.json()
    except requests.ConnectionError or json.decoder.JSONDecodeError:
        return {}


def income_statement_to_frame(d: Dict) -> pd.DataFrame:
    """Formats income statement data


    :param d: [description]
    :type d: Dict
    :return: Reformatted Firm income statement information
    :rtype: pd.DataFrame
    """
    if "income" in d.keys() and "symbol" in d.keys():
        return pd.DataFrame(d["income"]).assign(symbol=d["symbol"])
    else:
        return pd.DataFrame()


def get_market_cap(ticker: str, token: str) -> Dict:
    """Makes call to iexcloud for marketcap data

    :param ticker: [description]
    :type ticker: str
    :param token: [description]
    :type token: str
    :return: Firm security market capitalization
    :rtype: Dict
    """
    try:
        data_set: requests.Response = requests.get(
            url=f"https://cloud.iexapis.com/stable/stock/{ticker}/stats/marketcap",
            params={"period": "annual", "last": 4, "token": token},
        )
        return data_set.json()

    except requests.ConnectionError or json.decoder.JSONDecodeError:
        return {}


def get_factor_data(iex_matched_entities: pd.DataFrame, secrets: Dict) -> pd.DataFrame:
    """Pulls, formats and merges firm characteristic data from IEXCloud

    :param iex_matched_entities: Joined IEXCloud firms and ICIJ graph entities[description]
    :type iex_matched_entities: pd.DataFrame
    :param secrets: Dict with {'iex': YOUR_TOKEN}
    :type secrets: Dict
    :return: Merged firm balance sheet, income state and market cap information
    :rtype: pd.DataFrame
    """
    token: str = secrets["iex"]

    balance_sheet_data: pd.DataFrame = (
        iex_matched_entities.loc[:, ["symbol"]]
        .drop_duplicates()
        .assign(
            balance_sheet=lambda df: df.symbol.apply(get_balance_sheet, token=token)
        )
    )

    balancesheet: pd.DataFrame = pd.concat(
        balance_sheet_data.balance_sheet.apply(balancesheet_to_frame).tolist()
    )

    income_statement_data: pd.DataFrame = (
        iex_matched_entities.loc[:, ["symbol"]]
        .drop_duplicates()
        .assign(
            income_statement=lambda df: df.symbol.apply(
                get_income_statement, token=token
            )
        )
    )

    income_statement: pd.DataFrame = pd.concat(
        income_statement_data.income_statement.apply(income_statement_to_frame).tolist()
    )

    market_cap_data: pd.DataFrame = (
        iex_matched_entities.loc[:, ["symbol"]]
        .drop_duplicates()
        .assign(market_cap=lambda df: df.symbol.apply(get_market_cap, token=token))
    )

    return balancesheet, income_statement, market_cap_data


def get_price_data(
    iex_matched_entities: pd.DataFrame, release: Dict, window: Dict, secret: str
) -> pd.DataFrame:
    """Query iexcloud api for price data

    :param iex_matched_entities: Joined IEXCloud firms and ICIJ graph entities[description]
    :type iex_matched_entities: pd.DataFrame
    :param release: Date of release of leak
    :type release: Dict
    :param window: +- window for study from O'Donovan
    :type window: Dict
    :param secret: Dict with {'iex': YOUR_TOKEN}
    :type secret: str
    :return: Firm security price data over event window
    :rtype: pd.DataFrame
    """
    token: str = secret["iex"]
    release: pd.Timestamp = pd.to_datetime(release["paradise_papers"])

    start: datetime.datetime = (
        release + pd.tseries.offsets.BDay(window["start"])
    ).to_pydatetime()

    end: datetime.datetime = (
        release + pd.tseries.offsets.BDay(window["end"])
    ).to_pydatetime()

    unique_tickers: List = iex_matched_entities.symbol.drop_duplicates().tolist()

    historical_prices: List = []
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

    paradise_price: pd.DataFrame = pd.concat(historical_prices, axis=1)

    return paradise_price


def compute_centrality(
    paradise_graph: nx.Graph, iex_matched_entities: pd.DataFrame
) -> pd.DataFrame:
    """[summary]

    :param paradise_graph: Paradise papers graph
    :type paradise_graph: nx.Graph
    :param iex_matched_entities: Joined IEXCloud firms and ICIJ graph entities[description]
    :type iex_matched_entities: pd.DataFrame
    :return: Eigenvector centrality of firms in ICIJ graph
    :rtype: pd.DataFrame
    """
    eigenvector_centrality: pd.DataFrame = nx.eigenvector_centrality(
        paradise_graph, tol=1e-7, max_iter=5000
    )
    symbols_map: pd.DataFrame = iex_matched_entities.set_index("node_id").symbol

    centrality: pd.DataFrame = (
        pd.DataFrame({"centrality": eigenvector_centrality})
        .join(symbols_map, how="left")
        .groupby("symbol")
        .head(1)
        .set_index("symbol")
    )

    return centrality


def get_basis(
    matched: pd.DataFrame,
    indices: pd.DataFrame,
    price: pd.DataFrame,
    balancesheet: pd.DataFrame,
    income: pd.DataFrame,
    centrality: pd.DataFrame,
    distances: pd.DataFrame,
    release: str,
) -> Tuple[pd.DataFrame]:
    """Constructs the Basis matrix and pairwise distance matrix for our experiments

    :param matched: Matched graph nodes and listed companies
    :type matched: pd.DataFrame
    :param indices: Firm indices
    :type indices: pd.DataFrame
    :param price: Firm price over event window
    :type price: pd.DataFrame
    :param balancesheet: firm balancesheet
    :type balancesheet: pd.DataFrame
    :param income: Firm income statement
    :type income: pd.DataFrame
    :param centrality: Firm global graph centrality
    :type centrality: pd.DataFrame
    :param distances: shortest-path distance matrix
    :type distances: pd.DataFrame
    :param release: Date of Paradise Papers release
    :type release: str
    :return: Firm characteristics, subsetting pairwise shortest-path distance matrix and returns
    :rtype: Tuple[pd.DataFrame]
    """

    intersecting_index: Set = pd.DataFrame(
        {
            "symbol": list(
                set(matched.symbol)
                .intersection(set(price.columns.droplevel(1)))
                .intersection(set(balancesheet.symbol))
                .intersection(set(income.symbol))
                .intersection(set(centrality.index))
                .intersection(set(distances.index))
            )
        }
    )

    matched: pd.DataFrame = matched.merge(intersecting_index, how="inner", on="symbol")
    price: pd.DataFrame = price.loc[:, pd.IndexSlice[intersecting_index.symbol, :]]
    balancesheet: pd.DataFrame = balancesheet.merge(
        intersecting_index, how="inner", on="symbol"
    )
    income: pd.DataFrame = income.merge(intersecting_index, how="inner", on="symbol")
    centrality: pd.DataFrame = centrality.loc[intersecting_index.symbol, :]
    distances: pd.DataFrame = distances.loc[
        intersecting_index.symbol, intersecting_index.symbol
    ]

    index: pd.Series = (
        matched.groupby("symbol")
        .apply(lambda df: df.sample(1))
        .set_index("symbol")
        .exchange
    )

    rf: float = ((1 + 3.18e-05) ** 5) - 1

    rm_rates: pd.DataFrame = (
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

    returns: pd.DataFrame = (
        price.loc[:, pd.IndexSlice[:, "close"]]
        .pct_change()
        .add(1)
        .cumprod()
        .tail(1)
        .subtract(1)
        .subtract(rf)
    )

    returns.columns: pd.DataFrame = returns.columns.droplevel(1)
    returns: pd.DataFrame = returns.T.rename(columns=lambda x: "returns").dropna()
    returns["excess"]: pd.DataFrame = returns["returns"].subtract(
        rm_rates.loc[returns.index]
    )
    returns["rm"]: pd.DataFrame = rm_rates.loc[returns.index].to_frame().to_numpy()

    # remove islands
    inner_index: pd.DataFrame = returns.join(
        (
            distances.where(lambda df: df.sum() > 0)
            .dropna(how="all")
            .T.where(lambda df: df.sum() > 0)
            .dropna(how="all")
        ),
        how="inner",
    ).index

    renamed_distances: pd.DataFrame = distances.loc[inner_index, inner_index]
    returns: pd.DataFrame = returns.loc[inner_index, :]

    communities: pd.DataFrame = (
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
    communities: pd.DataFrame = communities.loc[:, communities.sum() > 1]

    factors: pd.DataFrame = pd.merge_asof(
        returns.reset_index()
        .rename(columns={"index": "symbol"})
        .assign(reportDate=pd.to_datetime(release["paradise_papers"]))
        .sort_values("reportDate"),
        balancesheet.merge(
            income.pipe(
                lambda df: df.drop(
                    columns=[
                        h
                        for h in df.columns
                        if h in ["minorityInterest", "fiscalDate", "currency"]
                    ]
                )
            ),
            on=["reportDate", "symbol"],
        )
        .assign(reportDate=lambda df: pd.to_datetime(df.reportDate))
        .sort_values("reportDate"),
        on="reportDate",
        by="symbol",
        direction="backward",
    )

    average_price: pd.DataFrame = (
        price.loc[:, pd.IndexSlice[:, "close"]]
        .mean()
        .reset_index()
        .rename(columns={"level_0": "symbol", 0: "price"})
        .set_index("symbol")
        .price
    )
    features: pd.DataFrame = (
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
        .join(centrality)
    )

    # check intersection
    i: Set = set(features.index).intersection(set(renamed_distances.index))

    X: pd.DataFrame = features.loc[i, :].drop(columns=["returns", "alpha"])
    y: pd.DataFrame = features.loc[i, ["returns"]].apply(
        winsorize, limits=[0.028 - 0.0, 0.062 + 0.02]
    )
    D: pd.DataFrame = renamed_distances.loc[i, i]

    return X, y, D
