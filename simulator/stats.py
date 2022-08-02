from typing import Iterable
import pandas as pd
import numpy as np


def price_to_returns(price_data: pd.Series) -> pd.Series:
    returns = price_data.pct_change()
    returns = returns.replace([np.inf, -np.inf], float("NaN")).fillna(0)
    return returns


def annual_return_to_period(annual_return: float, periods: int) -> float:
    return (1 + annual_return) ** (1 / periods) - 1 if annual_return > 0 else 0.0


def sharpe_ratio(
    price_data: pd.Series,
    annual_rf: float = 0.0,
    annual_periods: int = 365,
    annualize: bool = True,
) -> float:
    """
    Calculate the Sharpe ratio of a strategy, based on a set of returns.

    Parameters
    ----------
    price_data : pd.Series
        A pandas Series containing prices (e.g. an index)
    risk_free_rate : float
        The risk-free rate of borrowing/lending, in percent per year.
    periods : int
        The number of periods to use for the calculation.
    annualize : bool, optional
        Whether to annualize the Sharpe ratio.

    Returns
    -------
    float
        The Sharpe ratio.

    References
    ----------
    https://en.wikipedia.org/wiki/Sharpe_ratio
    """
    assert annual_periods > 0, "annual_periods must be greater than 0"

    returns = price_to_returns(price_data)
    period_rf = annual_return_to_period(annual_rf, annual_periods)

    sharpe = (returns.mean() - period_rf) / returns.std()

    if annualize:
        sharpe *= np.sqrt(annual_periods)

    return sharpe


def sortino_ratio(
    price_data: pd.Series,
    annual_rf: float = 0.0,
    annual_periods: int = 365,
    annualize: bool = True,
) -> float:
    """
    Calculate the Sortino ratio of a strategy, based on a set of returns.

    Parameters
    ----------
    price_data : pd.Series
        A pandas Series containing prices (e.g. an index)
    risk_free_rate : float
        The risk-free rate of borrowing/lending, in percent per year.
    periods : int
        The number of periods to use for the calculation.
    annualize : bool, optional
        Whether to annualize the Sortino ratio.

    Returns
    -------
    float
        The Sortino ratio.

    References
    ----------
    https://en.wikipedia.org/wiki/Sortino_ratio
    """
    assert annual_periods > 0, "annual_periods must be greater than 0"

    returns = price_to_returns(price_data)
    period_rf = annual_return_to_period(annual_rf, annual_periods)

    downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))

    sortino = (returns.mean() - period_rf) / downside

    if annualize:
        sortino *= np.sqrt(annual_periods)

    return sortino


def cagr(
    price_data: pd.Series,
    annual_rf: float = 0.0,
    annual_periods: int = 365,
) -> float:
    """Compound Annual Growth Rate (CAGR), from a set of prices

    Args:
        price_data (pd.Series): A pandas Series containing prices (e.g. an index)
        annual_rf (float): The risk-free rate of borrowing/lending, in percent per year.
        annual_periods (int, optional): The number of periods of data in a year.

    Returns:
        float: The CAGR
    """
    assert annual_periods > 0, "annual_periods must be greater than 0"

    periods = len(price_data) / annual_periods
    return (price_data.iloc[-1] / price_data.iloc[0]) ** (1 / periods) - 1


def maximum_drawdown(price_data: pd.Series) -> float:
    """Calculate the maximum drawdown of a strategy, based on a set of prices.

    Args:
        prices (pd.Series): A pandas Series containing prices (e.g. an index)

    Returns:
        float: The maximum drawdown
    """
    return (price_data / price_data.expanding(0).max()).min() - 1


def calmar_ratio(price_data: pd.Series) -> float:
    """Calculate the Calmar ratio of a strategy, based on a set of prices.

    Args:
        prices (pd.Series): A pandas Series containing prices (e.g. an index)

    Returns:
        float: The Calmar ratio
    """
    return cagr(price_data) / abs(maximum_drawdown(price_data))
