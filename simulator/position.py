from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta as delta_bsm
from simulator.chain import OptionChain


@dataclass
class OptionPosition:
    """Holds details of an options position"""
    type: Literal["call", "put"]
    strike: float
    maturity: pd.Timestamp
    size: float
    collateral_primary: float
    collateral_secondary: float
    iv_start: float
    price_start: Optional[float] = None

    def is_long(self) -> bool:
        """Returns True if the position is long"""
        return self.size > 0

    def is_short(self) -> bool:
        """Returns True if the position is short"""
        return self.size < 0

    def is_call(self) -> bool:
        """Returns True if the position is a call"""
        return self.type == "call"

    def moneyness(self, spot: float) -> float:
        """Returns the moneyness of the position"""
        return (spot - self.strike) if self.is_call() else (self.strike - spot)

    def moneyness_ratio(self, spot: float) -> float:
        """Returns the ratio of the moneyness to the underlying price"""
        return self.moneyness(spot) / self.strike

    def valuation(
        self, spot: float, time: pd.Timestamp, chain: Optional[OptionChain] = None
    ) -> float:
        """Returns the valuation of the position"""
        time_left = self.maturity - time
        if time_left <= pd.Timedelta("1h"):
            return self.delivery(spot)

        if chain is None:
            tau = time_left / pd.Timedelta("1y")
            return self.size * black_scholes(
                flag=self.type[:1],
                S=spot,
                K=self.strike,
                t=tau,
                r=0.0,
                sigma=self.iv_start,
            )
        else:
            try:
                days_to_maturity = time_left / pd.Timedelta("1d")
                price, _, _ = chain.get_price_by_strike(
                    self.strike, days_to_maturity, self.type, side="mark"
                )
                return price * self.size
            except Exception as e:
                print(e)
                tau = time_left / pd.Timedelta("1y")
                return self.size * black_scholes(
                    flag=self.type[:1],
                    S=spot,
                    K=self.strike,
                    t=tau,
                    r=0.0,
                    sigma=self.iv_start,
                )

    def delta(
        self, spot: float, time: pd.Timestamp, chain: Optional[OptionChain] = None
    ) -> float:
        """Returns the delta of the position"""
        time_left = self.maturity - time
        if time_left <= pd.Timedelta("1h"):
            has_delivery = self.delivery(spot) != 0
            if has_delivery:
                return self.size if self.is_call() else -self.size
            else:
                return 0.0

        if chain is None:
            tau = time_left / pd.Timedelta("1y")
            return self.size * delta_bsm(
                flag=self.type[:1],
                S=spot,
                K=self.strike,
                t=tau,
                r=0.0,
                sigma=self.iv_start,
            )

        else:
            tau = time_left / pd.Timedelta("1y")
            try:
                days_to_maturity = time_left / pd.Timedelta("1d")
                _, _, iv = chain.get_price_by_strike(
                    self.strike, days_to_maturity, self.type, side="mark"
                )
                return self.size * delta_bsm(
                    flag=self.type[:1],
                    S=spot,
                    K=self.strike,
                    t=tau,
                    r=0.0,
                    sigma=iv,
                )

            except Exception as e:
                print(e)
                tau = time_left / pd.Timedelta("1y")
                return self.size * delta_bsm(
                    flag=self.type[:1],
                    S=spot,
                    K=self.strike,
                    t=tau,
                    r=0.0,
                    sigma=self.iv_start,
                )

    def collateral_value(self, spot: float) -> float:
        """Returns the collateral value of the position"""
        return self.collateral_primary + self.collateral_secondary * spot

    def delivery(self, spot: float) -> float:
        """Returns the delivery of the position"""
        moneyness = self.moneyness(spot)
        return max(0, moneyness) * self.size

    def is_liquidated(self, spot: float) -> bool:
        """Returns True if the position is liquidated"""
        return (self.collateral_value(spot) < -self.delivery(spot)) and self.is_short()

    def is_expired(self, time: pd.Timestamp) -> bool:
        """Returns True if the position is expired"""
        return time >= self.maturity
