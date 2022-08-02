import warnings
from datetime import datetime
from math import sqrt
from typing import Callable, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta as delta_bsm
from py_vollib.black_scholes.implied_volatility import implied_volatility as imp_vol_bsm
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

warnings.filterwarnings("ignore")


class OptionChain:
    def __init__(
        self,
        chain: pd.DataFrame,
        default_iv: float = 1.0,
        iv_margin: float = 1.25,
        risk_free_rate: float = 0.0,
    ):
        if "iv_ratio" not in chain.columns:
            chain["iv_ratio"] = 1.0

        if "spot_ratio" not in chain.columns:
            chain["spot_ratio"] = 1.0

        self.chain = chain
        self.iv_margin = iv_margin
        self.default_iv = default_iv
        self.spot = (self.chain["underlying_price"] * self.chain["spot_ratio"]).mean()
        self.rf = risk_free_rate

    @staticmethod
    def _linear_interpolation(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
        return interp1d(xs, ys, kind="linear", fill_value="extrapolate")(x)

    @staticmethod
    def _interpolate_skew_line(
        strike: float,
        strikes: np.ndarray,
        ivs: np.ndarray,
        margin: float = 1.25,
    ) -> float:
        """Interpolate the skew line so that it is not higher than the max plus a margin
        and no lower than the min"""
        max_iv, min_iv = ivs.max(), ivs.min()
        iv = OptionChain._linear_interpolation(strike, strikes, ivs)
        return float(max(min(max_iv * margin, iv), min_iv))

    @staticmethod
    def _interpolate_term_structure(
        tau: float,
        taus: np.ndarray,
        ivs: np.ndarray,
    ) -> float:
        variances = taus * ivs**2
        variance = OptionChain._linear_interpolation(tau, taus, variances)
        return float(sqrt(variance / tau))

    def _get_strike_iv_interpolator(
        self,
        price_chain: pd.DataFrame,
        exp: int,
        option_type: Literal["call", "put"],
    ) -> Callable[[float], float]:
        if len(price_chain) == 0:
            return lambda strike: self.default_iv

        option_type = option_type.lower()[0]
        exp_counts = price_chain["exp"].value_counts()
        exp_multiple = exp_counts[exp_counts >= 2].index.values
        if len(exp_multiple) >= 2:
            exp_multiple.sort()
            earlier_exps = exp_multiple[exp_multiple < exp]
            later_exps = exp_multiple[exp_multiple >= exp]
            if len(earlier_exps) == 0:
                exp_lo, exp_hi = later_exps[0], later_exps[1]
            elif len(later_exps) == 0:
                exp_lo, exp_hi = earlier_exps[-2], earlier_exps[-1]
            else:
                exp_lo, exp_hi = earlier_exps[-1], later_exps[1]

            chain_lo = price_chain[price_chain["exp"] == exp_lo].sort_values("K")
            chain_hi = price_chain[price_chain["exp"] == exp_hi].sort_values("K")

            # strikes_lo, strikes_hi = chain_lo["K"], chain_hi["K"]

            strikes_lo = []
            ivs_lo = []
            for p, k, t, r in zip(
                chain_lo["p"], chain_lo["K"], chain_lo["exp"], chain_lo["iv_ratio"]
            ):
                tau = t / 31536000
                try:
                    iv = imp_vol_bsm(p, self.spot, k, tau, 0.0, option_type) * r
                    ivs_lo.append(iv)
                    strikes_lo.append(k)
                except:
                    pass

            strikes_hi = []
            ivs_hi = []
            for p, k, t, r in zip(
                chain_hi["p"], chain_hi["K"], chain_hi["exp"], chain_hi["iv_ratio"]
            ):
                tau = t / 31536000
                try:
                    iv = imp_vol_bsm(p, self.spot, k, tau, 0.0, option_type) * r
                    ivs_hi.append(iv)
                    strikes_hi.append(k)
                except:
                    pass

            return lambda strike: (
                self._interpolate_term_structure(
                    exp,
                    np.array([exp_lo, exp_hi]),
                    np.array(
                        [
                            self._interpolate_skew_line(
                                strike,
                                np.array(strikes_lo),
                                np.array(ivs_lo),
                                self.iv_margin,
                            ),
                            self._interpolate_skew_line(
                                strike,
                                np.array(strikes_hi),
                                np.array(ivs_hi),
                                self.iv_margin,
                            ),
                        ]
                    ),
                )
            )
        else:
            return lambda strike: self.default_iv

    def _interpolate_chain(
        self,
        chain_side: pd.DataFrame,
        strike: float,
        exp: int,
        option_type: Literal["call", "put"],
    ) -> float:
        iv_func = self._get_strike_iv_interpolator(chain_side, exp, option_type)
        return iv_func(strike)

    def _get_price_chain(
        self,
        option_type: Literal["call", "put"] = "call",
        side: Literal["bid", "mark", "ask"] = "ask",
    ) -> pd.DataFrame:

        df = self.chain[self.chain["type"] == option_type].copy()

        price_col = f"{side}_price"
        df["p"] = df[price_col] * self.spot
        df["K"] = df["strike"] * df["spot_ratio"]
        df["exp"] = (df["expiration"] - df["timestamp"]).dt.total_seconds().astype(int)
        df["intrinsic_value"] = df["p"] - (
            (self.spot - df["K"]) if option_type == "call" else (df["K"] - self.spot)
        )
        df = df[df["intrinsic_value"] > 0]
        df = df[["p", "K", "exp", "iv_ratio"]].reset_index(drop=True).dropna()

        return df

    def get_price_by_strike(
        self,
        strike: float,
        days_to_maturity: float = 7.0,
        option_type: Literal["call", "put"] = "call",
        side: Literal["bid", "mark", "ask"] = "ask",
    ) -> Tuple[float, float, float]:

        exp = int(days_to_maturity * 24 * 60 * 60)
        tau = exp / 60 / 60 / 24 / 365.25
        price_chain = self._get_price_chain(option_type, side)

        iv = self._interpolate_chain(price_chain, strike, exp, option_type)
        price = black_scholes(option_type[0], self.spot, strike, tau, self.rf, iv)

        return price, strike, iv

    def get_price_by_moneyness(
        self,
        moneyness: float,
        days_to_maturity: float = 7.0,
        option_type: Literal["call", "put"] = "call",
        side: Literal["bid", "mark", "ask"] = "ask",
    ) -> Tuple[float, float, float]:

        exp = int(days_to_maturity * 24 * 60 * 60)
        tau = exp / 60 / 60 / 24 / 365.25
        price_chain = self._get_price_chain(option_type, side)

        strike = (
            self.spot / (1 + moneyness)
            if option_type == "call"
            else self.spot / (1 - moneyness)
        )
        iv = self._interpolate_chain(price_chain, strike, exp, option_type)
        price = black_scholes(option_type[0], self.spot, strike, tau, self.rf, iv)

        return price, strike, iv

    def get_price_by_delta(
        self,
        delta: float,
        days_to_maturity: float = 7.0,
        option_type: Literal["call", "put"] = "call",
        side: Literal["bid", "mark", "ask"] = "ask",
    ) -> Tuple[float, float, float]:

        exp = int(days_to_maturity * 24 * 60 * 60)
        tau = exp / 60 / 60 / 24 / 365.25
        price_chain = self._get_price_chain(option_type, side)

        iv_interp = self._get_strike_iv_interpolator(price_chain, exp, option_type)

        delta_interp = lambda k: delta_bsm(
            option_type[0], self.spot, k, tau, self.rf, iv_interp(k)
        )
        strike = float(fsolve(lambda k: delta_interp(k) - delta, x0=self.spot)[0])
        iv = iv_interp(strike)
        price = black_scholes(option_type[0], self.spot, strike, tau, self.rf, iv)

        return price, strike, iv

    def get_strike_by_price(
        self,
        price: float,
        days_to_maturity: float = 7.0,
        option_type: Literal["call", "put"] = "call",
        side: Literal["bid", "mark", "ask"] = "ask",
    ) -> Tuple[float, float, float]:

        exp = int(days_to_maturity * 24 * 60 * 60)
        tau = exp / 60 / 60 / 24 / 365.25
        price_chain = self._get_price_chain(option_type, side)

        iv_interp = self._get_strike_iv_interpolator(price_chain, exp, option_type)

        price_interp = lambda k: black_scholes(
            option_type[0], self.spot, k, tau, self.rf, iv_interp(k)
        )
        price_guess = price_interp(self.spot)  # ATM option price as first guess
        strike = float(fsolve(lambda k: price_interp(k) - price, x0=price_guess)[0])
        iv = iv_interp(strike)

        return price, strike, iv


class OptionChainTimeline:
    def __init__(
        self,
        chain: pd.DataFrame,
        iv_ratio: float = 1.0,
        spot_ratio: float = 1.0,
        default_iv: float = 1.0,
        iv_margin: float = 1.25,
    ):
        if "iv_ratio" not in chain.columns:
            chain["iv_ratio"] = iv_ratio

        if "spot_ratio" not in chain.columns:
            chain["spot_ratio"] = spot_ratio

        self.chain = chain
        self.default_iv = default_iv
        self.iv_margin = iv_margin

    def get_timestamp_chain(
        self, timestamp: pd.Timestamp, risk_free_rate: float = 0.0
    ) -> Tuple[Optional[float], OptionChain]:
        chain = self.chain[self.chain["timestamp"] == timestamp]
        spot = (
            (chain["spot_ratio"] * chain["underlying_price"]).mean()
            if len(chain) > 0
            else None
        )
        return spot, OptionChain(chain, self.default_iv, self.iv_margin, risk_free_rate)

    def iterate(self, start: datetime, end: datetime) -> Iterable[Tuple[float, float]]:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        simulation_chain: pd.DataFrame = self.chain[
            (self.chain["timestamp"] >= start) & (self.chain["timestamp"] <= end)
        ]

        simulation_times = simulation_chain["timestamp"].unique()
        simulation_times.sort()

        timeline_df = (
            simulation_chain[["timestamp", "underlying_price", "spot_ratio"]]
            .groupby("timestamp")
            .mean()
            .rename(columns={"underlying_price": "spot"})
            .reset_index()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        for i, row in timeline_df.iterrows():
            timestamp = row["timestamp"]
            spot = row[i, "spot"] * row[i, "spot_ratio"]
            yield timestamp, spot
