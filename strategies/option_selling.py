from math import ceil
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
from simulator.action import SimulationAction
from simulator.chain import OptionChain
from simulator.position import OptionPosition
from simulator.status import SimulationStatus
from strategies.base_strategy import BaseOptionStrategy


class OptionSellingStrategy(BaseOptionStrategy):
    def __init__(
        self,
        trade_interval: Union[pd.Timedelta, str],
        time_to_maturity: Optional[Union[pd.Timedelta, str]] = None,
        collateralization_ratio: float = 1.0,
        utilization_ratio: float = 1.0,
        option_type: Literal["call", "put"] = "call",
        mode: Literal["moneyness", "delta"] = "moneyness",
        target: float = -0.1,
        liquidity_delta: float = 1.0,
        collateral_mode: Optional[Literal["primary", "secondary"]] = None,
    ):
        self.trade_interval = pd.Timedelta(trade_interval)
        if time_to_maturity is None:
            self.time_to_maturity = self.trade_interval
        else:
            self.time_to_maturity = pd.Timedelta(time_to_maturity)
        self.maturity_days = self.time_to_maturity.total_seconds() / (24 * 60 * 60)

        self.collateralization_ratio = collateralization_ratio
        self.utilization_ratio = utilization_ratio
        self.option_type = option_type
        self.mode = mode
        self.target = target
        self.liquidity_delta = liquidity_delta

        self.concurrency = ceil(self.time_to_maturity / self.trade_interval)

        if collateral_mode is None:
            self.cash_collat = option_type == "put"
        else:
            self.cash_collat = collateral_mode == "primary"

    def execute(
        self,
        step: int,
        time: pd.Timestamp,
        spot: float,
        status: SimulationStatus,
        chain: OptionChain,
        action_log: List[SimulationAction],
    ) -> Tuple[SimulationStatus, List[SimulationAction]]:

        last_action_time = self._get_last_strategy_action_time(action_log)
        new_actions = []
        if last_action_time is None or time - last_action_time >= self.trade_interval:
            pricing_fn = (
                chain.get_price_by_moneyness
                if self.mode == "moneyness"
                else chain.get_price_by_delta
            )
            price, strike, iv = pricing_fn(
                self.target, self.maturity_days, self.option_type, "bid"
            )
            liquidity = status.liquidity(spot)
            expenditure = liquidity * self.utilization_ratio / self.concurrency
            position_size = expenditure / spot / self.collateralization_ratio
            premium_earned = price * position_size
            collateral = expenditure

            position = OptionPosition(
                type=self.option_type,
                strike=strike,
                maturity=time + self.time_to_maturity,
                size=-position_size,
                collateral_primary=collateral if self.cash_collat else 0,
                collateral_secondary=collateral / spot if not self.cash_collat else 0,
                iv_start=iv,
                price_start=price,
            )

            liquidity_change = premium_earned - collateral
            status = self._add_and_rebalance(
                liquidity_change, status, self.liquidity_delta, spot
            )
            status.positions.append(position)

            message = (
                f"{time} @ {spot:.1f}: Sold {position.type} {position.strike:.3f} {position.maturity} "
                f"of size {position.size:.3f} with {position.collateral_primary:.3f} primary "
                f"and {position.collateral_secondary:.3f} secondary collateral "
                f"for {premium_earned:.2f} USD ({price:.2f} USD/contract)"
            )
            new_actions.append(
                SimulationAction(
                    step=step,
                    timestamp=time,
                    spot=spot,
                    type="sell",
                    description=message,
                    position=position,
                    liquidity_change=liquidity_change,
                )
            )
        return status, new_actions
