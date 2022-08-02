from math import ceil
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
from simulator.action import SimulationAction
from simulator.chain import OptionChain
from simulator.position import OptionPosition
from simulator.status import SimulationStatus
from strategies.base_strategy import BaseOptionStrategy
from copy import copy


class RollingOptionSellingStrategy(BaseOptionStrategy):
    def __init__(
        self,
        trade_interval: Union[pd.Timedelta, str],
        time_to_maturity: Union[pd.Timedelta, str],
        time_to_cancel: Optional[Union[pd.Timedelta, str]] = None,
        collateralization_ratio: float = 1.0,
        utilization_ratio: float = 1.0,
        option_type: Literal["call", "put"] = "call",
        mode: Literal["moneyness", "delta"] = "moneyness",
        target: float = -0.1,
        liquidity_delta: float = 1.0,
        spread_cross: Optional[Literal["full", "half", "none"]] = "half",
        collateral_mode: Optional[Literal["primary", "secondary"]] = None,
    ):
        self.trade_interval = pd.Timedelta(trade_interval)
        self.time_to_maturity = pd.Timedelta(time_to_maturity)
        if time_to_cancel is None:
            self.time_to_cancel = self.trade_interval
        else:
            self.time_to_cancel = pd.Timedelta(time_to_cancel)
        self.maturity_days = self.time_to_maturity.total_seconds() / (24 * 60 * 60)
        self.days_to_cancel = self.time_to_cancel.total_seconds() / (24 * 60 * 60)

        self.collateralization_ratio = collateralization_ratio
        self.utilization_ratio = utilization_ratio
        self.option_type = option_type
        self.mode = mode
        self.target = target
        self.liquidity_delta = liquidity_delta

        self.concurrency = ceil(self.time_to_cancel / self.trade_interval)

        if spread_cross == "full":
            self.buy_side = "ask"
            self.sell_side = "bid"
        elif spread_cross == "half":
            self.buy_side = "ask"
            self.sell_side = "mark"
        else:
            self.buy_side = "mark"
            self.sell_side = "mark"

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

        for position in status.positions:
            if (position.maturity - time) <= self.time_to_cancel:
                moneyness = position.moneyness_ratio(spot)
                price, strike, iv = chain.get_price_by_moneyness(
                    moneyness, self.days_to_cancel, self.option_type, self.sell_side
                )
                premium_spent = price * abs(position.size)
                collateral_retrieved = position.collateral_value(spot)
                liquidity_change = -premium_spent + collateral_retrieved
                status = self._add_and_rebalance(
                    liquidity_change, status, self.liquidity_delta, spot
                )
                message = (
                    f"{time} @ {spot:.2f}: Bought {position.type} {position.strike:.2f} {position.maturity} "
                    f"of size {position.size:.2f} for {premium_spent:.2f} ({price:.2f} per notional) "
                    f"cancelling previous short position"
                )
                new_position = copy(position)
                new_position.size = -position.size
                new_position.price_start = price
                new_actions.append(
                    SimulationAction(
                        step=step,
                        timestamp=time,
                        spot=spot,
                        type="buy",
                        description=message,
                        position=new_position,
                        liquidity_change=liquidity_change,
                    )
                )
                status.positions.remove(position)

        if last_action_time is None or time - last_action_time >= self.trade_interval:
            # Sell an option
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
