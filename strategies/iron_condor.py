from math import ceil
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
from simulator.action import SimulationAction
from simulator.chain import OptionChain
from simulator.position import OptionPosition
from simulator.status import SimulationStatus
from strategies.base_strategy import BaseOptionStrategy


class IronCondorStrategy(BaseOptionStrategy):
    """Defines an Iron Condor strategy"""
    def __init__(
        self,
        trade_interval: Union[pd.Timedelta, str],
        time_to_maturity: Optional[Union[pd.Timedelta, str]] = None,
        collateralization_ratio: float = 1.0,
        # EG question / note: leg_coll_ratio allocates 25% of collateral to each leg.
        # There's probably a better way to do this?
        leg_coll_ratio: float = 0.25,
        utilization_ratio: float = 1.0,
        mode: Literal["moneyness", "delta"] = "delta",
        short_put_target: float = 0.15,
        short_call_target: float = 0.15,
        long_put_target: float = 0.10,
        long_call_target: float = 0.10,
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
        self.leg_coll_ratio = leg_coll_ratio
        self.utilization_ratio = utilization_ratio
        self.mode = mode
        self.short_put_target = short_put_target
        self.short_call_target = short_call_target
        self.long_put_target = long_put_target
        self.long_call_target = long_call_target
        self.liquidity_delta = liquidity_delta

        self.concurrency = ceil(self.time_to_maturity / self.trade_interval)

        if collateral_mode is None:
            self.cash_collat = False
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

            # Sell a call option
            pricing_fn = (
                chain.get_price_by_moneyness
                if self.mode == "moneyness"
                else chain.get_price_by_delta
            )
            short_call_price, short_call_strike, short_call_iv = pricing_fn(
                self.short_call_target, self.maturity_days, "call", "bid"
            )
            liquidity = status.liquidity(spot)
            expenditure = liquidity * self.utilization_ratio / self.concurrency * self.leg_coll_ratio
            position_size = expenditure / spot / self.collateralization_ratio
            premium_earned = short_call_price * position_size
            collateral = expenditure

            call_short_position = OptionPosition(
                type="call",
                strike=short_call_strike,
                maturity=time + self.time_to_maturity,
                size=-position_size,
                collateral_primary=collateral if self.cash_collat else 0,
                collateral_secondary=collateral / spot if not self.cash_collat else 0,
                iv_start=short_call_iv,
                price_start=short_call_price,
            )

            liquidity_change = -collateral
            status = self._add_and_rebalance(
                liquidity_change, status, self.liquidity_delta, spot
            )
            status.positions.append(call_short_position)

            message = (
                f"{time} @ {spot:.1f}: Sold {call_short_position.type} \
                {call_short_position.strike:.3f} {call_short_position.maturity} "
                f"of size {call_short_position.size:.3f} with \
                {call_short_position.collateral_primary:.3f} primary "
                f"and {call_short_position.collateral_secondary:.3f} secondary collateral "
                f"for {premium_earned:.2f} USD ({short_call_price:.2f} USD/contract)"
            )
            new_actions.append(
                SimulationAction(
                    step=step,
                    timestamp=time,
                    spot=spot,
                    type="sell",
                    description=message,
                    position=call_short_position,
                    liquidity_change=liquidity_change + premium_earned,
                )
            )

            # Sell a put option
            short_put_price, short_put_strike, short_put_iv = pricing_fn(
                self.short_put_target, self.maturity_days, "put", "bid"
            )
            liquidity = status.liquidity(spot)
            expenditure = liquidity * self.utilization_ratio / self.concurrency * self.leg_coll_ratio
            position_size = expenditure / spot / self.collateralization_ratio
            premium_earned = short_put_price * position_size
            collateral = expenditure

            put_short_position = OptionPosition(
                type="put",
                strike=short_put_strike,
                maturity=time + self.time_to_maturity,
                size=-position_size,
                collateral_primary=collateral if self.cash_collat else 0,
                collateral_secondary=collateral / spot if not self.cash_collat else 0,
                iv_start=short_put_iv,
                price_start=short_put_price,
            )

            liquidity_change = -collateral
            status = self._add_and_rebalance(
                liquidity_change, status, self.liquidity_delta, spot
            )
            status.positions.append(put_short_position)

            message = (
                f"{time} @ {spot:.1f}: Sold {put_short_position.type} \
                {put_short_position.strike:.3f} {put_short_position.maturity} "
                f"of size {put_short_position.size:.3f} with \
                {put_short_position.collateral_primary:.3f} primary "
                f"and {put_short_position.collateral_secondary:.3f} secondary collateral "
                f"for {premium_earned:.2f} USD ({short_put_price:.2f} USD/contract)"
            )
            new_actions.append(
                SimulationAction(
                    step=step,
                    timestamp=time,
                    spot=spot,
                    type="sell",
                    description=message,
                    position=put_short_position,
                    liquidity_change=liquidity_change + premium_earned,
                )
            )

            # Buy a put option
            long_put_price, long_put_strike, long_put_iv = pricing_fn(
                self.long_put_target, self.maturity_days, "put", "ask"
            )
            put_long_position = OptionPosition(
                type="put",
                strike=long_put_strike,
                maturity=time + self.time_to_maturity,
                size=position_size,
                collateral_primary=0.0,
                collateral_secondary=0.0,
                iv_start=long_put_iv,
                price_start=long_put_price,
            )
            status.positions.append(put_long_position)
            message = (
                f"{time} @ {spot:.1f}: Bought {put_long_position.type} \
                {put_long_position.strike:.3f} {put_long_position.maturity} "
                f"of size {put_long_position.size:.3f} "
                f"for {premium_earned:.2f} USD ({long_put_price:.2f} USD/contract)"
            )
            new_actions.append(
                SimulationAction(
                    step=step,
                    timestamp=time,
                    spot=spot,
                    type="buy",
                    description=message,
                    position=put_long_position,
                    liquidity_change=liquidity_change - premium_earned,
                )
            )

            # Buy a call option
            long_call_price, long_call_strike, long_call_iv = pricing_fn(
                self.long_call_target, self.maturity_days, "call", "ask"
            )
            call_long_position = OptionPosition(
                type="call",
                strike=long_call_strike,
                maturity=time + self.time_to_maturity,
                size=position_size,
                collateral_primary=0.0,
                collateral_secondary=0.0,
                iv_start=long_call_iv,
                price_start=long_call_price,
            )
            status.positions.append(call_long_position)
            message = (
                f"{time} @ {spot:.1f}: Bought {call_long_position.type} \
                {call_long_position.strike:.3f} {call_long_position.maturity} "
                f"of size {call_long_position.size:.3f} "
                f"for {premium_earned:.2f} USD ({long_call_price:.2f} USD/contract)"
            )
            new_actions.append(
                SimulationAction(
                    step=step,
                    timestamp=time,
                    spot=spot,
                    type="buy",
                    description=message,
                    position=call_long_position,
                    liquidity_change=liquidity_change - premium_earned,
                )
            )
        return status, new_actions