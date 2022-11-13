from typing import Iterable, List, Tuple, Optional, Union, TYPE_CHECKING

import pandas as pd
from simulator.action import SimulationAction
from simulator.chain import OptionChain, OptionChainTimeline
from simulator.position import OptionPosition
from simulator.result import SimulationResult
from simulator.status import SimulationStatus

if TYPE_CHECKING:
    from strategies.base_strategy import BaseOptionStrategy


class OptionSimulator:
    """Simulates the performance of an option strategy"""
    TIMELINE_COLUMNS = [
        "timestamp",
        "spot",
        "primary",
        "secondary",
        "liquidity",
        "num_long_positions",
        "num_short_positions",
        "size_long_positions",
        "size_short_positions",
        "collateral_primary",
        "collateral_secondary",
        "total_collaterals",
        "positions_delta",
        "total_delta",
        "positions_value",
        "total_value",
    ]

    def __init__(
        self,
        chain_timeline: "OptionChainTimeline",
        strategies: List["BaseOptionStrategy"],
        timing_signal: Optional[pd.Series] = None,
    ):
        assert len(strategies) > 0
        if timing_signal is not None:
            assert all(isinstance(s, int) for s in timing_signal.unique().tolist())
            assert max(timing_signal.unique().tolist()) < len(strategies)
            assert min(timing_signal.unique().tolist()) >= 0
        else:
            timing_signal = pd.Series()

        self.chain_timeline = chain_timeline
        self.strategies = strategies
        self.timing_signal = timing_signal.sort_index()

    def _reset_strategies(self):
        for strategy in self.strategies:
            strategy.reset()

    def _log_message(self, message: str):
        print(message)

    def _iterate_timeline(
        self, start: pd.Timestamp, end: pd.Timestamp, interval: pd.Timedelta
    ) -> Iterable[Tuple[int, pd.Timestamp, float, int, OptionChain]]:
        """Iterate through the timeline, yielding the step, timestamp, timing signal,
        strategy, and chain."""

        start: pd.Timestamp = pd.Timestamp(start)
        end: pd.Timestamp = pd.Timestamp(end)

        num_steps = int((end - start) / interval)
        last_spot = 1.0
        last_chain = None
        for i in range(num_steps):
            timestamp = start + i * interval
            spot, chain = self.chain_timeline.get_timestamp_chain(timestamp)
            if spot is None:
                spot = last_spot
                chain = last_chain
            signal: int = self.timing_signal.get(timestamp)
            if signal is None:  # Take the previous signal if no signal is found
                previous = self.timing_signal[self.timing_signal.index < timestamp]
                signal = previous.iloc[-1] if len(previous) > 0 else 0

            last_spot = spot
            last_chain = chain
            yield i, timestamp, spot, signal, chain

    @staticmethod
    def _check_liquidations(
        step: int, time: pd.Timestamp, spot: float, status: SimulationStatus
    ) -> Tuple[SimulationStatus, List[SimulationAction]]:
        """Check for liquidations and update status."""
        liquidated_positions = [p for p in status.positions if p.is_liquidated(spot)]
        actions = []
        for position in liquidated_positions:
            status.positions.remove(position)

            message = (
                f"{time} @ {spot:.1f}: Liquidated {position.type} {position.strike:.3f} {position.maturity}"
                f"of size {position.size:.3f} with {position.collateral_primary:.3f} primary "
                f"and {position.collateral_secondary:.3f} secondary collateral"
            )
            actions.append(
                SimulationAction(
                    step=step,
                    timestamp=time,
                    spot=spot,
                    type="liquidation",
                    description=message,
                    position=position,
                    liquidity_change=0.0,
                )
            )
        return status, actions

    @staticmethod
    def _check_expirations(
        step: int, time: pd.Timestamp, spot: float, status: SimulationStatus
    ) -> Tuple[SimulationStatus, List[SimulationAction]]:
        expired_positions = [p for p in status.positions if p.is_expired(time)]
        actions = []
        for position in expired_positions:
            collateral = position.collateral_value(spot)
            delivery = position.delivery(spot)
            liquidity_change = delivery + collateral  # short side delivery is negative
            side = "long" if position.is_long() else "short"

            status.primary += liquidity_change
            status.positions.remove(position)

            message = (
                f"{time} @ {spot:.1f}: Expired {side} {position.type} {position.strike:.3f} "
                f"of size {position.size:.3f} with delivery of {delivery:.3f}"
            )
            actions.append(
                SimulationAction(
                    step=step,
                    timestamp=time,
                    spot=spot,
                    type="expiration",
                    description=message,
                    position=position,
                    liquidity_change=liquidity_change,
                )
            )
        return status, actions

    def _get_timeline_row(
        self,
        time: pd.Timestamp,
        spot: float,
        status: SimulationStatus,
        chain: Optional[OptionChain] = None,
    ) -> pd.Series:
        liquidity = status.liquidity(spot)
        collateral_primary = status.collateral_primary()
        collateral_secondary = status.collateral_secondary()
        total_collaterals = collateral_primary + collateral_secondary * spot
        positions_value = sum(p.valuation(spot, time, chain) for p in status.positions)
        total_value = liquidity + positions_value + total_collaterals
        positions_delta = status.positions_delta(spot, time, chain)
        total_delta = status.secondary + positions_delta + collateral_secondary

        return pd.Series(
            {
                "timestamp": time,
                "spot": spot,
                "primary": status.primary,
                "secondary": status.secondary,
                "liquidity": liquidity,
                "num_long_positions": status.num_long_positions(),
                "num_short_positions": status.num_short_positions(),
                "size_long_positions": status.size_long_positions(),
                "size_short_positions": status.size_short_positions(),
                "collateral_primary": collateral_primary,
                "collateral_secondary": collateral_secondary,
                "total_collaterals": total_collaterals,
                "positions_delta": positions_delta,
                "total_delta": total_delta,
                "positions_value": positions_value,
                "total_value": total_value,
            }
        )

    def simulate(
        self,
        start: Union[pd.Timestamp, str],
        end: Union[pd.Timestamp, str],
        starting_capital: float = 1000000.0,
        time_step: Union[pd.Timedelta, str] = pd.Timedelta("1h"),
        verbose: bool = False,
    ) -> SimulationResult:
        """Simulate the strategy over the timeline and return a SimulationResult object."""
        self._reset_strategies()
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        time_step = pd.Timedelta(time_step)

        primary = starting_capital
        secondary = 0.0
        active_positions: List[OptionPosition] = []
        status = SimulationStatus(primary, secondary, active_positions)

        timeline = pd.DataFrame(columns=self.TIMELINE_COLUMNS)
        action_log: List[SimulationAction] = []

        for step, time, spot, signal, chain in self._iterate_timeline(
            start, end, time_step
        ):
            # Check liquidations
            status, liq_actions = self._check_liquidations(step, time, spot, status)

            # Check expirations
            status, exp_actions = self._check_expirations(step, time, spot, status)

            # Execute strategy
            strategy = self.strategies[signal]
            status, strategy_actions = strategy.execute(
                step, time, spot, status, chain, action_log
            )

            # Log status and actions
            timeline.loc[step] = self._get_timeline_row(time, spot, status, chain)
            for action in liq_actions + exp_actions + strategy_actions:
                action_log.append(action)
                if verbose:
                    self._log_message(action.description)

        return SimulationResult(
            timeline=timeline,
            actions=action_log,
        )
