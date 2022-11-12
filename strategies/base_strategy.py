from abc import abstractmethod
from typing import List, Optional, Tuple

import pandas as pd
from simulator.action import SimulationAction
from simulator.chain import OptionChain
from simulator.status import SimulationStatus


class BaseOptionStrategy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def execute(
        self,
        step: int,
        time: pd.Timestamp,
        spot: float,
        status: SimulationStatus,
        chain: OptionChain,
        action_log: List[SimulationAction],
    ) -> Tuple[SimulationStatus, List[SimulationAction]]:
        pass

    def reset(self):
        pass

    # EG note: duplicated method - maybe remove?
    # @staticmethod
    # def _get_last_strategy_action_time(
    #     action_log: List[SimulationAction],
    # ) -> Optional[pd.Timestamp]:
    #     for action in reversed(action_log):
    #         if action.type in ("buy", "sell"):
    #             return action.timestamp
    #     return None

    @staticmethod
    def _get_last_strategy_action_time(
        action_log: List[SimulationAction],
        filter_type: Optional[str] = None,
    ) -> Optional[pd.Timestamp]:
        for action in reversed(action_log):
            if filter_type is None:
                if action.type in ("buy", "sell"):
                    return action.timestamp
            else:
                if action.type == filter_type:
                    return action.timestamp
        return None

    @staticmethod
    def _add_and_rebalance(
        quantity: float, status: SimulationStatus, delta: float, spot: float
    ) -> SimulationStatus:
        total = status.liquidity(spot) + quantity
        status.primary = total * (1 - delta)
        status.secondary = total * delta / spot
        return status
