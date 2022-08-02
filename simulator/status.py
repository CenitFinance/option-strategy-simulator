from dataclasses import dataclass
from typing import List

import pandas as pd
from simulator.chain import OptionChain
from simulator.position import OptionPosition


@dataclass
class SimulationStatus:
    primary: float
    secondary: float
    positions: List[OptionPosition]

    def liquidity(self, spot: float) -> float:
        return self.primary + self.secondary * spot

    def num_long_positions(self) -> int:
        return sum(1 for p in self.positions if p.is_long())

    def num_short_positions(self) -> int:
        return sum(1 for p in self.positions if p.is_short())

    def size_long_positions(self) -> int:
        return abs(sum(p.size for p in self.positions if p.is_long()))

    def size_short_positions(self) -> int:
        return abs(sum(p.size for p in self.positions if p.is_short()))

    def collateral_primary(self) -> float:
        return sum(p.collateral_primary for p in self.positions)

    def collateral_secondary(self) -> float:
        return sum(p.collateral_secondary for p in self.positions)

    def total_value(self, spot: float, time: pd.Timestamp) -> float:
        liquidity = self.liquidity(spot)
        positions_value = sum(p.valuation(spot, time) for p in self.positions)
        collaterals = sum(p.collateral_value(spot) for p in self.positions)
        return liquidity + positions_value + collaterals

    def positions_delta(
        self, spot: float, time: pd.Timestamp, chain: OptionChain
    ) -> float:
        return sum(p.delta(spot, time, chain) for p in self.positions)

    def total_delta(self, spot: float, time: pd.Timestamp, chain: OptionChain) -> float:
        liquidity_delta = self.secondary
        positions_delta = self.positions_delta(spot, time, chain)
        collateral_delta = self.collateral_secondary()
        return liquidity_delta + positions_delta + collateral_delta
