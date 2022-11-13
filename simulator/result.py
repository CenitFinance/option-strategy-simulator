from simulator.action import SimulationAction
from typing import List


import pandas as pd
from simulator.stats import (
    calmar_ratio,
    sortino_ratio,
    sharpe_ratio,
    cagr,
    maximum_drawdown,
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib as mpl


class SimulationResult:
    """Holds the results of a backtest"""
    def __init__(self, timeline: pd.DataFrame, actions: List[SimulationAction]):
        self.timeline = timeline
        self.actions = actions

        timestamp = self.timeline["timestamp"]
        self.timedelta: pd.Timedelta = timestamp.iloc[1] - timestamp.iloc[0]
        annual_periods = pd.Timedelta("1y") / self.timedelta

        value_ts = self.timeline["total_value"]
        spot_ts = self.timeline["spot"]

        buy_and_hold_ammount: float = value_ts.iloc[0] / spot_ts.iloc[0]
        bnh_ts = spot_ts * buy_and_hold_ammount

        self.metrics = {
            "Sharpe": sharpe_ratio(value_ts, annual_periods=annual_periods),
            "Sortino": sortino_ratio(value_ts, annual_periods=annual_periods),
            "CAGR": cagr(value_ts, annual_periods=annual_periods),
            "MDD": maximum_drawdown(value_ts),
            "Calmar": calmar_ratio(value_ts),
        }

        self.benchmark_metrics = {
            "Sharpe": sharpe_ratio(bnh_ts, annual_periods=annual_periods),
            "Sortino": sortino_ratio(bnh_ts, annual_periods=annual_periods),
            "CAGR": cagr(bnh_ts, annual_periods=annual_periods),
            "MDD": maximum_drawdown(bnh_ts),
            "Calmar": calmar_ratio(bnh_ts),
        }

        self.nav = value_ts
        metrics_df = pd.DataFrame(
            {
                metric: [self.metrics[metric], self.benchmark_metrics[metric]]
                for metric in self.metrics.keys()
            }
        )
        self.metrics_df = metrics_df.set_index(
            pd.Index(["Option Selling", "Buy and Hold"]), drop=True
        )
        self.action_log_table = self._get_action_log_table(actions)

    @staticmethod
    def _get_action_log_table(actions: List[SimulationAction]) -> pd.DataFrame:
        action_log_table = pd.DataFrame(
            {
                "timestamp": [action.timestamp for action in actions],
                "step": [action.step for action in actions],
                "spot": [action.spot for action in actions],
                "type": [action.type for action in actions],
                "position_type": [a.position.type for a in actions],
                "position_strike": [a.position.strike for a in actions],
                "position_maturity": [a.position.maturity for a in actions],
                "position_size": [a.position.size for a in actions],
                "position_collateral_primary": [
                    a.position.collateral_primary for a in actions
                ],
                "position_collateral_secondary": [
                    a.position.collateral_secondary for a in actions
                ],
                "position_iv_start": [a.position.iv_start for a in actions],
                "position_price_start": [a.position.price_start for a in actions],
                "liquidity_change": [action.liquidity_change for action in actions],
            }
        )
        return action_log_table

    def plot_simulation(self, title: str, save_location: str = None, figsize=(20, 24)):
        """Plots the backtest results"""
        timeline = self.timeline.copy()

        # Select time series
        time = timeline["timestamp"]
        value = timeline["total_value"]
        spot = timeline["spot"]
        bnh_amount = value.iloc[0] / spot.iloc[0]
        bnh = spot * bnh_amount
        liquidity = timeline["liquidity"]
        positions_value = timeline["positions_value"]
        collateral_locked_value = timeline["total_collaterals"]
        size_long = timeline["size_long_positions"]
        size_short = timeline["size_short_positions"]
        secondary = timeline["secondary"]
        collateral_secondary = timeline["collateral_secondary"]
        delta = timeline["total_delta"]
        positions_delta = timeline["positions_delta"]

        title_size = 25
        tick_labelsize = 12
        axis_labelsize = 14
        legend_labelsize = 12

        grid_color = "gray"
        value_color = "orange"
        spot_color = "purple"
        long_color = "green"
        short_color = "red"

        gs_kw = dict(width_ratios=[1], height_ratios=[3, 2, 2, 2])

        fig, axdict = plt.subplot_mosaic(
            [["value"], ["position"], ["composition"], ["size"]],
            gridspec_kw=gs_kw,
            figsize=figsize,
            constrained_layout=False,
        )

        ax_v, ax_c, ax_p, ax_s = (
            axdict["value"],
            axdict["composition"],
            axdict["position"],
            axdict["size"],
        )

        # Plot value
        ax_v.set_title(title, fontsize=title_size)
        ax_v.plot(time, value, color=value_color, lw=2.5, label="Total Value")
        ax_v.plot(time, bnh, color=spot_color, lw=1.0, ls="-", label="Buy and Hold")

        lylim = ax_v.get_ylim()
        ax_v.set_ylim(bottom=0, top=lylim[1])
        ax_v.set_xlim(left=time.iloc[0], right=time.iloc[-1])

        ax_v.legend(loc="upper left", fontsize=legend_labelsize)
        ax_v.set_ylabel("Value (Primary)", fontsize=axis_labelsize)

        ax_v.grid(which="major", color=grid_color, linestyle="-.", linewidth=0.5)
        ax_v.grid(
            which="minor", color=grid_color, linestyle="-.", linewidth=0.25, alpha=0.7
        )

        # Plot position
        idx_timeline = timeline.set_index("timestamp")
        upper_lim = spot.max() * 10

        max_strike = max(action.position.strike for action in self.actions)
        lylim = ax_p.get_ylim()
        ax_p.set_ylim(bottom=0, top=max(max(spot) * 1.2, max_strike * 1.2))
        ax_p.set_xlim(left=time.iloc[0], right=time.iloc[-1])
        lylim = ax_p.get_xlim()
        x_transform = (
            lambda x: (x - time.iloc[0])
            / (time.iloc[-1] - time.iloc[0])
            * (lylim[1] - lylim[0])
            + lylim[0]
        )

        ax_p.set_ylabel("Spot", fontsize=axis_labelsize)

        ax_p.grid(which="major", color=grid_color, linestyle="-.", linewidth=0.5)
        ax_p.grid(
            which="minor", color=grid_color, linestyle="-.", linewidth=0.25, alpha=0.7
        )

        ax_p.plot(
            [0, 1], [-1, -1], color=long_color, lw=1, label="Long Position Strike"
        )
        ax_p.plot(
            [0, 1], [-1, -1], color=short_color, lw=1, label="Short Position Strike"
        )
        ax_p.scatter(
            [1],
            [-1],
            s=[200],
            color=long_color,
            marker="|",
            label="Delivery Profit",
        )
        ax_p.scatter(
            [1],
            [-1],
            s=[200],
            color=short_color,
            marker="|",
            label="Delivery Loss",
        )

        ax_p.scatter(
            [1],
            [-1],
            s=[80],
            color=long_color,
            marker="^",
            label="Call Short Premium Profit",
        )

        ax_p.scatter(
            [1],
            [-1],
            s=[80],
            color=long_color,
            marker="v",
            label="Put Short Premium Profit",
        )

        ax_p.scatter(
            [1],
            [-1],
            s=[80],
            color=short_color,
            marker="^",
            label="Call Long Premium Loss",
        )

        ax_p.scatter(
            [1],
            [-1],
            s=[80],
            color=short_color,
            marker="v",
            label="Put Long Premium Loss",
        )
        ax_p.plot(time, spot, color=spot_color, lw=1.0, ls="-", label="Underlying")

        ax_p.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=5,
            fontsize=legend_labelsize,
        )
        # ax_p.legend(loc="upper left", fontsize=legend_labelsize)

        recorded_pos = []
        average_premium = sum(
            abs(a.position.price_start * a.position.size) for a in self.actions
        ) / len(self.actions)
        size_func = lambda x: 8 * (x / average_premium) ** 0.5
        for action in self.actions:
            if action.type in ("buy", "sell"):
                pos = action.position
                pos_tuple = (pos.strike, pos.maturity, pos.type)
                start = x_transform(action.timestamp)
                expiration = x_transform(pos.maturity)
                current_spot = idx_timeline.loc[action.timestamp]["spot"]
                final_spot = idx_timeline["spot"].get(pos.maturity, current_spot)
                has_delivery = pos.delivery(final_spot) != 0
                strike = pos.strike
                is_cancellation = pos_tuple in recorded_pos
                premium = abs(pos.price_start * pos.size)
                ms = size_func(premium)
                ls = ":" if is_cancellation else "-"
                m = "^" if pos.is_call() else "v"
                color = long_color if action.type == "buy" else short_color
                prem_color = short_color if action.type == "buy" else long_color

                ax_p.plot([start], [strike], marker=m, markersize=ms, color=prem_color)

                # ax_p.arrow(start, current_spot, 0, strike - current_spot, color=color)
                ax_p.arrow(start, strike, expiration - start, 0, color=color, ls=ls)

                dy = upper_lim if pos.is_call() else -strike
                # ax_p.arrow(expiration, strike, 0, dy, color=color, alpha=0.5, ls=ls)

                if has_delivery:
                    dy = final_spot - strike
                    ax_p.arrow(expiration, strike, 0, dy, color=color, lw=3, ls=ls)
                recorded_pos.append(pos_tuple)

        # Plot composition
        ax_c.set_xlim(left=time.iloc[0], right=time.iloc[-1])
        ax_c.plot(
            time, liquidity, color="darkgreen", lw=2.0, label="Liquidity Available"
        )
        ax_c.plot(
            time,
            collateral_locked_value,
            color="darkblue",
            lw=2.0,
            label="Collateral Value",
        )
        ax_c.plot(
            time,
            positions_value,
            color="darkorange",
            lw=2.0,
            label="Options Value",
        )

        ax_c.legend(loc="upper left", fontsize=legend_labelsize)
        ax_c.set_ylabel("Value (Primary)", fontsize=axis_labelsize)

        ax_c.grid(which="major", color=grid_color, linestyle="-.", linewidth=0.5)
        ax_c.grid(
            which="minor", color=grid_color, linestyle="-.", linewidth=0.25, alpha=0.7
        )

        # Plot size
        ax_s.set_xlim(left=time.iloc[0], right=time.iloc[-1])
        ax_s.plot(
            time,
            size_long,
            color=long_color,
            lw=2.0,
            label="Long Options Position Size",
        )
        ax_s.plot(
            time,
            size_short,
            color=short_color,
            lw=2.0,
            label="Short Options Position Size",
        )
        ax_s.plot(
            time,
            secondary,
            color="violet",
            lw=1.0,
            label="Liquid Secondary",
        )
        ax_s.plot(
            time,
            collateral_secondary,
            color="darkblue",
            lw=1.0,
            label="Collateral in Secondary",
        )
        ax_s.plot(
            time,
            delta,
            color="blue",
            lw=1.0,
            label="Total Delta",
        )
        ax_s.plot(
            time,
            positions_delta,
            color="green",
            lw=1.0,
            label="Options Delta",
        )
        ax_s.plot(
            time,
            [bnh_amount] * len(time),
            color="purple",
            ls="--",
            lw=1.0,
            label="Buy and Hold Delta",
        )

        ax_s.legend(loc="upper left", fontsize=legend_labelsize)
        ax_s.set_ylabel("Position Size", fontsize=axis_labelsize)

        ax_s.grid(which="major", color=grid_color, linestyle="-.", linewidth=0.5)
        ax_s.grid(
            which="minor", color=grid_color, linestyle="-.", linewidth=0.25, alpha=0.7
        )

        if save_location:
            plt.savefig(save_location, dpi=300)
        else:
            plt.show()
