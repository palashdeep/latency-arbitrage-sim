################################################################################################################################################
# Simulation of a simple latency-arbitrage scenario
# - Market fundamental value V_t follows a random walk
# - Market-maker posts quotes based on the last observed value (V_{t-1}) -> stale quotes when V jumps
# - Fast trader reacts instantly (latency = 0) and trades against stale quotes when profitable
# - Slow trader reacts after `slow_latency` steps and trades then (may be less profitable)
# This model is intentionally simple/transparent to illustrate the PnL advantage of speed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from caas_jupyter_tools import display_dataframe_to_user

np.random.seed(42)

def simulate_latency_arbitrage(T=200000, sigma=0.5, spread=0.5, fee=0.0, slow_latency=5):
    """
    T : number of time steps
    sigma : std dev of fundamental shocks (epsilon_t)
    spread : constant spread posted by market-maker (s)
    fee : per-share transaction cost
    slow_latency : integer delay (in steps) for the slow trader
    Returns DataFrame with per-trade profits for fast and slow traders and summary stats.
    """
    # generate fundamentals V (start at 0)
    eps = np.random.normal(0, sigma, size=T)
    V = np.empty(T+1)
    V[0] = 0.0
    for t in range(1, T+1):
        V[t] = V[t-1] + eps[t-1]
    # quotes posted at time t are based on V[t-1]:
    # ask_t = V[t-1] + spread/2, bid_t = V[t-1] - spread/2
    ask = V[:-1] + spread/2
    bid = V[:-1] - spread/2

    fast_profits = []
    slow_profits = []
    fast_trades_idx = []
    slow_trades_idx = []

    # For each "event" at time t (1..T), a jump eps[t] occurs (V[t] - V[t-1])
    # Fast trader acts immediately at price ask_{t} or bid_{t} (which are based on V[t-1])
    # He profits if the jump magnitude exceeds spread/2 (so V[t] crosses the stale quote)
    for t in range(1, T+1):
        jump = V[t] - V[t-1]  # equals eps[t-1]
        # Upward move: profitable to buy at stale ask if V[t] > ask_{t-1}
        if jump > 0 and jump > spread/2:
            # Fast trader buys 1 share at ask based on V[t-1]
            fast_profit = V[t] - ask[t-1] - fee
            fast_profits.append(fast_profit)
            fast_trades_idx.append(t)
            # schedule slow trader to act at t + slow_latency (if within range)
            tau = t + slow_latency
            if tau <= T:
                slow_profit = V[tau] - ask[tau-1] - fee
                slow_profits.append(slow_profit)
                slow_trades_idx.append(tau)
        # Downward move: profitable to sell at stale bid if V[t] < bid_{t-1}
        elif jump < 0 and -jump > spread/2:
            # Fast trader sells 1 share at bid
            fast_profit = bid[t-1] - V[t] - fee  # profit from selling high and value being lower
            fast_profits.append(fast_profit)
            fast_trades_idx.append(t)
            tau = t + slow_latency
            if tau <= T:
                slow_profit = bid[tau-1] - V[tau] - fee
                slow_profits.append(slow_profit)
                slow_trades_idx.append(tau)

    # Build DataFrame of trades
    df_fast = pd.DataFrame({
        'time': fast_trades_idx,
        'profit': fast_profits
    })
    df_slow = pd.DataFrame({
        'time': slow_trades_idx,
        'profit': slow_profits
    })
    return V, df_fast, df_slow

def confidence_interval(data, alpha=0.05):
    """Compute mean and (1-alpha) confidence interval using normal approximation."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    se = std / np.sqrt(n)   # standard error
    z = 1.96  # ~95% CI
    lower = mean - z * se
    upper = mean + z * se
    return mean, (lower, upper)

# Run simulation with default parameters:
pnl_adv = []
fast_profits = []
slow_profits = []
for seed in range(50):
    V, df_fast, df_slow = simulate_latency_arbitrage(T=1000, sigma=10, spread=2, fee=0.1, slow_latency=5, seed=seed)
    fast_profits.append(df_fast['profit'].sum())
    slow_profits.append(df_slow['profit'].sum())
    pnl_adv.append(fast_profits[-1] - slow_profits[-1])

# Compute confidence intervals
fast_mean, fast_ci = confidence_interval(fast_profits)
slow_mean, slow_ci = confidence_interval(slow_profits)
adv_mean, adv_ci = confidence_interval(pnl_adv)

print("Fast trader total PnL: mean = %.2f, 95%% CI = [%.2f, %.2f]" % (fast_mean, fast_ci[0], fast_ci[1]))
print("Slow trader total PnL: mean = %.2f, 95%% CI = [%.2f, %.2f]" % (slow_mean, slow_ci[0], slow_ci[1]))
print("PnL Advantage: mean = %.2f, 95%% CI = [%.2f, %.2f]" % (adv_mean, adv_ci[0], adv_ci[1]))

# Summary statistics
def summary(df):
    return {
        'n_trades': len(df),
        'total_pnl': df['profit'].sum(),
        'mean_pnl_per_trade': df['profit'].mean() if len(df)>0 else 0.0,
        'std_pnl_per_trade': df['profit'].std() if len(df)>0 else 0.0
    }

fast_stats = summary(df_fast)
slow_stats = summary(df_slow)

summary_df = pd.DataFrame([fast_stats, slow_stats], index=['fast', 'slow']).T
# display_dataframe_to_user("Latency Arbitrage Summary", summary_df)

# Show per-trade profit distribution (histograms)
plt.figure(figsize=(8,4))
plt.hist(df_fast['profit'].clip(-2,2), bins=80)  # clip for visibility
plt.title('Fast trader per-trade profit distribution (clipped to [-2,2])')
plt.xlabel('Profit per trade')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df_slow['profit'].clip(-2,2), bins=80)
plt.title('Slow trader per-trade profit distribution (clipped to [-2,2])')
plt.xlabel('Profit per trade')
plt.ylabel('Count')
plt.show()

# Show running cumulative PnL over time (sampled)
def cumulative_pnl_over_time(df, T):
    arr = np.zeros(T+1)
    for idx, p in zip(df['time'], df['profit']):
        arr[idx] += p
    return np.cumsum(arr)

cum_fast = cumulative_pnl_over_time(df_fast, len(V)-1)
cum_slow = cumulative_pnl_over_time(df_slow, len(V)-1)

plt.figure(figsize=(10,4))
plt.plot(cum_fast, label='fast')
plt.plot(cum_slow, label='slow')
plt.legend()
plt.title('Cumulative PnL over time (fast vs slow)')
plt.xlabel('time step')
plt.ylabel('Cumulative PnL')
plt.show()

# Provide numeric summary as a small DataFrame
numeric_summary = pd.DataFrame({
    'metric': ['n_trades', 'total_pnl', 'mean_pnl_per_trade', 'std_pnl_per_trade'],
    'fast': [fast_stats['n_trades'], fast_stats['total_pnl'], fast_stats['mean_pnl_per_trade'], fast_stats['std_pnl_per_trade']],
    'slow': [slow_stats['n_trades'], slow_stats['total_pnl'], slow_stats['mean_pnl_per_trade'], slow_stats['std_pnl_per_trade']]
})
# display_dataframe_to_user("Numeric results (fast vs slow)", numeric_summary)


"""
Experiment harness for latency-arb simulation.

Assumptions (adapt names if yours differ):
- FastTrader(decision args...) has .decide_and_order(mm_quote, V, t, estimated_vars) -> order dict or None
- SlowTrader(...).decide_and_order(mm_quote, V, t, estimated_vars, signal_history) -> order dict or None
- Trader objects have .apply_fill(side, filled, avg_price)
- MM objects have .get_quote(V, t), .execute_market_order(side,size) and .end_of_timestep_update()
- All factories produce fresh instances for every run

Usage examples at bottom.
"""

import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, List, Tuple
from utils import ci_mean, get_stock_price_random_walk
from fast_Trader import fast_Trader
from slow_Trader import slow_Trader
from naive_MM import naive_MM
from smart_MM import smart_MM
from market import market
# -----------------------
# Core single-run runner
# -----------------------
def run_single_experiment(
    T,
    seed,
    mm_factory,
    fast_factory,
    slow_factory,
    market_factory,
    verbose = False,
    **kwargs
):
    """
    Run one simulation.
    - T: timesteps
    - seed: RNG seed for reproducibility
    - mm_factory, fast_factory, slow_factory: functions returning fresh instances
    - setup_signals(V) optional: transform true V into public signals (default identity)
    - estimated_vars: dict passed to traders (process_var, signal_var) - optional
    Returns dict with final PnL per agent and fills df.
    """
    np.random.seed(seed)
    random.seed(seed)

    # -------------------
    # Generate true value V (simple random walk)
    # -------------------
    sigma = 0.6  # you can expose as parameter
    V0 = 100.0
    V = get_stock_price_random_walk(V0, T, sigma=sigma)

    # -------------------
    # Instantiate agents
    # -------------------
    mm = mm_factory()
    fast = fast_factory()
    slow = slow_factory()
    market = market_factory()

    trader_registry = {slow.name: slow, fast.name: fast,}

    fills = []
    # main loop
    for t in range(1, T-1):
        # mm posts quote
        mm_quote = mm.get_quote(V, t)
        # traders decide
        orders = []
        # fast: decide_and_order(mm_quote, V, t, estimated_vars)
        o_fast = fast.decide_and_order(t, V, market, mm_quote)
        if o_fast: orders.append(o_fast)

        # slow: decide_and_order(mm_quote, V, t, estimated_vars, signal_history)
        o_slow = slow.decide_and_order(t, V, market, mm_quote)
        if o_slow: orders.append(o_slow)

        # If no orders -> continue
        if not orders:
            mm.end_of_timestep_update()
            continue

        # market clears orders
        fills_t = market.clear_market_orders(mm)
        
        for fill in fills_t:
            # Apply fill to trader
            filled = fill.get("filled", 0.0)
            avg_price = fill.get("avg_price", 0.0)
            trader = trader_registry[fill["trader"]]
            trader.apply_fill(fill["side"], filled, avg_price)
        
        fills.extend(fills_t)
        # end-of-timestep mm refill / housekeeping
        mm.end_of_timestep_update()

    # Mark-to-market PnL
    final_V = float(V[-1])
    fast_pnl = getattr(fast, "cash", 0.0) + getattr(fast, "inventory", 0.0) * final_V
    slow_pnl = getattr(slow, "cash", 0.0) + getattr(slow, "inventory", 0.0) * final_V
    mm_pnl = getattr(mm, "cash", 0.0) + getattr(mm, "inventory", 0.0) * final_V

    result = {
        "fast_pnl": float(fast_pnl),
        "slow_pnl": float(slow_pnl),
        "mm_pnl": float(mm_pnl),
        "fills": pd.DataFrame(fills)
    }
    return result

# -----------------------
# Multi-run wrapper (compute CIs)
# -----------------------
def run_multi(
    n_runs,
    T,
    mm_factory,
    fast_factory,
    slow_factory,
    market_factory,
    **run_kwargs
):
    rows = []
    details = []
    for i in range(n_runs):
        seed = run_kwargs.pop("seed", 1000) + i
        out = run_single_experiment(T=T, seed=seed, mm_factory=mm_factory,
                                    fast_factory=fast_factory, slow_factory=slow_factory, market_factory=market_factory,
                                    **run_kwargs)
        details.append(out)
        rows.append({"fast_pnl": out["fast_pnl"], "slow_pnl": out["slow_pnl"], "mm_pnl": out["mm_pnl"]})
    df = pd.DataFrame(rows)
    return df, details

# -----------------------
# Experiment: latency sweep
# -----------------------
def latency_sweep(
    latencies,
    n_runs,
    T,
    mm_factory_base,
    fast_factory_builder,
    slow_factory,
    market_factory,
    **runner_kwargs
):
    """
    Sweep fast trader latency values and return summary DF with mean and 95% CI.
    - fast_factory_builder(latency) -> a zero-arg factory that returns a FastTrader with given latency
    - mm_factory_base: zero-arg factory to create MM (same across runs)
    - slow_factory: zero-arg factory for slow trader
    """
    summary_rows = []
    for L in latencies:
        print(f"Running latency {L} ...")
        fast_factory = fast_factory_builder(L)
        df, det = run_multi(n_runs=n_runs, T=T, mm_factory=mm_factory_base, fast_factory=fast_factory, slow_factory=slow_factory, market_factory=market_factory, **runner_kwargs)
        mean_fast, ci_fast = ci_mean(df["fast_pnl"].tolist())
        mean_slow, ci_slow = ci_mean(df["slow_pnl"].tolist())
        mean_mm, ci_mm = ci_mean(df["mm_pnl"].tolist())
        summary_rows.append({
            "latency": L,
            "fast_mean": mean_fast, "fast_ci_lower": ci_fast[0], "fast_ci_upper": ci_fast[1],
            "slow_mean": mean_slow, "slow_ci_lower": ci_slow[0], "slow_ci_upper": ci_slow[1],
            "mm_mean": mean_mm, "mm_ci_lower": ci_mm[0], "mm_ci_upper": ci_mm[1],
        })
    return pd.DataFrame(summary_rows)

# -----------------------
# Experiment: compare MM types
# -----------------------
def compare_mm_types(
    mm_factories,
    n_runs,
    T,
    fast_factory,
    slow_factory,
    market_factory,
    **runner_kwargs
):
    rows = []
    for name, mm_fac in mm_factories.items():
        print("Running MM type:", name)
        df, _ = run_multi(n_runs=n_runs, T=T, mm_factory=mm_fac, fast_factory=fast_factory, slow_factory=slow_factory, market_factory=market_factory, **runner_kwargs)
        m_fast, cif = ci_mean(df["fast_pnl"].tolist())
        m_slow, cis = ci_mean(df["slow_pnl"].tolist())
        m_mm, cim = ci_mean(df["mm_pnl"].tolist())
        rows.append({"mm": name, "fast_mean": m_fast, "fast_ci_lo": cif[0], "fast_ci_hi": cif[1],
                     "slow_mean": m_slow, "slow_ci_lo": cis[0], "slow_ci_hi": cis[1],
                     "mm_mean": m_mm, "mm_ci_lo": cim[0], "mm_ci_hi": cim[1]})
    return pd.DataFrame(rows)

# -----------------------
# Plotting helpers
# -----------------------
def plot_latency_results(df_summary):
    plt.figure(figsize=(8,5))
    x = df_summary["latency"].values
    plt.errorbar(x, df_summary["fast_mean"], yerr=[df_summary["fast_mean"]-df_summary["fast_ci_lower"], df_summary["fast_ci_upper"]-df_summary["fast_mean"]],
                 fmt='-o', label='fast PnL (mean ± 95% CI)')
    plt.errorbar(x, df_summary["slow_mean"], yerr=[df_summary["slow_mean"]-df_summary["slow_ci_lower"], df_summary["slow_ci_upper"]-df_summary["slow_mean"]],
                 fmt='-s', label='slow PnL (mean ± 95% CI)')
    plt.xlabel("Fast trader latency (timesteps)")
    plt.ylabel("Total PnL (mean ± 95% CI)")
    plt.legend(); plt.grid(True); plt.show()

def plot_mm_comparison(df_summary):
    x = df_summary["mm"]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x, df_summary["fast_mean"], yerr=[df_summary["fast_mean"]-df_summary["fast_ci_lo"], df_summary["fast_ci_hi"]-df_summary["fast_mean"]], alpha=0.6, label='fast')
    ax.bar(x, df_summary["slow_mean"], yerr=[df_summary["slow_mean"]-df_summary["slow_ci_lo"], df_summary["slow_ci_hi"]-df_summary["slow_mean"]], alpha=0.6, label='slow', bottom=df_summary["fast_mean"])
    ax.set_ylabel("Mean PnL"); ax.set_title("PnL by MM type"); ax.legend()
    plt.show()

# -----------------------
# Example usage (adapt factories to your constructors)
# -----------------------
if __name__ == "__main__":
    # Example factories using your real constructors:
    # Replace the lambdas below with your actual constructors or wrappers.
    def mm_naive_factory():
        # return NaiveMM(...)  # use your real class
        return naive_MM()  # placeholder: adapt parameters

    def mm_smart_factory():
        # return SmartMM(...)  # your real class
        return smart_MM()

    def fast_factory_builder(latency):
        def make():
            # return FastTrader(name="fast", latency=latency, noise=..., base_size=..., ...)
            return fast_Trader(name="fast", latency=latency)
        return make

    def slow_factory():
        # return SlowTrader(name="slow", latency=5, noise=0.5, base_size=8.0, agg_span=10)
        return slow_Trader(name="slow", latency=5)
    
    def market_factory():
        # return Market(...)  # your real market class
        return market()  # placeholder

    # Run a small latency sweep example (quick) - adjust n_runs/T for real experiments
    latencies = [0, 1, 2, 3, 5, 10]
    df_latency = latency_sweep(latencies, n_runs=10, T=800,
                               mm_factory_base=mm_smart_factory,
                               fast_factory_builder=fast_factory_builder,
                               slow_factory=slow_factory, market_factory=market_factory)
    print(df_latency)
    plot_latency_results(df_latency)

    # Compare MM types
    mm_factories = {"naive": mm_naive_factory, "smart": mm_smart_factory}
    df_mm = compare_mm_types(mm_factories, n_runs=10, T=800,
                             fast_factory=fast_factory_builder(0),
                             slow_factory=slow_factory, market_factory=market_factory)
    print(df_mm)
    # plot_mm_comparison(df_mm)

# Summary statistics
# def summary(df):
#     return {
#         'n_trades': len(df),
#         'total_pnl': df['profit'].sum(),
#         'mean_pnl_per_trade': df['profit'].mean() if len(df)>0 else 0.0,
#         'std_pnl_per_trade': df['profit'].std() if len(df)>0 else 0.0
#     }

# fast_stats = summary(df_fast)
# slow_stats = summary(df_slow)

# summary_df = pd.DataFrame([fast_stats, slow_stats], index=['fast', 'slow']).T
# # display_dataframe_to_user("Latency Arbitrage Summary", summary_df)

# # Show per-trade profit distribution (histograms)
# plt.figure(figsize=(8,4))
# plt.hist(df_fast['profit'].clip(-2,2), bins=80)  # clip for visibility
# plt.title('Fast trader per-trade profit distribution (clipped to [-2,2])')
# plt.xlabel('Profit per trade')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(8,4))
# plt.hist(df_slow['profit'].clip(-2,2), bins=80)
# plt.title('Slow trader per-trade profit distribution (clipped to [-2,2])')
# plt.xlabel('Profit per trade')
# plt.ylabel('Count')
# plt.show()

# cum_fast = cumulative_pnl_over_time(df_fast, len(V)-1)
# cum_slow = cumulative_pnl_over_time(df_slow, len(V)-1)

# plt.figure(figsize=(10,4))
# plt.plot(cum_fast, label='fast')
# plt.plot(cum_slow, label='slow')
# plt.legend()
# plt.title('Cumulative PnL over time (fast vs slow)')
# plt.xlabel('time step')
# plt.ylabel('Cumulative PnL')
# plt.show()

# # Provide numeric summary as a small DataFrame
# numeric_summary = pd.DataFrame({
#     'metric': ['n_trades', 'total_pnl', 'mean_pnl_per_trade', 'std_pnl_per_trade'],
#     'fast': [fast_stats['n_trades'], fast_stats['total_pnl'], fast_stats['mean_pnl_per_trade'], fast_stats['std_pnl_per_trade']],
#     'slow': [slow_stats['n_trades'], slow_stats['total_pnl'], slow_stats['mean_pnl_per_trade'], slow_stats['std_pnl_per_trade']]
# })