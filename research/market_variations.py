import uuid
import pandas as pd
import numpy as np

class market:
    def __init__(self):
        """
        mm : Market Maker object with method get_quote(V, t) and execute_market_order(side, size)
        """
        self.order_book = []  # list of pending market orders

    def submit_market_order(self, t, trader, mm_quote, side, size):
        """
        side : "buy" or "sell"
        size : float
        Executes a market order against the market maker's current quote.
        Returns a dict with keys:
            - filled: float
            - avg_price: float
            - mm_cash_change: float
            - mm_inventory_change: float
        """
        order_id = str(uuid.uuid4())

        self.order_book.append({
            "order_id": order_id,
            "arrival_time": t,
            "trader": trader.name,
            "side": side,
            "size": size,
            "latency": trader.lag,
            "quote": mm_quote
        })

        return order_id

    def clear_market_orders(self, market_maker):
        """
        orders: list of dicts with keys:
          - trader: Trader object
          - side: "buy" or "sell"
          - size: float
          - latency: float (lower is earlier)
          - order_id: optional id
        Orders are processed for the current self.time using MM.get_quote(V, time).
        """
        orders = self.order_book
        self.order_book = []

        # Add tiny jitter to break ties; sort by latency then jitter
        for o in orders:
            o["_sort_key"] = o.get("latency", 0.0) + np.random.normal(0, 1e-6)

        orders_sorted = sorted(orders, key=lambda x: x["_sort_key"])

        fills = []
        for o in orders_sorted:
            trader = o["trader"]
            side = o["side"]
            size = o["size"]
            # Execute against MM
            fill = market_maker.execute_market_order(side, size)
            
            fills.append({
                "order_id": o["order_id"],
                "trader": trader,
                "side": side,
                "requested": size,
                "filled": fill["filled"],
                "avg_price": fill["avg_price"],
            })

        return fills
    
import numpy as np

class naive_MM:

    def __init__(self, lag=10, spread=5, freq=5, inventory=0, cash=0, start=100.0):
        """A naive Market Maker that quotes around a lagged mid-price with fixed spread."""
        self.lag = lag
        self.spread = spread
        self.freq = freq
        self.inventory = inventory
        self.mids = [start]
        self.quotes = [(start - spread/2, start + spread/2)]
        self.cash = cash
        self.depth_at_best = 500  # fixed depth at best quotes

    def get_quote(self, V, t):
        """Simulate a naive Market Maker"""
        eps = np.random.normal(0, 0.5)
        if t % self.freq == 0:
            mid = V[t-self.lag] + eps
            self.mids.append(mid)
            quote = (mid - self.spread/2, mid + self.spread/2)
            self.quotes.append(quote)
        
        return self.quotes[-1]
    
    # def execute_order(self, size, buy=True):
    #     """Order execution"""
    #     bid, ask = self.quotes[-1]
    #     if buy:
    #         self.inventory -= size
    #         self.cash += ask * size
    #         return True
    #     elif not buy:
    #         self.inventory += size
    #         self.cash -= bid * size
    #         return True
    #     return False

    def execute_market_order(self, side: str, size: float):
        """
        side: "buy" means trader buys from MM (MM sells)
              "sell" means trader sells to MM (MM buys)
        size: requested size
        Returns: dict with fill details (filled_size, avg_price, mm_cash_change, mm_inventory_change, trader_fill_info)
        """
        bid, ask = self.quotes[-1]
        filled = 0.0
        cash_change = 0.0
        inv_change = 0.0
        depth = self.depth_at_best
        # Determine execution price schedule
        if side == "buy":
            # Trader buys at ask: MM sells
            # Fill up to depth at ask
            take = min(size, depth)
            filled += take
            cash_change += take * ask    # MM receives cash (selling)
            inv_change -= take           # MM sold => inventory decreases
            remaining = size - take
            if remaining > 0:
                # fill remaining at linearly worse price: ask + impact_per_unit * (units_over / depth)
                penalty = self.impact_per_unit * (remaining / max(depth,1.0))
                worse_price = ask + penalty
                filled += remaining
                cash_change += remaining * worse_price
                inv_change -= remaining
        elif side == "sell":
            # Trader sells to MM at bid: MM buys
            take = min(size, depth)
            filled += take
            cash_change -= take * bid   # MM pays cash to buy
            inv_change += take
            remaining = size - take
            if remaining > 0:
                penalty = self.impact_per_unit * (remaining / max(depth,1.0))
                worse_price = bid - penalty
                filled += remaining
                cash_change -= remaining * worse_price
                inv_change += remaining
        else:
            raise ValueError("side must be 'buy' or 'sell'")
        
        # Update MM state
        self.cash += cash_change
        self.inventory += inv_change

        avg_price = cash_change / filled if filled > 0 else 0.0
        return {
            "filled": filled,
            "avg_price": avg_price,
            "mm_cash_change": cash_change,
            "mm_inventory_change": inv_change
        }
    
    def end_of_timestep_update(self):
        """Refill depth at timestep boundary (call once per simulation step)."""
        return None
    
    def mark_to_market(self, V, t):
        """Current PnL = cash + inventory * true value"""
        return self.cash + self.inventory * V[t]