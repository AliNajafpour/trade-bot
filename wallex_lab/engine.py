"""Cash spot ledger with explicit next-open fills and candle execution limits.

Trade-Bot/scripts/quote_execution.py:make_targets inspired separate requests,
filled events, and known outcomes; prepare_gold.py:audit_minutes inspired strict
input admission. The FX calendar, bid/ask feed and USD/oz cost model are not used.
This is a new fractional spot cash ledger, not a port of gold campaign profits.
"""
import numpy as np
import pandas as pd

from .features import _validated_frame


def backtest(frame, target, config, start=0):
    """Simulate close-known allocations at the next contiguous candle open.

    target is an absolute portfolio fraction, capped by max_exposure. Allocation
    rebalances each bar, after per-side costs. start resets inventory/cash; the
    preceding close may supply the first test order. A missing interval becomes
    known at the first subsequent observed bar. Its stale crossing request is
    cancelled and any held inventory exits at that observed open, including the
    price jump and both-side costs. No historical close fill is invented.
    trades contains fills; roundtrips contains fully closed inventory episodes.
    All profit, loss, fees and notional values use the input quote currency.
    """
    data, timestamps, gaps, interval = _validated_frame(frame)
    target = np.asarray(target, dtype=float)
    n = len(data)
    if target.shape != (n,) or not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("target must be length-N finite allocations in [0,1]")
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or not 0 <= start < n:
        raise ValueError("start must be an in-range integer")
    opens, closes = data.open.to_numpy(), data.close.to_numpy()
    days = timestamps.date
    open_stamps = [stamp.isoformat() for stamp in timestamps[start:]]
    close_stamps = [stamp.isoformat() for stamp in timestamps[start:] + pd.Timedelta(seconds=interval)]
    forced_exits=config.get('forced_exit_indices',[])
    if (not isinstance(forced_exits,(list,tuple)) or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not start<=i<n for i in forced_exits)
            or len(set(forced_exits))!=len(forced_exits)):
        raise ValueError('forced_exit_indices must be unique in-range execution indices')
    forced_exits=set(forced_exits)
    values = {"initial_cash": 5_000_000., "max_exposure": .5, "fee_bps": 10.,
              "slippage_bps": 5., "min_notional": 0.}
    values.update({k: config[k] for k in values if k in config})
    try:
        values = {k: float(v) for k, v in values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("execution configuration must be numeric") from exc
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("execution configuration must be finite")
    initial = values["initial_cash"]
    cap = values["max_exposure"]
    fee = values["fee_bps"] / 10000
    slip = values["slippage_bps"] / 10000
    if initial <= 0 or not 0 <= cap <= 1 or not 0 <= fee < 1 or not 0 <= slip < 1 or values["min_notional"] < 0:
        raise ValueError("cash must be positive; exposure in [0,1]; costs in [0,10000) bps; minimum nonnegative")
    stops = {}
    for key in ("max_drawdown_stop", "daily_loss_stop"):
        limit = config.get(key)
        if limit is not None:
            limit = float(limit)
            if not np.isfinite(limit) or not 0 < limit <= 1:
                raise ValueError(f"{key} must be a fractional limit in (0,1] or None")
        stops[key] = limit
    cash, quantity, basis, realized = initial, 0., 0., 0.
    episode_cost, episode_proceeds, episode_start = 0., 0., None
    fills, episodes, equity = [], [], []
    attempted, rejected = 0, 0
    fees_paid, slippage_paid, turnover = 0., 0., 0.
    risk_events = []
    peak, day_start_equity = initial, initial
    active_day, pending_stop = None, None
    drawdown_blocked, day_blocked = False, False

    def execute(side, amount, reference, i, stamp, reason, signal_index=None):
        nonlocal cash, quantity, basis, realized, attempted, rejected
        nonlocal episode_cost, episode_proceeds, episode_start, fees_paid, slippage_paid, turnover
        if amount <= 0 or reason == "rebalance" and amount * reference <= initial * 1e-12:
            return
        attempted += 1
        if reason == "rebalance" and amount * reference < values["min_notional"]:
            rejected += 1
            return
        price = reference * (1 + slip if side == "buy" else 1 - slip)
        notional = amount * price
        charge = notional * fee
        pnl = 0.
        if side == "buy":
            if quantity == 0:
                episode_start = stamp
            cost = notional + charge
            cash -= cost
            quantity += amount
            basis += cost
            episode_cost += cost
        else:
            allocated_basis = basis * min(amount / quantity, 1.)
            proceeds = notional - charge
            pnl = proceeds - allocated_basis
            cash += proceeds
            quantity -= amount
            basis -= allocated_basis
            realized += pnl
            episode_proceeds += proceeds
            if quantity == 0:
                quantity, basis = 0., 0.
                episodes.append({"entry_timestamp": episode_start, "exit_timestamp": stamp,
                                 "cost": float(episode_cost), "proceeds": float(episode_proceeds),
                                 "pnl": float(episode_proceeds - episode_cost), "exit_reason": reason})
                episode_cost, episode_proceeds, episode_start = 0., 0., None
        if cash < -initial * 1e-10 or quantity < 0:
            raise ArithmeticError("cash/inventory ledger invariant failed")
        cash = max(cash, 0.)
        fees_paid += charge
        slippage_paid += amount * abs(price - reference)
        turnover += notional
        fills.append({"bar_index": int(i), "timestamp": stamp, "signal_index": signal_index,
                      "side": side, "quantity": float(amount), "reference_price": float(reference),
                      "price": float(price), "notional": float(notional), "fee": float(charge),
                      "realized_pnl": float(pnl), "cash_after": float(cash),
                      "quantity_after": float(quantity), "reason": reason})

    for i in range(start, n):
        opening, closing = float(opens[i]), float(closes[i])
        stamp = open_stamps[i - start]
        day = days[i]
        if day != active_day:
            active_day, day_blocked = day, False
            day_start_equity = cash + quantity * opening
        stop_this_open = pending_stop
        if gaps[i]:
            if quantity:
                execute("sell", quantity, opening, i, stamp, "gap_liquidation")
            pending_stop = None
        elif pending_stop:
            if quantity:
                execute("sell", quantity, opening, i, stamp, pending_stop, i - 1)
            pending_stop = None
        if i in forced_exits:
            if quantity:
                execute("sell",quantity,opening,i,stamp,"plan_exit",i-1)
            stop_this_open="plan_exit"
        if i > 0 and not gaps[i] and not stop_this_open and not drawdown_blocked and not day_blocked:
            allocation = min(float(target[i - 1]), cap)
            marked = cash + quantity * opening
            desired_difference = allocation * marked - quantity * opening
            if desired_difference > initial * 1e-12:
                all_in = opening * (1 + slip) * (1 + fee)
                amount = desired_difference / (opening + allocation * (all_in - opening))
                execute("buy", min(amount, cash / all_in), opening, i, stamp, "rebalance", i - 1)
            elif desired_difference < -initial * 1e-12:
                net_sale = opening * (1 - slip) * (1 - fee)
                amount = -desired_difference / (opening - allocation * (opening - net_sale))
                execute("sell", min(amount, quantity), opening, i, stamp, "rebalance", i - 1)
        close_stamp = close_stamps[i - start]
        if i == n - 1 and quantity:
            execute("sell", quantity, closing, i, close_stamp, "end_liquidation")
        marked = cash + quantity * closing
        peak = max(peak, marked)
        for key, reference, blocked in (("max_drawdown_stop", peak, drawdown_blocked),
                                        ("daily_loss_stop", day_start_equity, day_blocked)):
            if stops[key] is not None and not blocked and marked <= reference * (1 - stops[key]):
                risk_events.append({"bar_index": int(i), "timestamp": close_stamp, "reason": key,
                                    "equity": float(marked), "reference_equity": float(reference)})
                if key == "max_drawdown_stop":
                    drawdown_blocked = True
                else:
                    day_blocked = True
                if quantity and pending_stop is None:
                    pending_stop = key
        equity.append({"bar_index": int(i), "timestamp": close_stamp, "cash": float(cash),
                       "quantity": float(quantity), "close": closing, "equity": float(marked),
                       "exposure": float(quantity * closing / marked) if marked else 0.,
                       "realized_pnl": float(realized)})
    curve = np.r_[initial, [row["equity"] for row in equity]]
    drawdown = curve / np.maximum.accumulate(curve) - 1
    daily_closes = pd.Series(curve[1:], index=pd.to_datetime([row["timestamp"] for row in equity], utc=True)).groupby(lambda stamp: stamp.date()).last()
    daily_returns = daily_closes / daily_closes.shift(1, fill_value=initial) - 1
    profits = [r["pnl"] for r in episodes]
    losses = -sum(p for p in profits if p < 0)
    gap_count = int(gaps[start:].sum())
    metrics = {"initial_cash": float(initial), "final_equity": float(cash),
               "total_return": float(cash / initial - 1), "net_profit": float(cash - initial),
               "realized_pnl": float(realized), "max_drawdown": float(-drawdown.min()),
               "order_count": int(attempted + gap_count), "fill_count": len(fills),
               "roundtrip_count": len(episodes), "trade_count": len(episodes),
               "win_rate": float(np.mean(np.asarray(profits) > 0)) if profits else None,
               "profit_factor": float(sum(p for p in profits if p > 0) / losses) if losses else None,
               "fees_paid": float(fees_paid), "slippage_paid": float(slippage_paid),
               "traded_notional": float(turnover), "rejected_min_notional_count": int(rejected),
               "forced_subminimum_exit_count": sum(fill["side"] == "sell" and fill["reason"] != "rebalance"
                                                     and fill["notional"] < values["min_notional"] for fill in fills),
               "gap_count": gap_count, "cancelled_gap_orders": gap_count,
               "start_index": int(start), "bars": n - int(start), "interval_seconds": interval,
               "fee_bps_per_side": values["fee_bps"], "slippage_bps_per_side": values["slippage_bps"],
               "max_exposure": cap, "min_notional": values["min_notional"],
               **stops, "drawdown_stop_triggered": drawdown_blocked,
               "daily_loss_stop_count": sum(event["reason"] == "daily_loss_stop" for event in risk_events),
               "daily_returns": {str(k): float(v) for k, v in daily_returns.items()},
               "mean_daily_return": float(daily_returns.mean()),
               "best_daily_return": float(daily_returns.max()), "worst_daily_return": float(daily_returns.min()),
               "execution_limits": ["Candle open/close proxy fills; no depth, queue, volume participation or precision model",
                                    "Fractional quantities are unrounded; venue quantity precision is unmodeled",
                                    "Forced exits ignore venue minima; simulated final cash is not guaranteed account-executable",
                                    "Exposure cap enforced at execution; subsequent price moves can increase marked exposure",
                                    "Drawdown sampled at candle closes; intrabar risk is unobserved",
                                    "Optional risk stops observe closes and fill next observed open; losses may exceed limits",
                                    "Daily risk resets on UTC candle-open dates; a pending exit still fills next open",
                                    "Missing intervals cancel stale orders and liquidate held inventory at the first observed open",
                                    "Inventory remains exposed during missing intervals; unavailable prices and risk breaches are unobserved",
                                    "Final candle close liquidates inventory; final close signal cannot execute"]}
    return {"metrics": metrics, "trades": fills, "equity": equity, "roundtrips": episodes,
            "risk_events": risk_events}
