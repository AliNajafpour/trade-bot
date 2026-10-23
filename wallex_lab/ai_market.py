"""Financial-model opportunity discovery; indicator templates do not decide entries."""
import hashlib
import json
import math

import numpy as np
import pandas as pd
from . import fast_market
from .spot import canonical_symbol


class AIMarket:
    def __init__(self, config=None, bank=None):
        self.config = dict(config or {})
        if bank is None:
            if self.config.get('benchmark_case_dir'):
                from .benchmark_signals import BenchmarkBank
                bank = BenchmarkBank(self.config['benchmark_case_dir'], self.config.get('benchmark_combo'),
                    device=self.config.get('model_device', 'cpu'), margin=self.config.get('benchmark_margin'),
                    fee_profile=self.config.get('benchmark_fee_profile', 'tmn_level1_taker'))
            else:
                from .model_signals import ForecastBank
                bank = ForecastBank(root=self.config.get('model_root', 'runs/wallex-20261021-causal'),
                    device=self.config.get('model_device', 'cpu'),
                    kronos_root=self.config.get('kronos_root', 'runs/orchestrator-model-bank'))
        self.bank = bank

    def evaluate(self, frame, symbol, asof, book, config):
        config = {**self.config, **config}
        # Reuse causal feature/freshness validation, not the old indicator entry rules.
        result = fast_market.evaluate(frame, symbol, asof, book, config)
        eligible = result['status'] == 'setup' or (result['status'] == 'no_setup'
            and result['reason'] == 'latest completed candle has no eligible setup')
        if not eligible:
            return result
        interval = result['clock']['interval_seconds']
        clocks = pd.to_datetime(frame.timestamp, utc=True)
        complete = frame.loc[clocks + pd.Timedelta(seconds=interval) <= pd.Timestamp(asof)].copy()
        complete.attrs['interval_seconds'] = interval
        predictions = self.bank.predict(complete, canonical_symbol(symbol).replace('/', ''))
        result['model_forecasts'] = predictions
        result['setup'] = None
        if not predictions.get('model_ready') or predictions.get('status') != 'ok':
            result.update(status='unavailable', reason='required trained financial models unavailable')
            return result
        models = predictions['models']
        research = predictions.get('research_candidate')
        required = set(predictions.get('required_models', ('hgb', 'gru', 'kronos')))
        if set(models) != required or len({m['horizon_bars'] for m in models.values()}) != 1:
            result.update(status='unavailable', reason='three aligned financial models required')
            return result
        fee, slip = float(config.get('fee_bps', 20)), float(config.get('slippage_bps', 15))
        cost = 2 * (fee + slip) / 10000
        gross = np.array([models[name]['gross_return'] for name in sorted(required)], dtype=float)
        if not np.isfinite(gross).all():
            result.update(status='unavailable', reason='nonfinite model forecasts')
            return result
        net = gross - cost
        votes = int(np.sum(net > 0))
        median = float(predictions['aggregate_gross_return'] - cost) if research else float(np.median(net))
        if research:
            predictions.update(aggregate_net_return=median, agreement_positive=votes,
                current_roundtrip_cost_fraction=cost,
                current_cost_assumptions=dict(fee_bps_per_side=fee, slippage_bps_per_side=slip))
            for model in models.values():
                model['net_return'] = float(model['gross_return'] - cost)
        horizon = int(next(iter(models.values()))['horizon_bars'])
        hold = min(horizon, int(21600 / interval)) if research else min(24, int(21600 / interval))
        result['summary'].update({name + '_gross_return': float(models[name]['gross_return']) for name in models})
        result['summary'].update(model_median_net_return=median, model_disagreement=float(np.std(net)),
            model_positive_votes=votes, model_horizon_bars=horizon, model_horizon_minutes=horizon * interval / 60)
        margin = float(predictions.get('entry_margin', 0)) if research else 0.
        if research:
            result['summary'].update(model_entry_margin=margin, model_gross_return_threshold=cost + margin,
                model_margin_after_cost=median - margin, model_current_roundtrip_cost=cost)
        if (not research and votes < 2) or median <= margin:
            result.update(status='no_setup', reason='financial-model consensus does not clear current roundtrip costs')
            return result
        summary = result['summary']
        stop = max(.003, min(.02, 2 * summary['ATR_fraction']))
        target = min(.08, max(.65 * summary['range_48'], 1.5 * summary['volatility_48'] * math.sqrt(hold)))
        rr = (target - cost) / (stop + cost)
        if rr < float(config.get('minimum_net_rr', 1)):
            result.update(status='no_setup', reason='model opportunity fails independent risk geometry')
            return result
        bars = complete.tail(48)
        positive_volume = bars.volume.iloc[-12:]
        positive_volume = positive_volume[positive_volume > 0]
        close = float(bars.close.iloc[-1])
        candle = result['clock']['candle_close']
        model_identity = predictions.get('model_identity', {name: model.get('artifact', {}).get('sha256') for name, model in models.items()})
        fingerprint = hashlib.sha256(json.dumps([symbol, candle, model_identity], sort_keys=True).encode()).hexdigest()
        result.update(status='setup', reason='explicit research model combination exceeds current costs and frozen margin' if research else 'three financial models evaluated; majority forecast exceeds current costs', fingerprint=fingerprint,
            setup=dict(template='financial_model_consensus', family='financial_model_consensus', symbol=symbol,
                reference_price=close, current_close=close, current_volume=float(bars.volume.iloc[-1]),
                stop_fraction=stop, target_fraction=target, roundtrip_cost_fraction=cost,
                reward_risk_after_cost=rr, potential_net_reward_fraction=target - cost,
                confirmation_price=float(bars.high.iloc[-4:].max()) * 1.0005,
                reclaim_price=max(float(bars.close.iloc[-12:].mean()) * 1.0005, close * 1.001),
                minimum_volume=float(positive_volume.median()) if len(positive_volume) else None,
                timestamp=candle, max_hold_bars=hold,
                geometry_caveat='Forecasts are uncalibrated model estimates; risk target geometry is not expected profit'))
        if research:
            result['setup'].update(research_candidate=research, horizon_bars=horizon,
                horizon_minutes=horizon * interval / 60, arm_price=close * 1.001,
                expires_at=(pd.Timestamp(candle) + pd.Timedelta(seconds=hold * interval)).isoformat())
        return result
