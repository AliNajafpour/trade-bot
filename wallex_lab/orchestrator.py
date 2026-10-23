"""Event-driven specialist coordination; the only execution backend is paper."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import time

import pandas as pd
from . import fast_market, specialists
from .context import timestamp
from .features import _validated_frame
from .fees import DEFAULT_PATH as DEFAULT_FEES_PATH, official_public_rates
from .locking import exclusive
from .spot import PaperBroker, canonical_symbol, load_exclusions, market_allowed, size_order, _position_limit

DEFAULT_EXCLUSIONS = Path(__file__).resolve().parents[1] / 'config/credit-excluded-markets.txt'


def entry_decision(analysis):
    """Only the final cross-specialist decision can authorize a plan."""
    decision = analysis.get('decision', {})
    if analysis.get('status') != 'ok' or decision.get('status') != 'observed':
        raise ValueError('A valid final specialist decision is required')
    if decision.get('stance') not in ('enter', 'watch', 'reject'):
        raise ValueError('Invalid final stance')
    if decision['stance'] == 'reject':
        return decision
    if analysis.get('regime', {}).get('stance') == 'reject':
        raise ValueError('Technical rejection cannot be overridden')
    for role in ('sentiment', 'fundamentals'):
        if analysis.get(role, {}).get('event_risk') == 'high':
            raise ValueError('High event risk blocks new buys')
    trigger = decision.get('trigger')
    if trigger not in ('immediate', 'breakout_with_volume', 'reclaim_mean_with_volume') or (
            (decision['stance'] == 'enter') != (trigger == 'immediate')):
        raise ValueError('Final stance and trigger disagree')
    for key, choices in (('watch_minutes', (15, 30, 60, 120)), ('holding_minutes', (15, 60, 180, 360, 720, 1440))):
        if type(decision.get(key)) is not int or decision[key] not in choices:
            raise ValueError('Invalid final decision duration')
    multiplier = decision.get('risk_multiplier')
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or not 0 < multiplier <= 1:
        raise ValueError('Invalid final risk multiplier')
    if multiplier > analysis.get('regime', {}).get('risk_multiplier', 0):
        raise ValueError('Final decision cannot increase technical risk allowance')
    if any(analysis.get(role, {}).get('status') != 'observed' for role in ('sentiment', 'fundamentals')):
        if decision['stance'] != 'watch' or multiplier > .5:
            raise ValueError('Missing context requires conditional WATCH and at most half risk')
    return decision


def fresh_quote(quote, asof, max_age=30):
    if not isinstance(quote, dict):
        return False
    try:
        bid, ask = quote['bid'], quote['ask']
        return (type(bid) in (int, float) and type(ask) in (int, float)
            and math.isfinite(bid) and math.isfinite(ask) and 0 < bid < ask
            and 0 <= (timestamp(asof) - timestamp(quote['available_at'])).total_seconds() <= max_age)
    except (ValueError, KeyError, TypeError):
        return False


class Orchestrator:
    def __init__(self, directory, config=None, analyzer=None, scanner=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = dict(mode='paper', max_evaluations=12, fee_bps=None, slippage_bps=0.,
            max_position_fraction=.15, risk_per_trade_fraction=.005, minimum_net_rr=1.,
            max_spread_bps=100., max_quote_age_seconds=30., max_candle_age_seconds=1800.,
            cooldown_minutes=15., technical_models=True, model_device='cpu',
            max_analysis_age_seconds=1800., decision_contract='cross_specialist_v1',
            initial_cash={'TMN': 5_000_000., 'USDT': 0.})
        self.config.update(config or {})
        if self.config.get('review_mode','legacy') not in ('legacy','model_only','model_context'):
            raise ValueError('Unknown review mode')
        if self.config.get('entry_window_ends_at') is not None:timestamp(self.config['entry_window_ends_at'])
        _position_limit(self.config.get('max_open_positions'))
        if self.config.get('account_path') is not None:
            self.config['account_path'] = str(Path(self.config['account_path']).resolve())
        if self.config['fee_bps'] is None:
            fee_path = Path(self.config.get('official_fees_path', DEFAULT_FEES_PATH))
            self.config['fee_bps'] = official_public_rates(fee_path)
            evidence = json.loads(fee_path.read_text(encoding='utf-8'))
            self.config.update(fee_source='official_public_taker_level_1_scenario_account_unknown',
                fee_source_sha256=evidence['source']['sha256'], fee_account_verified=False,
                fee_source_url=evidence['source']['url'], fee_checked_utc=evidence['checked_utc'])
        else:
            self.config.update(fee_source='explicit_override_assumption_account_unknown', fee_account_verified=False)
        self.config['execution_cost_scope'] = 'official public fee scenario or explicit override; spread observed in bid/ask; additional market impact unknown'
        if self.config['mode'] not in ('paper', 'replay'):
            raise ValueError('Only paper or explicitly labelled historical replay is available')
        budget = self.config['max_evaluations']
        if type(budget) is not int or not 1 <= budget <= 100:
            raise ValueError('max_evaluations must be an integer in [1,100]')
        if not 0 <= float(self.config['cooldown_minutes']) <= 1440:
            raise ValueError('Invalid cooldown')
        age = self.config['max_analysis_age_seconds']
        if type(age) not in (int, float) or not math.isfinite(age) or not 0 < age <= 7200:
            raise ValueError('Analysis validity must be (0,7200] seconds')
        self.exclusions = load_exclusions(self.config.get('exclusions_path', DEFAULT_EXCLUSIONS))
        self.analyzer = analyzer or specialists.analyze
        if scanner is None and self.config['technical_models']:
            from .ai_market import AIMarket
            scanner = AIMarket(self.config).evaluate
        self.scanner = scanner or fast_market.evaluate
        self.path = self.directory / 'state.json'
        with exclusive(self.directory / 'orchestrator.lock'):
            self.state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else dict(
                config=self.config, evaluations=0, seen=[], plans={}, cooldowns={}, last_asof={})
            if self.state['config'] != self.config:
                raise ValueError('Existing orchestrator configuration differs; use a new directory')
            self.broker = PaperBroker(self.config.get('account_path') or self.directory / 'account.json',
                self.config['initial_cash'], self.config['fee_bps'])
            self._save()

    def _save(self):
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(self.state, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def _log(self, kind, symbol, asof, message, **fields):
        row = dict(event=kind, symbol=symbol, asof=asof, mode=self.config['mode'],
            recorded_at=datetime.now(timezone.utc).isoformat(), message=message, **fields)
        with (self.directory / 'decisions.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        return row

    def _entry_window_open(self,asof):
        end=self.config.get('entry_window_ends_at')
        current=datetime.now(timezone.utc) if self.config['mode']=='paper' else timestamp(asof)
        return end is None or current<timestamp(end)

    def process(self, frame, symbol, asof, quote=None, context=None, usdt_frame=None, *, allow_entries=True):
        """One tick. Caller may poll every few seconds; duplicate setups never re-infer."""
        if type(allow_entries) is not bool:
            raise ValueError('allow_entries must be boolean')
        # ponytail: synchronous inference holds this coordinator's lock; use a reserved-job worker before live concurrent execution.
        with exclusive(self.directory / 'orchestrator.lock'):
            # Reload under the single-writer lock, so two processes cannot use stale budgets.
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
            return self._process(frame, canonical_symbol(symbol), timestamp(asof).isoformat(), quote, context or {}, usdt_frame, allow_entries)

    def _process(self, frame, symbol, asof, quote, context, usdt_frame, allow_entries=True):
        now = timestamp(asof)
        if self.config['mode'] == 'paper' and abs((datetime.now(timezone.utc) - now).total_seconds()) > 30:
            return self._log('blocked', symbol, asof, 'زمان ورودی با ساعت جاری سازگار نیست؛ سفارش کاغذی متوقف شد.')
        previous = self.state['last_asof'].get(symbol)
        if previous and now < timestamp(previous):
            raise ValueError('Snapshot clock moved backwards')
        self.state['last_asof'][symbol] = asof
        valid_book = fresh_quote(quote, asof, self.config['max_quote_age_seconds'])
        account = self.broker.account
        position = account['positions'].get(symbol)
        if position:
            # A crash may occur after the broker commits a buy but before plan removal.
            self.state['plans'].pop(symbol, None)
            if valid_book:
                reason = ('stop' if quote['bid'] <= position['stop_price'] else
                    'target' if quote['bid'] >= position['target_price'] else
                    'expiry' if position.get('exit_mode') not in ('barriers_only','price_barriers') and now >= timestamp(position['expires_at']) else None)
                if reason:
                    try:
                        fill = self.broker.submit('exit-' + position['opened_at'] + '-' + symbol, symbol,
                            'sell', position['quantity'], quote, asof, exclusions=self.exclusions)
                    except ValueError as error:
                        self._save()
                        return self._log('exit_deferred', symbol, asof, 'خروج لازم است ولی اجرای کاغذی تأیید نشد.', reason=reason, error=str(error))
                    self.state['cooldowns'][symbol] = (now + pd.Timedelta(minutes=self.config['cooldown_minutes'])).isoformat()
                    self._save()
                    return self._log('exit', symbol, asof, 'موقعیت اسپات کاغذی بسته شد.', reason=reason, fill=fill)
            self._save()
            return self._log('position_monitored' if valid_book else 'exit_deferred', symbol, asof,
                'پایش موقعیت ادامه دارد.' if valid_book else 'اردربوک تازه در دسترس نیست؛ خروج قابل شبیه‌سازی نیست.')
        if not allow_entries or not self._entry_window_open(asof):
            self.state['plans'].pop(symbol, None)
            self._save()
            return self._log('entries_paused', symbol, asof, 'ورود تازه متوقف است؛ طرح ورود لغو شد و پایش خروج ادامه دارد.')
        if not market_allowed(symbol, self.exclusions):
            self.state['plans'].pop(symbol, None)
            self._save()
            return self._log('market_blocked', symbol, asof, 'این بازار در فهرست ممنوع حساب اعتباری است.')
        plan = self.state['plans'].get(symbol)
        if plan:
            if now >= timestamp(plan['expires_at']):
                del self.state['plans'][symbol]
                self._save()
                return self._log('plan_expired', symbol, asof, 'طرح ورود بدون تأیید منقضی شد.', plan_id=plan['id'])
            context_key = specialists.context_fingerprint({} if self.config.get('review_mode')=='model_only' else context, symbol, asof)
            age = (now - timestamp(plan['reviewed_at'])).total_seconds()
            if context_key != plan['context_fingerprint'] or age >= self.config['max_analysis_age_seconds']:
                del self.state['plans'][symbol]
                self._save()
                return self._log('plan_invalidated', symbol, asof,
                    'اطلاعات زمینه تغییر کرد یا تحلیل قدیمی شد؛ پیش از ورود تحلیل تازه لازم است.',
                    reason='context_changed' if context_key != plan['context_fingerprint'] else 'analysis_stale', plan_id=plan['id'])
            if not valid_book or timestamp(quote['available_at']) <= timestamp(plan['available_after']):
                self._save()
                return self._log('waiting_quote', symbol, asof, 'منتظر اردربوک تازه پس از پایان تحلیل هستیم.')
            if plan['trigger'] != 'immediate':
                # No model call: only later, completed bars can confirm a frozen WATCH plan.
                bars = frame.copy()
                dates = pd.to_datetime(bars.timestamp, utc=True)
                interval = plan['interval_seconds']
                bars = bars.loc[dates + pd.Timedelta(seconds=interval) <= pd.Timestamp(now)]
                if not len(bars):
                    self._save()
                    return self._log('watch', symbol, asof, 'کندل کامل تازه‌ای برای تأیید نداریم.')
                bar = bars.iloc[-1]
                try:
                    _, clocks, gaps, _ = _validated_frame(bars)
                except ValueError as error:
                    self._save()
                    return self._log('watch', symbol, asof, 'داده کندل برای تأیید معتبر نیست.', error=str(error))
                if any(gaps & (clocks >= pd.Timestamp(plan['candle_close']))):
                    del self.state['plans'][symbol]
                    self._save()
                    return self._log('plan_cancelled_gap', symbol, asof, 'شکاف داده پس از تحلیل، طرح ورود را لغو کرد.')
                later = pd.Timestamp(bar.timestamp) + pd.Timedelta(seconds=interval) > pd.Timestamp(plan['candle_close'])
                level = plan['setup']['confirmation_price'] if plan['trigger'] == 'breakout_with_volume' else plan['setup']['reclaim_price']
                volume = plan['setup'].get('minimum_volume')
                valid_bar = all(math.isfinite(float(bar[k])) for k in ('close', 'volume'))
                recent = (now - (pd.Timestamp(bar.timestamp) + pd.Timedelta(seconds=interval))).total_seconds() <= self.config['max_candle_age_seconds']
                quote_after_close = timestamp(quote['available_at']) >= pd.Timestamp(bar.timestamp) + pd.Timedelta(seconds=interval)
                checks=dict(later_bar=bool(later),recent_bar=bool(recent),quote_after_close=bool(quote_after_close),
                    finite_bar=bool(valid_bar),price_above_level=bool(valid_bar and bar.close>level),
                    volume_positive=bool(valid_bar and bar.volume>0),
                    volume_at_least_minimum=bool(valid_bar and volume is not None and bar.volume>=volume))
                if not all(checks.values()):
                    self._save()
                    return self._log('watch', symbol, asof, 'شرط قیمت و حجم طرح هنوز تأیید نشده است.',
                        watch_conditions=dict(price_level=float(level),current_close=float(bar.close),
                            minimum_volume=float(volume) if volume is not None else None,current_volume=float(bar.volume),
                            candle_closed_at=(pd.Timestamp(bar.timestamp)+pd.Timedelta(seconds=interval)).isoformat(),checks=checks))
            risk = size_order(symbol, plan['setup'], plan['analysis'], self.broker.account, quote,
                {**self.config, 'mode': 'spot', 'asof': asof, 'fee_bps': self.config['fee_bps'][symbol.split('/')[1]]
                 if isinstance(self.config['fee_bps'], dict) else self.config['fee_bps']}, self.exclusions)
            if not risk['approved']:
                del self.state['plans'][symbol]
                self._save()
                return self._log('risk_rejected', symbol, asof, 'کنترل ریسک اجازه ورود نداد.', risk=risk)
            if plan['setup'].get('exit_mode')=='barriers_only':
                protection={key:deepcopy(plan['setup'][key]) for key in ('exit_mode','expires_at','barrier_stop_fraction','barrier_target_fraction','original_entry_signal')}
            else:
                protection = dict(stop_fraction=plan['setup']['stop_fraction'], target_fraction=plan['setup']['target_fraction'],
                    expires_at=min(now + pd.Timedelta(minutes=plan['holding_minutes']),timestamp(plan['forecast_deadline'])).isoformat())
                for level in ('stop_price', 'target_price'):
                    if level in plan['setup']: protection[level]=plan['setup'][level]
            if self.config.get('max_open_positions') is not None:
                protection['max_open_positions'] = self.config['max_open_positions']
            if self.config.get('entry_window_ends_at') is not None:
                protection.update(entry_window_ends_at=self.config['entry_window_ends_at'],entry_window_clock=self.config['mode'])
            if not self._entry_window_open(asof):
                del self.state['plans'][symbol];self._save()
                return self._log('entries_paused',symbol,asof,'پنجره ورود پایان یافت؛ خرید تازه ثبت نشد.',reason='entry_window_ended')
            try:
                fill = self.broker.submit(plan['id'], symbol, 'buy', risk['quantity'], quote, asof,
                    plan=protection, exclusions=self.exclusions)
            except ValueError as error:
                self._save()
                return self._log('entry_deferred', symbol, asof, 'اجرای کاغذی ورودی تأیید نشد.', error=str(error), risk=risk)
            del self.state['plans'][symbol]
            self._save()
            return self._log('entry', symbol, asof, 'شرط ورود و کنترل ریسک تأیید شد؛ خرید اسپات کاغذی ثبت شد.', risk=risk, fill=fill)
        cooldown = self.state['cooldowns'].get(symbol)
        if cooldown and now < timestamp(cooldown):
            self._save()
            return self._log('cooldown', symbol, asof, 'فاصله پس از خروج هنوز تمام نشده است.')
        began = time.perf_counter()
        fees = self.config['fee_bps']
        scan_config = {**self.config, 'fee_bps': fees[symbol.split('/')[1]] if isinstance(fees, dict) else fees}
        fast = self.scanner(frame, symbol, asof, quote, scan_config)
        elapsed_ms = (time.perf_counter() - began) * 1000
        if fast['status'] != 'setup':
            self._save()
            return self._log('scan', symbol, asof, 'اسکن بازار موقعیت قابل بررسی پیدا نکرد.', fast_status=fast['status'], reason=fast['reason'],
                fast_ms=elapsed_ms, model_forecasts=fast.get('model_forecasts'), numeric_summary=fast.get('summary'))
        context_key = specialists.context_fingerprint({} if self.config.get('review_mode')=='model_only' else context, symbol, asof)
        key = hashlib.sha256((fast['fingerprint'] + context_key).encode()).hexdigest()
        if key in self.state['seen']:
            self._save()
            return self._log('duplicate_setup', symbol, asof, 'این موقعیت قبلاً تحلیل شده است؛ تماس LLM تکرار نشد.')
        if self.state['evaluations'] >= self.config['max_evaluations']:
            self._save()
            return self._log('budget_exhausted', symbol, asof, 'بودجه تحلیل تازه تمام شده؛ پایش و خروج همچنان فعال است.')
        summary = deepcopy(fast['summary'])
        summary.update({key: float(fast['setup'][key]) for key in (
            'stop_fraction', 'target_fraction', 'roundtrip_cost_fraction', 'reward_risk_after_cost') if key in fast['setup']})
        currency = symbol.split('/')[1]
        summary.update(account_cash_quote=float(account['cash'].get(currency, 0)),
            account_position_count=len(account['positions']), max_position_fraction=float(self.config['max_position_fraction']),
            risk_per_trade_fraction=float(self.config['risk_per_trade_fraction']))
        if usdt_frame is not None:
            tether = usdt_frame.copy()
            stamps = pd.to_datetime(tether.timestamp, utc=True)
            tether = tether.loc[stamps + pd.Timedelta(minutes=15) <= pd.Timestamp(now)]
            if len(tether) >= 13 and (now - (pd.Timestamp(tether.timestamp.iloc[-1]) + pd.Timedelta(minutes=15))).total_seconds() <= 1800:
                summary.update(iran_usdt_tmn=float(tether.close.iloc[-1]), iran_usdt_return_12=float(tether.close.iloc[-1] / tether.close.iloc[-13] - 1))
        self.state['seen'].append(key)
        self.state['evaluations'] += 1
        self._save()  # Reserve before inference; a crash cannot spend the same setup twice.
        inference_began = time.perf_counter()
        model_only=self.config.get('review_mode')=='model_only'
        if model_only:
            net=summary.get('model_net_return',summary.get('model_median_net_return'))
            if type(net) not in (int,float) or not math.isfinite(net) or net<=0 or not fast.get('model_forecasts'):
                return self._log('decision_blocked',symbol,asof,'سیگنال عددی معتبر و مثبت مدل برای ورود لازم است.',reason='invalid_numeric_model_signal')
            analysis=dict(status='ok',review_mode='model_only',regime=dict(status='numeric_model',stance='enter',risk_multiplier=1.),
                decision=dict(status='computed',stance='enter',trigger='immediate',risk_multiplier=1.,watch_minutes=15,holding_minutes=1440,
                    rationale='Explicit model-only experiment: positive trained numeric signal; no LLM or context veto claimed.'),
                sentiment=dict(status='not_requested',event_risk='unknown'),fundamentals=dict(status='not_requested',event_risk='unknown'),
                receipts=[],errors=[],numeric_model_approval=dict(expected_net_return=net,fingerprint=fast['fingerprint'],
                    source='scanner trained-model output; fresh original-signal risk check remains mandatory'))
        else:
            try:
                analysis = self.analyzer(summary, context, symbol, asof, self.directory / 'analysis' / key)
            except Exception as error:
                return self._log('analysis_failed', symbol, asof, 'تحلیل تخصصی شکست خورد؛ سفارش ایجاد نشد.', error=f'{type(error).__name__}: {error}')
        completed = datetime.now(timezone.utc).isoformat()
        if analysis['status'] != 'ok':
            return self._log('analysis_failed', symbol, asof, 'خروجی معتبر تحلیل تخصصی نداریم؛ سفارش ایجاد نشد.', analysis=analysis)
        try:
            decision = analysis['decision'] if model_only else entry_decision(analysis)
        except (ValueError, TypeError, KeyError) as error:
            return self._log('decision_blocked', symbol, asof, 'تصمیم نهایی اجازه ساخت طرح ورود نداد.', error=str(error), analysis=analysis)
        if decision['stance'] != 'reject':
            if not self._entry_window_open(asof):
                return self._log('entries_paused',symbol,asof,'تحلیل پس از پایان پنجره ورود تمام شد؛ طرح ورود ساخته نشد.',analysis=analysis,reason='entry_window_ended')
            available = asof if self.config['mode'] == 'replay' else completed
            horizon_minutes = fast['summary'].get('model_horizon_minutes', 360)
            deadline = timestamp(fast['setup'].get('timestamp',fast['clock']['candle_close'])) + pd.Timedelta(minutes=horizon_minutes)
            if fast['setup'].get('expires_at') is not None:
                deadline=min(deadline,timestamp(fast['setup']['expires_at']))
            if timestamp(available) >= deadline:
                return self._log('decision_expired', symbol, asof, 'افق پیش‌بینی پیش از پایان تحلیل تمام شد؛ طرح ورود ساخته نشد.', analysis=analysis)
            self.state['plans'][symbol] = dict(id='entry-' + key, setup=fast['setup'], analysis=analysis,
                trigger=decision['trigger'], available_after=available, reviewed_at=asof, context_fingerprint=context_key,
                expires_at=min(timestamp(available) + pd.Timedelta(minutes=decision['watch_minutes']), deadline).isoformat(),
                forecast_deadline=deadline.isoformat(),
                holding_minutes=horizon_minutes if fast['setup'].get('holding_policy')=='validated_horizon' else min(decision['holding_minutes'], horizon_minutes),
                requested_holding_minutes=decision['holding_minutes'],candle_close=fast['clock']['candle_close'],
                interval_seconds=fast['clock']['interval_seconds'])
            self._save()
        return self._log('analysis', symbol, asof, 'بررسی عددی مدل ثبت شد؛ LLM و تأیید زمینه در این آزمایش استفاده نشد.' if model_only else 'تحلیل تخصصی ثبت شد؛ هر ورود به داده تازه و تأیید ریسک نیاز دارد.',
            analysis=analysis, setup=fast['setup'], fast_ms=elapsed_ms, model_forecasts=fast.get('model_forecasts'),
            analysis_seconds=time.perf_counter() - inference_began, inference_completed_at=completed,
            latency_assumption='historical replay does not model inference latency' if self.config['mode'] == 'replay' else 'requires post-inference quote')
