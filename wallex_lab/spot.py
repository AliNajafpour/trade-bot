"""Offline spot risk checks and a cash-funded persistent paper ledger.

Execution follows engine.py cash/fee admission principles, using fresh observed
bid/ask proxies instead of candle prices. No exchange client or order API exists.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .locking import exclusive
from .fees import official_public_rates


def _number(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{name} is outside its allowed range')
    return value


def _fees(value):
    if isinstance(value, dict):
        if set(value) != {'TMN', 'USDT'}:
            raise ValueError('fee map must specify exactly TMN and USDT')
        return {quote: _number(value[quote], quote + ' fee_bps', 0, 9999) for quote in ('TMN', 'USDT')}
    return _number(value, 'fee_bps', 0, 9999)


def _position_limit(value):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError('max_open_positions must be a positive integer or None')
    return value


def _time(value):
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('timestamp needs a timezone')
        return stamp.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('a timezone-aware timestamp is required') from exc


def canonical_symbol(value):
    """Canonical exact spot pair; never use substring exclusion matching."""
    if not isinstance(value, str) or not value.isascii():
        raise ValueError('symbol must be ASCII text')
    value = value.upper()
    match = re.fullmatch(r'([A-Z0-9]{1,20})/(TMN|USDT)', value)
    if match is None:
        match = re.fullmatch(r'([A-Z0-9]{1,20})(TMN|USDT)', value)
    if match is None or re.search(r'(?:[2-9][LS]|BULL|BEAR|MARGIN|LEVERAGE)$', match[1]):
        raise ValueError('invalid or leveraged market symbol')
    return f'{match[1]}/{match[2]}'


def load_exclusions(path):
    return {canonical_symbol(line.strip()) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.lstrip().startswith('#')}


def market_allowed(symbol, exclusions):
    return canonical_symbol(symbol) not in {canonical_symbol(item) for item in exclusions}


def _spot_only(value):
    if any(key in value for key in ('leverage', 'margin', 'borrow', 'short', 'shorts', 'margin_mode', 'credit')):
        raise ValueError('leverage, margin, credit and short options are forbidden')
    if value.get('mode', 'spot') != 'spot':
        raise ValueError('only spot mode is allowed')


def _quote(quote, asof=None, allow_locked=False):
    bid = _number(quote.get('bid'), 'bid', minimum=1e-300)
    ask = _number(quote.get('ask'), 'ask', minimum=1e-300)
    if bid > ask or bid == ask and not allow_locked:
        raise ValueError('quote must have bid < ask')
    available = _time(quote.get('available_at'))
    if asof is not None and not 0 <= (_time(asof) - available).total_seconds() <= 30:
        raise ValueError(f'quote is future-dated or older than 30 seconds (age={(_time(asof)-available).total_seconds():.3f}s)')
    for side in ('ask_quantity', 'bid_quantity'):
        if side in quote:
            _number(quote[side], side)
    return bid, ask, available


def _equity(account, currency):
    cash = account.get('cash', {})
    total = _number(cash.get(currency, 0), f'{currency} cash')
    for symbol, position in account.get('positions', {}).items():
        if canonical_symbol(symbol).split('/')[1] == currency:
            quantity = _number(position.get('quantity'), 'position quantity')
            price = _number(position.get('mark_price', position.get('entry_price')), 'position mark', minimum=1e-300)
            total += quantity * price
    if not math.isfinite(total):
        raise ValueError('quote equity is nonfinite')
    return total


def protection_geometry(plan, entry_price):
    """Keep forecast price barriers fixed when the eventual entry quote changes."""
    if plan.get('exit_mode')=='barriers_only':
        stop=_number(plan.get('barrier_stop_fraction'),'barrier_stop_fraction',1e-12,1-1e-12)
        target=_number(plan.get('barrier_target_fraction'),'barrier_target_fraction',1e-12,10)
        return stop,target,entry_price*(1-stop),entry_price*(1+target)
    if 'stop_price' in plan or 'target_price' in plan:
        stop_price=_number(plan.get('stop_price'), 'stop_price', minimum=1e-300)
        target_price=_number(plan.get('target_price'), 'target_price', minimum=1e-300)
        if not stop_price < entry_price < target_price:
            raise ValueError('forecast barriers no longer bracket the entry price')
        stop, target=1-stop_price/entry_price, target_price/entry_price-1
    else:
        stop = _number(plan.get('stop_fraction'), 'stop_fraction', 1e-12, 1)
        target = _number(plan.get('target_fraction'), 'target_fraction', 1e-12, 10)
        stop_price, target_price=entry_price*(1-stop), entry_price*(1+target)
    return stop, target, stop_price, target_price


def entry_signal_net_return(signal,ask,fee_bps,slippage_bps=0):
    """Reprice an original signal; never transfer its probabilities to new barriers."""
    if not isinstance(signal,dict):raise ValueError('Original entry signal is required')
    _spot_only(signal)
    if signal.get('exit_mode')=='barriers_only' or 'barrier_stop_fraction' in signal:raise ValueError('Original signal must retain its original forecast geometry')
    reference=_number(signal.get('reference_price'),'original signal reference price',1e-300)
    fee=_number(fee_bps,'signal fee_bps',0,9999)/10000
    slip=_number(slippage_bps,'signal slippage_bps',0,9999)/10000
    if 'first_touch_probabilities' in signal:
        probabilities=signal['first_touch_probabilities']
        if not isinstance(probabilities,list) or len(probabilities)!=3:raise ValueError('Three original first-touch probabilities required')
        probabilities=[_number(p,'original first-touch probability',0,1) for p in probabilities]
        if not math.isclose(sum(probabilities),1.,rel_tol=1e-6):raise ValueError('Original probabilities must sum to one')
        _,_,stop,target=protection_geometry(signal,ask)
        entry=ask*(1+slip)*(1+fee)
        reward=target*(1-slip)*(1-fee)/entry-1;loss=stop*(1-slip)*(1-fee)/entry-1
        if reward<=0:raise ValueError('Original target no longer clears costs')
        if reward/-loss<_number(signal.get('minimum_net_rr',0),'original minimum net RR',0):raise ValueError('Original signal RR no longer qualifies')
        if probabilities[0]<_number(signal.get('probability_threshold',0),'original probability threshold',0,1):raise ValueError('Original signal probability below threshold')
        timeout=signal.get('train_timeout_return')
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=-1:raise ValueError('Finite original timeout return above -1 required')
        baseline=_number(signal.get('fee_bps',fee_bps),'original timeout fee_bps',0,9999)/10000
        timeout=(1+timeout)*(1+baseline)/(1-baseline)*reference/ask*(1-fee)/(1+fee)*(1-slip)/(1+slip)-1
        return probabilities[0]*reward+probabilities[1]*loss+probabilities[2]*timeout
    net=signal.get('expected_net_return')
    if type(net) not in (int,float) or not math.isfinite(net) or net<=0:raise ValueError('Positive original model net return required')
    baseline=_number(signal.get('fee_bps'),'original fee_bps',0,9999)/10000
    return (1+net)*(1+baseline)/(1-baseline)*reference/ask*(1-fee)/(1+fee)*(1-slip)/(1+slip)-1


def size_order(symbol, setup, analysis, account, quote, config, exclusions):
    """Fail-closed cash-only sizing; payoff geometry is not a profit forecast."""
    result = dict(approved=False, reasons=[], quantity=0., notional=0., risk_budget=0., estimated_loss_at_stop=0.)
    try:
        symbol = canonical_symbol(symbol)
        currency = symbol.split('/')[1]
        _spot_only(setup)
        _spot_only(config)
        limit = _position_limit(config.get('max_open_positions'))
        if limit is not None and len(account.get('positions', {})) >= limit:
            raise ValueError('max_open_positions reached')
        if not market_allowed(symbol, exclusions):
            raise ValueError('market is excluded by policy')
        if any(canonical_symbol(k) == symbol and _number(v.get('quantity'), 'position quantity') > 0
               for k, v in account.get('positions', {}).items()):
            raise ValueError('existing position cannot be duplicated')
        bid, ask, stamp = _quote(quote, config.get('asof'))
        stop, target, stop_price, target_price = protection_geometry(setup, ask)
        if bid <= stop_price:
            raise ValueError('current bid has already breached the stop price')
        fees = _fees(config['fee_bps']) if config.get('fee_bps') is not None else official_public_rates()
        fee = (fees[currency] if isinstance(fees, dict) else fees) / 10000
        slip = _number(config.get('slippage_bps', 0), 'slippage_bps', 0, 9999) / 10000
        barrier_only=setup.get('exit_mode')=='barriers_only'
        if barrier_only:
            if setup.get('expires_at') is not None:raise ValueError('Barrier-only setup requires expires_at=None')
            signal=setup.get('original_entry_signal')
            original_net=entry_signal_net_return(signal,ask,fee*10000,slip*10000)
            if signal.get('stop_price') is not None and bid<=signal['stop_price']:raise ValueError('Original signal stop already breached')
            result['original_signal_expected_net_return']=original_net
            if original_net<=0:raise ValueError('Original model entry signal is no longer positive at fresh ask')
        model_only=config.get('review_mode')=='model_only'
        if model_only and analysis.get('review_mode')!='model_only':raise ValueError('Explicit numeric model review required')
        council=config.get('review_mode')=='council'
        if council and analysis.get('review_mode')!='council':raise ValueError('Explicit council review required')
        position_fraction = _number(config.get('max_position_fraction', .15), 'max_position_fraction', 0, 1)
        risk_fraction = _number(config.get('risk_per_trade_fraction', .005), 'risk_per_trade_fraction', 0, 1)
        regime = analysis.get('regime', {})
        sentiment = analysis.get('sentiment', {})
        multiplier = _number(regime.get('risk_multiplier', None if council else .5), 'risk_multiplier', 0, 1)
        decision = analysis.get('decision')
        if council and (not isinstance(decision,dict) or decision.get('stance')!='enter'):
            raise ValueError('Observed council entry decision required')
        if model_only and (not isinstance(decision,dict) or decision.get('stance')!='enter' or decision.get('trigger')!='immediate'):
            raise ValueError('Computed model-only immediate entry decision required')
        if decision is not None:
            if decision.get('status') != ('computed' if model_only else 'observed') or decision.get('stance') not in ('enter', 'watch'):
                raise ValueError('final decision does not permit entry')
            multiplier = min(multiplier, _number(decision.get('risk_multiplier'), 'final risk multiplier', 0, 1))
        if council:
            cutoff = _time(config.get('asof'))
            require_reports = config.get('require_reports', True)
            if type(require_reports) is not bool:
                raise ValueError('require_reports must be a boolean')
            for kind in ('market', 'macro'):
                report = analysis.get('reports', {}).get(kind)
                if report is None and not require_reports:
                    multiplier = min(multiplier, _number(config.get('unknown_news_risk_multiplier', .5), 'unknown news multiplier', 0, 1))
                    continue
                if not isinstance(report, dict) or report.get('status') not in ('ok', 'partial'):
                    raise ValueError('Usable council ' + kind + ' report required')
                if not _time(report.get('generated_at')) <= cutoff < _time(report.get('valid_until')):
                    raise ValueError('Council ' + kind + ' report is future-dated or expired')
                if report.get('invalidated_at') is not None and _time(report['invalidated_at']) <= cutoff:
                    raise ValueError('Council ' + kind + ' report was invalidated')
                assessment = report.get('assessment', {})
                _number(assessment.get('confidence'), kind + ' report confidence', 0, 1)
                multiplier = min(multiplier, _number(assessment.get('risk_multiplier'), kind + ' report risk multiplier', 0, 1))
        else:
            fundamentals = analysis.get('fundamentals', {})
            if fundamentals.get('event_risk') == 'high':
                raise ValueError('high fundamental event risk blocks new buys')
            if fundamentals.get('status') != 'observed' and not model_only:
                multiplier = min(multiplier, .5)
            event_risk = sentiment.get('event_risk', 'unknown')
            if event_risk not in ('low', 'medium', 'elevated', 'high', 'unknown'):
                raise ValueError('unrecognized event risk')
            if event_risk == 'high':
                raise ValueError('high event risk blocks new buys')
            confidence = sentiment.get('confidence')
            if confidence is not None:
                confidence = _number(confidence, 'sentiment confidence', 0, 1)
            if not model_only and (event_risk == 'unknown' or confidence is None or confidence < .5):
                multiplier = min(multiplier, _number(config.get('unknown_news_risk_multiplier', .5), 'unknown news multiplier', 0, 1))
        spread_bps = (ask - bid) / ask * 10000
        if spread_bps > _number(config.get('max_spread_bps', 100), 'max_spread_bps'):
            raise ValueError('spread exceeds limit')
        downside = ask * (stop + 2 * (fee + slip))
        entry_cost=ask*(1+slip)*(1+fee)
        if barrier_only:downside=entry_cost-stop_price*(1-slip)*(1-fee)
        target_net=target_price*(1-slip)*(1-fee)/entry_cost-1
        stop_net=stop_price*(1-slip)*(1-fee)/entry_cost-1
        if target_net<=0:raise ValueError('target payoff does not clear execution costs')
        net_rr = target_net/(-stop_net)
        if net_rr < _number(config.get('minimum_net_rr', 1), 'minimum_net_rr', 0):
            raise ValueError('cost-adjusted payoff geometry is below minimum')
        if 'first_touch_probabilities' in setup and not barrier_only:
            probabilities=setup['first_touch_probabilities']
            if not isinstance(probabilities,list) or len(probabilities)!=3:raise ValueError('Three first-touch probabilities required')
            probabilities=[_number(p,'first-touch probability',0,1) for p in probabilities]
            if not math.isclose(sum(probabilities),1.,rel_tol=1e-6):raise ValueError('First-touch probabilities must sum to one')
            timeout=setup.get('train_timeout_return')
            if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=-1:raise ValueError('Finite TRAIN timeout return above -1 required')
            reference=_number(setup.get('reference_price'),'first-touch scoring reference price',1e-300)
            # Timeout is net at the scoring ask; reprice its exit value at the actual entry ask.
            fresh_timeout=(1+timeout)*reference/ask*(1-slip)/(1+slip)-1
            expected=probabilities[0]*target_net+probabilities[1]*stop_net+probabilities[2]*fresh_timeout
            result['fresh_timeout_net_return']=fresh_timeout
            result['fresh_expected_net_return']=expected
            if expected<=0:raise ValueError('learned expected return is no longer positive at the fresh entry quote')
        equity = _equity(account, currency)
        ledger = account.get('risk', {}).get(currency, {})
        for key, reference in (('max_drawdown_stop', ledger.get('peak_equity')), ('daily_loss_stop', ledger.get('day_start_equity') if ledger.get('day') == stamp.date().isoformat() else None)):
            if config.get(key) is not None:
                limit = _number(config[key], key, 1e-12, 1)
                if reference is None:
                    raise ValueError(f'{key} requires persisted quote-specific risk baseline')
                if equity <= _number(reference, key + ' reference', 1e-300) * (1 - limit):
                    raise ValueError(f'{key} breached')
        cash = _number(account.get('cash', {}).get(currency, 0), 'cash')
        budget = equity * risk_fraction * multiplier
        quantity = min(budget / downside, cash / entry_cost, equity * position_fraction / ask)
        if 'ask_quantity' in quote:
            quantity = min(quantity, quote['ask_quantity'])
        notional = quantity * ask
        minima = config.get('min_notional', 50000 if currency == 'TMN' else 5)
        if isinstance(minima, dict):
            minima = minima.get(currency, 50000 if currency == 'TMN' else 5)
        if quantity <= 0 or notional < _number(minima, 'min_notional'):
            raise ValueError('cash, depth or minimum notional prevents a buy')
        result.update(approved=True, symbol=symbol, quote_currency=currency, quantity=quantity, notional=notional,
                      risk_budget=budget, estimated_loss_at_stop=quantity * downside, net_rr=net_rr,
                      equity=equity, risk_multiplier=multiplier,
                      assumptions=['Same-quote equity only; absent marks use entry price',
                                   'Unrounded fractional paper quantity; geometry is not expected profit'])
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        result['reasons'].append(str(exc))
    return result


class PaperBroker:
    """Single-writer offline ledger. Kernel locks and atomic replace protect saves."""
    def __init__(self, path, initial_cash=None, fee_bps=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + '.lock')
        fee = _fees(fee_bps) if fee_bps is not None else None
        with exclusive(self.lock_path,timeout=2):
            if self.path.exists():
                state = self._read()
                if fee is not None and state['fee_bps'] != fee:
                    raise ValueError('Explicit fee_bps differs from persisted paper ledger')
            else:
                if fee is None:
                    fee = official_public_rates()
                cash = {'TMN': 5_000_000., 'USDT': 0.} if initial_cash is None else dict(initial_cash)
                if set(cash) - {'TMN', 'USDT'}:
                    raise ValueError('only TMN and USDT quote balances are supported')
                cash = {k: _number(cash.get(k, 0), k + ' cash') for k in ('TMN', 'USDT')}
                risk = {k: dict(day=None, day_start_equity=v, peak_equity=v) for k, v in cash.items()}
                self._save(dict(version=1, fee_bps=fee, cash=cash, positions={}, orders={}, fills=[], risk=risk))

    def _read(self):
        state = json.loads(self.path.read_text(encoding='utf-8'))
        if state.get('version') != 1 or not all(k in state for k in ('cash', 'positions', 'orders', 'fills', 'risk')):
            raise ValueError('invalid paper ledger schema; refusing to reset')
        _fees(state.get('fee_bps'))
        for currency in ('TMN', 'USDT'):
            _equity(state, currency)
        return state

    def _save(self, state):
        data = json.dumps(state, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        descriptor, name = tempfile.mkstemp(prefix=self.path.name + '.', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @property
    def account(self):
        with exclusive(self.lock_path,timeout=2):
            return self._read()  # json.loads already returns an independent snapshot.

    @property
    def state(self):
        return self.account

    def amend_protection(self, amendment_id, symbol, stop_price, target_price, quote, asof, max_risk_fraction, *, portfolio_quotes=None):
        """Atomically replace held spot barriers without creating a fill or expiry."""
        if not isinstance(amendment_id, str) or not amendment_id or len(amendment_id) > 200:
            raise ValueError('nonempty amendment_id <=200 characters required')
        symbol = canonical_symbol(symbol)
        stop_price = _number(stop_price, 'stop_price', 1e-300)
        target_price = _number(target_price, 'target_price', 1e-300)
        fraction = _number(max_risk_fraction, 'max_risk_fraction', 0, 1)
        portfolio_quotes = {} if portfolio_quotes is None else deepcopy(portfolio_quotes)
        if not isinstance(portfolio_quotes, dict):
            raise ValueError('portfolio_quotes must map held symbols to fresh quotes')
        now = _time(asof)
        request = dict(symbol=symbol, stop_price=stop_price, target_price=target_price,
                       quote=deepcopy(quote), asof=now.isoformat(), max_risk_fraction=fraction,
                       portfolio_quotes=portfolio_quotes)
        request = json.loads(json.dumps(request, allow_nan=False, default=lambda v: _time(v).isoformat()))
        digest = sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with exclusive(self.lock_path, timeout=2):
            state = self._read()
            amendments = state.setdefault('amendments', {})
            if amendment_id in amendments:
                if amendments[amendment_id]['request_sha256'] != digest:
                    raise ValueError('amendment_id already used with a different payload')
                return deepcopy(amendments[amendment_id]['receipt'])
            position = state['positions'].get(symbol)
            if position is None:
                raise ValueError('protection amendment requires owned spot inventory')
            _spot_only(position)
            quantity = _number(position.get('quantity'), 'position quantity', 1e-300)
            bid, ask, _ = _quote(quote, now, allow_locked=True)
            if not stop_price < bid <= ask < target_price:
                raise ValueError('protection must bracket current bid and ask')
            old_stop, old_target = position['stop_price'], position['target_price']
            position.update(stop_price=stop_price, target_price=target_price, mark_price=bid)
            currency = symbol.split('/')[1]
            rates = state['fee_bps']
            fee = (rates[currency] if isinstance(rates, dict) else rates) / 10000
            other_marks = {}
            for held_symbol, held in state['positions'].items():
                if held_symbol != symbol and canonical_symbol(held_symbol).split('/')[1] == currency:
                    observed_quote = portfolio_quotes.get(held_symbol)
                    if not isinstance(observed_quote, dict):
                        raise ValueError('Fresh portfolio quote required for ' + held_symbol)
                    held_bid, _, _ = _quote(observed_quote, now, allow_locked=True)
                    held['mark_price'] = other_marks[held_symbol] = held_bid
            equity = _equity(state, currency)
            risk_budget = equity * fraction
            losses = {}
            for held_symbol, held in state['positions'].items():
                if canonical_symbol(held_symbol).split('/')[1] != currency:
                    continue
                held_quantity = _number(held.get('quantity'), 'position quantity', 1e-300)
                basis = _number(held.get('cost_basis'), 'position cost basis')
                stop = _number(held.get('stop_price'), 'position stop_price', 1e-300)
                losses[held_symbol] = max(0., basis - held_quantity * stop * (1 - fee))
            estimated_loss = sum(losses.values())
            if not math.isfinite(estimated_loss) or estimated_loss > risk_budget:
                raise ValueError('amended account stop risk exceeds same-quote equity risk budget')
            receipt = dict(amendment_id=amendment_id, status='paper_protection_amended', symbol=symbol,
                timestamp=now.isoformat(), quote=deepcopy(request['quote']), quantity=quantity,
                old_stop_price=old_stop, old_target_price=old_target,
                new_stop_price=stop_price, new_target_price=target_price, expires_at=position.get('expires_at'),
                quote_currency=currency, equity=equity, risk_budget=risk_budget,
                estimated_loss_at_stop=estimated_loss, position_estimated_loss_at_stop=losses[symbol],
                max_risk_fraction=fraction, exit_fee_bps=fee * 10000, fresh_other_marks=other_marks,
                portfolio_quotes=deepcopy(request['portfolio_quotes']),
                assumption='Same-quote equity marks every held symbol at its fresh observed bid; '
                           'stop loss includes remaining entry basis and exit fee; stop execution price is not guaranteed')
            amendments[amendment_id] = dict(request_sha256=digest, request=request, receipt=receipt)
            self._save(state)
            return deepcopy(receipt)

    def submit(self, order_id, symbol, side, quantity, quote, asof, plan=None, exclusions=None):
        if not isinstance(order_id, str) or not order_id or len(order_id) > 200:
            raise ValueError('nonempty order_id <=200 characters required')
        symbol = canonical_symbol(symbol)
        if side not in ('buy', 'sell'):
            raise ValueError('only spot buy and sell are allowed')
        quantity = _number(quantity, 'quantity', 1e-300)
        now = _time(asof)
        plan = {} if plan is None else deepcopy(plan)
        _spot_only(plan)
        limit = _position_limit(plan.get('max_open_positions'))
        request = dict(symbol=symbol, side=side, quantity=quantity, quote=deepcopy(quote), asof=now.isoformat(), plan=plan)
        request = json.loads(json.dumps(request, allow_nan=False, default=lambda v: _time(v).isoformat()))
        with exclusive(self.lock_path,timeout=2):
            state = self._read()
            if order_id in state['orders']:
                if state['orders'][order_id]['request'] != request:
                    raise ValueError('order_id already used with a different payload')
                return deepcopy(state['orders'][order_id]['fill'])
            bid, ask, _ = _quote(quote, now)
            currency = symbol.split('/')[1]
            price = ask if side == 'buy' else bid
            depth_key = 'ask_quantity' if side == 'buy' else 'bid_quantity'
            if depth_key in quote and quantity > _number(quote[depth_key], depth_key):
                raise ValueError('requested quantity exceeds observed depth')
            notional = quantity * price
            rates = state['fee_bps']
            fee = notional * (rates[currency] if isinstance(rates, dict) else rates) / 10000
            if not math.isfinite(notional + fee):
                raise ValueError('nonfinite order cost')
            if symbol in state['positions']:
                state['positions'][symbol]['mark_price'] = bid
            equity_before = _equity(state, currency)
            if side == 'buy':
                if plan.get('entry_window_ends_at') is not None:
                    clock=plan.get('entry_window_clock','paper')
                    if clock not in ('paper','replay'):raise ValueError('Invalid entry-window clock')
                    observed_now=datetime.now(timezone.utc) if clock=='paper' else now
                    if observed_now>=_time(plan['entry_window_ends_at']):raise ValueError('Entry window ended before atomic buy admission')
                if limit is not None and len(state['positions']) >= limit:
                    raise ValueError('max_open_positions reached')
                if exclusions is None:
                    policy_path = Path(__file__).resolve().parents[1] / 'config' / 'credit-excluded-markets.txt'
                    if not policy_path.is_file():
                        raise ValueError('buy requires the exclusion policy file or explicit exclusions')
                    exclusions = load_exclusions(policy_path)
                if not market_allowed(symbol, exclusions):
                    raise ValueError('market is excluded by policy')
                if symbol in state['positions']:
                    raise ValueError('existing position cannot be duplicated')
                if notional + fee > state['cash'].get(currency, 0):
                    raise ValueError('insufficient same-quote cash including fee')
                stop, target, stop_price, target_price = protection_geometry(plan, price)
                barrier_only=plan.get('exit_mode')=='barriers_only'
                price_barriers=plan.get('exit_mode')=='price_barriers'
                if price_barriers:
                    if 'stop_price' not in plan or 'target_price' not in plan:
                        raise ValueError('Price-barrier position requires absolute stop_price and target_price')
                    if plan.get('expires_at') is not None:
                        raise ValueError('Price-barrier position requires expires_at=None')
                    if bid <= stop_price:
                        raise ValueError('current bid has already breached the stop price')
                    expires=None
                elif barrier_only:
                    if plan.get('expires_at') is not None:raise ValueError('Barrier-only position requires expires_at=None')
                    expires=None
                else:
                    expires = _time(plan.get('expires_at'))
                    if expires <= now:raise ValueError('position expiry must be in the future')
                state['cash'][currency] -= notional + fee
                state['positions'][symbol] = dict(quantity=quantity, entry_price=price, mark_price=bid,
                    stop_price=stop_price, target_price=target_price,
                    opened_at=now.isoformat(), expires_at=expires.isoformat() if expires else None, cost_basis=notional + fee)
                if barrier_only:
                    state['positions'][symbol].update(exit_mode='barriers_only',original_entry_signal=deepcopy(plan.get('original_entry_signal')))
                elif price_barriers:
                    state['positions'][symbol]['exit_mode']='price_barriers'
                    state['positions'][symbol]['original_entry_signal']=deepcopy(plan.get('original_entry_signal'))
                realized = 0.
            else:
                position = state['positions'].get(symbol)
                if position is None or quantity > position['quantity']:
                    raise ValueError('sell exceeds owned inventory')
                cost_basis = position['cost_basis'] * quantity / position['quantity']
                realized = notional - fee - cost_basis
                position['cost_basis'] -= cost_basis
                position['quantity'] -= quantity
                state['cash'][currency] += notional - fee
                if position['quantity'] == 0:
                    del state['positions'][symbol]
                else:
                    position['mark_price'] = bid
            ledger = state['risk'][currency]
            marked = _equity(state, currency)
            if ledger['day'] != now.date().isoformat():
                ledger['day'], ledger['day_start_equity'] = now.date().isoformat(), equity_before
            ledger['peak_equity'] = max(ledger['peak_equity'], equity_before, marked)
            fill = dict(order_id=order_id, status='paper_filled', symbol=symbol, side=side, quantity=quantity,
                        price=price, notional=notional, fee=fee, realized_pnl=realized, timestamp=now.isoformat(),
                        assumption='Observed bid/ask paper proxy; no network order, queue or guaranteed real fill; fractional quantity unrounded')
            state['fills'].append(fill)
            state['orders'][order_id] = dict(request=request, fill=fill)
            self._save(state)
            return deepcopy(fill)
