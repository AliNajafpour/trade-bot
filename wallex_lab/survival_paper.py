"""Isolated, CPU-only native survival paper account using an existing live feed."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import torch

from .barrier_dataset import sha256
from .barrier_survival import SurvivalPredictor
from .context import fetch_public
from .data import load_market, normalize_history
from .locking import atomic_replace, exclusive
from .spot import PaperBroker, _quote, _equity, load_exclusions, size_order


GRID = ((.005,.005),(.005,.01),(.01,.005),(.01,.01),(.01,.02),
        (.02,.01),(.02,.02),(.02,.04),(.04,.02))
POLICY = dict(p_tp_min=.50, p_no_touch_max=.25, minimum_net_rr=1.,
              max_position_fraction=.15, risk_per_trade_fraction=.005,
              max_account_stop_risk_fraction=.02, min_notional=1.,
              max_open_positions=4, max_spread_bps=100.)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    atomic_replace(temporary, path)


def now():
    return datetime.now(timezone.utc)


def payoff(prediction, forecast, ask, bid, fee_bps, policy):
    """Conditional-hit comparison only; no-touch has no assigned payoff."""
    fee = fee_bps / 10000
    # Price-touch targets refer to traded prices. Reserve the currently observed
    # spread at both exit barriers, then actual exits still require executable bids.
    spread = ask - bid
    cost = ask * (1 + fee)
    reward = (prediction['tp'] - spread) * (1 - fee) / cost - 1
    loss = 1 - max(0., prediction['sl'] - spread) * (1 - fee) / cost
    tp, sl, untouched = (forecast[k] for k in ('p_tp_first','p_sl_first','p_no_touch'))
    if not all(0 <= p <= 1 for p in (tp, sl, untouched)) or abs(tp+sl+untouched-1)>1e-5:
        raise ValueError('Invalid competing-risk probabilities')
    touch = tp + sl
    conditional = tp / touch if touch else 0.
    margin = conditional * reward - (1 - conditional) * loss
    reasons = list(prediction['support_flags'])
    if tp < policy['p_tp_min']: reasons.append('absolute_p_tp_below_threshold')
    if untouched > policy['p_no_touch_max']: reasons.append('no_touch_above_limit')
    if reward <= 0: reasons.append('target_does_not_clear_fees_and_spread')
    if loss <= 0 or reward/max(loss,1e-12) < policy['minimum_net_rr']: reasons.append('net_reward_risk_below_limit')
    if margin <= 0: reasons.append('conditional_hit_margin_nonpositive')
    return dict(eligible=not reasons, reasons=reasons, conditional_hit_margin=margin,
                net_reward=reward, net_loss=loss, p_tp_given_touch=conditional,
                net_rr=reward/max(loss,1e-12))


def completed_history(path, decision, needed):
    frame = load_market(path, 60)
    return validate_history(frame,decision,needed)


def validate_history(frame, decision, needed):
    frame = frame.loc[frame.timestamp + pd.Timedelta(minutes=1) <= decision].tail(needed).copy()
    if len(frame) < needed:
        raise ValueError(f'insufficient_history: {len(frame)}/{needed} completed M1 rows')
    if frame.timestamp.iloc[-1] + pd.Timedelta(minutes=1) != decision:
        raise ValueError('history_does_not_end_at_decision')
    gaps = frame.timestamp.diff().dropna().dt.total_seconds().ne(60)
    if gaps.any():
        raise ValueError(f'history_gap: {int(gaps.sum())} gaps inside {needed} required M1 rows')
    frame.attrs['interval_seconds'] = 60
    return frame


class Runner:
    def __init__(self, config):
        self.config = config
        self.history_source = {'binance_proxy':'Binance proxy, Wallex execution','kucoin_proxy':'KuCoin proxy, Wallex execution'}.get(config.get('history_mode'),'Wallex native M1')
        if 'mexc' in config.get('proxy_venues',{}).values(): self.history_source='KuCoin/MEXC proxy, Wallex execution'
        self.root = Path(config['run_dir'])
        self.root.mkdir(parents=True, exist_ok=True)
        seed_path = Path(config['capital_seed'])
        seed = json.loads(seed_path.read_text(encoding='utf-8'))
        receipt = dict(initial_usdt=seed['initial_usdt'], fee_bps=seed['fee_bps'],
                       source=str(seed_path.resolve()), source_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(), paper_only=True)
        frozen = self.root/'capital-seed.json'
        if frozen.exists() and json.loads(frozen.read_text()) != receipt:
            raise ValueError('Capital receipt changed; refusing existing ledger')
        if not frozen.exists(): save(frozen, receipt)
        self.broker = PaperBroker(self.root/'account.json', {'USDT':seed['initial_usdt'],'TMN':0}, seed['fee_bps'])
        torch.set_num_threads(1)
        self.predictor = SurvivalPredictor.load(config['model'])
        self.proxy_sources = {}
        if config.get('history_mode') == 'binance_proxy':
            manifest_path = Path(config['binance_archive'])/'manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            for compact in self.predictor.metadata['symbols']:
                record = manifest['symbols'].get(compact)
                if record is None: continue
                for name in ('values','times'):
                    if sha256(record[name+'_path']) != record[name+'_sha256']:
                        raise ValueError('Binance archive hash mismatch: '+compact)
                self.proxy_sources[compact] = record
            save(self.root/'proxy-sources.json',dict(manifest_path=str(manifest_path.resolve()),manifest_sha256=sha256(manifest_path),sources=self.proxy_sources))
        self.policy = dict(POLICY, **config.get('policy', {}))
        self.exclusions = load_exclusions(Path(__file__).resolve().parents[1]/'config/credit-excluded-markets.txt')
        self.initial = seed['initial_usdt']
        prior = self.root/'status.json'
        persisted = json.loads(prior.read_text()) if prior.exists() else {}
        self.last_scan = persisted.get('last_scan')
        self.last_boundary = persisted.get('last_boundary')
        if persisted.get('model') != config['model']: self.last_boundary = None
        if persisted.get('history_source')!=self.history_source: self.last_boundary = None
        self.worker = None
        self.errors = {}
        self.execution = threading.Lock()

    def wallex_anchor(self, compact, decision):
        shared = Path(self.config['source'])/'market-data'/compact/'wallex-cache.csv'
        cached = self.root/'wallex-anchors'/compact/'recent.csv'
        frames = [load_market(path,60) for path in (shared,cached) if path.exists()]
        native = pd.concat(frames).drop_duplicates('timestamp',keep='last') if frames else pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
        anchor = native.loc[native.timestamp==decision-pd.Timedelta(minutes=1)]
        receipts = []
        path = cached if cached.exists() else shared
        for attempt in range(3):
            if len(anchor)==1: break
            # Native responses can precede candle publication and are cached for 15 seconds.
            if attempt: time.sleep(8)
            query = urlencode(dict(symbol=compact,resolution='1',**{'from':int(decision.timestamp())-300,'to':int(decision.timestamp())-1}))
            raw,receipt = fetch_public('https://api.wallex.ir/v1/udf/history?'+query,cached.parent,'wallex_anchor',8)
            if receipt.get('status')!=200: raise ValueError('Wallex completed candle temporarily unavailable: HTTP '+str(receipt.get('status')))
            fresh,_ = normalize_history(json.loads(raw),60,decision.timestamp())
            native = pd.concat([native,fresh]).drop_duplicates('timestamp',keep='last').sort_values('timestamp').tail(300)
            anchor = native.loc[native.timestamp==decision-pd.Timedelta(minutes=1)]
            cached.parent.mkdir(parents=True,exist_ok=True)
            temporary=cached.with_suffix('.tmp');native.to_csv(temporary,index=False);atomic_replace(temporary,cached)
            path=cached;receipts.append(receipt)
        if len(anchor)!=1: raise ValueError('Wallex completed candle is still being published; retrying this decision')
        return anchor,path,receipts

    def history(self, compact, decision, needed):
        native_path = Path(self.config['source'])/'market-data'/compact/'wallex-cache.csv'
        if self.config.get('history_mode') == 'kucoin_proxy':
            from .council_runtime import _reference
            venue=self.config.get('proxy_venues',{}).get(compact,'kucoin')
            if venue not in ('kucoin','mexc'): raise ValueError('Unsupported coherent history venue')
            directory=self.root/('proxy-'+venue)/compact
            directory.mkdir(parents=True,exist_ok=True)
            cache=directory/(venue+'-cache.csv')
            quality_path=directory/'reference-quality.json'
            cache_reused=False
            try:
                frame=completed_history(cache,decision,needed)
                quality=json.loads(quality_path.read_text()) if quality_path.exists() else dict(venue=venue)
                if quality['venue']!=venue: raise ValueError('Cached proxy venue differs from configured venue')
                receipts=[];source=venue.upper()+' public spot REST, validated exact-window cache'
                cache_reused=True
            except (OSError,ValueError,KeyError):
                frame,receipts,source=_reference(dict(run_dir=str(self.root),reference_venue=venue,reference_history_days=7),directory,compact[:-4]+'/USDT',int(decision.timestamp()))
                quality=json.loads(quality_path.read_text())
                if quality['venue']!=venue: raise ValueError('Expected full coherent '+venue+' proxy history')
                frame=validate_history(frame,decision,needed)
            anchor,native_path,anchor_receipts=self.wallex_anchor(compact,decision)
            wallex_close=float(anchor.close.iloc[0]);proxy_close=float(frame.close.iloc[-1])
            ratio=wallex_close/proxy_close
            if not np.isfinite(ratio) or ratio<=0: raise ValueError('Invalid venue price basis')
            frame.loc[:,['open','high','low','close']] *= ratio
            provenance=dict(source=source+', Wallex execution',venue=venue,receipts=receipts,basis_ratio=ratio,basis_decision_at=decision.isoformat(),wallex_close=wallex_close,proxy_close=proxy_close,
                cache_reused=cache_reused,cache_validated_at=now().isoformat(),validated_history_minutes=needed,cache_quality_path=str(quality_path.resolve()),
                full_history_path=str((directory/(venue+'-cache.csv')).resolve()),full_history_sha256=sha256(directory/(venue+'-cache.csv')),
                wallex_anchor_path=str(native_path.resolve()),wallex_anchor_receipts=anchor_receipts,wallex_anchor=anchor.to_json(orient='records',date_format='iso'),
                mapping='Entire coherent '+venue+' OHLC window scaled by same-decision Wallex close / proxy close. Proxy volume retained. No hole patching.',
                calibration_limitation='Wallex-trained/calibrated probabilities have not been validated on '+venue+' proxy features. Paper research experiment.')
            save(directory/f'{decision.value}-provenance.json',provenance)
            return frame,provenance
        if self.config.get('history_mode') != 'binance_proxy':
            return completed_history(native_path,decision,needed), dict(source='Wallex native M1',path=str(native_path.resolve()))
        if compact not in self.proxy_sources: raise ValueError('missing_binance_archive_for_trained_symbol')
        record = self.proxy_sources[compact]
        times = np.load(record['times_path'],mmap_mode='r')
        values = np.load(record['values_path'],mmap_mode='r')
        lower = int(np.searchsorted(times,decision.value-needed*60_000_000_000))
        upper = int(np.searchsorted(times,decision.value))
        frame = pd.DataFrame(np.array(values[lower:upper,:5]),columns=['open','high','low','close','volume'])
        frame.insert(0,'timestamp',pd.to_datetime(np.array(times[lower:upper]),utc=True))
        directory = self.root/'proxy'/compact
        directory.mkdir(parents=True,exist_ok=True)
        cached = directory/'recent.csv'
        if cached.exists():
            frame = pd.concat([frame,load_market(cached,60)]).drop_duplicates('timestamp',keep='last').sort_values('timestamp')
        begin = int(frame.timestamp.iloc[-1].timestamp())+60
        cutoff = int(decision.timestamp())
        receipts=[]
        while begin < cutoff:
            end = min(cutoff,begin+1000*60)
            failures=[]
            for host in ('https://data-api.binance.vision','https://api.binance.com'):
                try:
                    url=host+'/api/v3/klines?'+urlencode(dict(symbol=compact,interval='1m',startTime=begin*1000,endTime=end*1000-1,limit=1000))
                    raw,receipt=fetch_public(url,directory,'binance_proxy_m1',8)
                    receipts.append(receipt)
                    if receipt.get('status')!=200: raise ValueError('HTTP '+str(receipt.get('status')))
                    rows=json.loads(raw)
                    if not isinstance(rows,list) or not rows: raise ValueError('Empty or invalid Binance candles')
                    payload=dict(s='ok',t=[r[0]/1000 for r in rows],o=[r[1] for r in rows],h=[r[2] for r in rows],l=[r[3] for r in rows],c=[r[4] for r in rows],v=[r[5] for r in rows])
                    recent,_=normalize_history(payload,60,cutoff)
                    if recent.empty: raise ValueError('No completed Binance candles')
                    frame=pd.concat([frame,recent]).drop_duplicates('timestamp',keep='last').sort_values('timestamp')
                    break
                except (ValueError,OSError,RuntimeError) as error:
                    failures.append(host+': '+str(error))
            else: raise ValueError('Binance REST unavailable: '+'; '.join(failures))
            begin=end
        frame=validate_history(frame,decision,needed)
        temporary=cached.with_suffix('.tmp');frame.to_csv(temporary,index=False);atomic_replace(temporary,cached)
        anchor,native_path,anchor_receipts=self.wallex_anchor(compact,decision)
        wallex_close=float(anchor.close.iloc[0]);binance_close=float(frame.close.iloc[-1])
        ratio=wallex_close/binance_close
        if not np.isfinite(ratio) or ratio<=0: raise ValueError('Invalid venue price basis')
        history_hash=sha256(cached)
        frame.loc[:,['open','high','low','close']] *= ratio
        provenance=dict(source='Binance proxy, Wallex execution',archive_values_sha256=record['values_sha256'],archive_times_sha256=record['times_sha256'],
            recent_cache=str(cached.resolve()),unmapped_binance_history_sha256=history_hash,receipts=receipts,
            mapping='Entire Binance OHLC window multiplied by same-decision completed Wallex close / Binance close. Volume remains Binance volume.',
            basis_ratio=ratio,basis_decision_at=decision.isoformat(),wallex_close=wallex_close,binance_close=binance_close,
            wallex_anchor_path=str(native_path.resolve()),wallex_anchor_receipts=anchor_receipts,wallex_anchor=anchor.to_json(orient='records',date_format='iso'),
            calibration_limitation='Model trained/calibrated on Wallex; Binance proxy probabilities are not validated for this distribution. Paper research experiment only.')
        save(directory/f'{decision.value}-provenance.json',provenance)
        return frame,provenance

    def feed(self):
        return json.loads((Path(self.config['source'])/'runtime-state.json').read_text(encoding='utf-8'))

    def quote(self, feed, symbol, asof):
        quote = feed.get('quotes',{}).get(symbol)
        try:
            if not isinstance(quote,dict): raise ValueError('missing_quote')
            _quote(quote,asof)
            if not all(quote.get(k,0)>0 for k in ('bid_quantity','ask_quantity')): raise ValueError('missing_executable_depth')
        except ValueError:
            from .council_runtime import fetch_quote
            quote=fetch_quote(dict(run_dir=str(self.root/'public-feed')),symbol)
            asof=now()
        _quote(quote, asof)
        if not all(quote.get(k,0)>0 for k in ('bid_quantity','ask_quantity')):
            raise ValueError('missing_executable_depth')
        return quote

    def protect(self, feed):
        with self.execution:
            account = self.broker.account
            for symbol, position in account['positions'].items():
                try:
                    stamp = now()
                    quote = self.quote(feed, symbol, stamp)
                    stamp = now()
                    reason = 'stop' if quote['bid'] <= position['stop_price'] else 'target' if quote['bid'] >= position['target_price'] else None
                    if reason:
                        quantity = min(position['quantity'], quote['bid_quantity'])
                        order_id = hashlib.sha256(f"{symbol}-{position['opened_at']}-{quote['available_at']}-{reason}".encode()).hexdigest()
                        fill = self.broker.submit(order_id,symbol,'sell',quantity,quote,stamp)
                        save(self.root/'exits'/f'{order_id}.json',dict(reason=reason,fill=fill,quote=quote))
                    self.errors.pop('protect_'+symbol,None)
                except (ValueError,KeyError,OSError) as error:
                    self.errors['protect_'+symbol] = str(error)

    def scan(self, decision):
        result = dict(decision_at=decision.isoformat(), started_at=now().isoformat(), model=self.config['model'], results=[], predictions_available=0)
        needed = self.predictor.metadata['minimum_history_minutes']
        for compact in self.predictor.metadata['symbols']:
            symbol = compact[:-4]+'/USDT'
            row = dict(symbol=symbol, status='abstain', reasons=[], candidates=[])
            result['results'].append(row)
            history_ready = False
            try:
                frame, provenance = self.history(compact,decision,needed)
                history_ready = True
                row['history_provenance'] = provenance
                quote = self.quote(self.feed(),symbol,now())
                predictions = []
                for tp, sl in GRID:
                    prediction = self.predictor.predict(frame,compact,decision,quote['ask'],tp_fraction=tp,sl_fraction=sl)
                    result['predictions_available'] += 1
                    predictions.append(prediction)
                    for forecast in prediction['forecasts']:
                        gates = payoff(prediction,forecast,quote['ask'],quote['bid'],self.broker.account['fee_bps']['USDT'],self.policy)
                        row['candidates'].append(dict(tp=prediction['tp'],sl=prediction['sl'],forecast=forecast,**gates))
                save(self.root/'predictions'/f"{decision.value}-{compact}.json",dict(history_provenance=provenance,quote=quote,predictions=predictions))
                eligible = sorted((x for x in row['candidates'] if x['eligible']),key=lambda x:(-x['conditional_hit_margin'],x['forecast']['horizon_minutes'],x['tp'],x['sl']))
                row['reasons'] = sorted({reason for x in row['candidates'] for reason in x['reasons']})
                if eligible:
                    choice = eligible[0]
                    with self.execution:
                        stamp = now()
                        fresh = self.quote(self.feed(),symbol,stamp)
                        stamp = now()
                        if (stamp-pd.Timestamp(decision).to_pydatetime()).total_seconds()>120:
                            raise ValueError('decision_older_than_120_seconds')
                        # Never transfer support/probabilities to a changed executable entry.
                        if fresh['ask'] != quote['ask']: raise ValueError('ask_changed_after_prediction')
                        fresh_gates = payoff(dict(tp=choice['tp'],sl=choice['sl'],support_flags=[]),choice['forecast'],fresh['ask'],fresh['bid'],self.broker.account['fee_bps']['USDT'],self.policy)
                        if not fresh_gates['eligible']: raise ValueError('fresh_cost_gate_failed')
                        plan = dict(stop_price=choice['sl'],target_price=choice['tp'],exit_mode='price_barriers',expires_at=None,max_open_positions=self.policy['max_open_positions'],original_entry_signal=choice)
                        account = self.broker.account
                        marks = self.feed()
                        for held_symbol, held in account['positions'].items():
                            held['mark_price'] = self.quote(marks,held_symbol,stamp)['bid']
                        analysis = dict(review_mode='model_only',regime=dict(risk_multiplier=1.),decision=dict(status='computed',stance='enter',trigger='immediate',risk_multiplier=1.))
                        risk = dict(self.policy,fee_bps=account['fee_bps'],asof=stamp.isoformat(),review_mode='model_only')
                        sized = size_order(symbol,plan,analysis,account,fresh,risk,self.exclusions)
                        if not sized['approved']: raise ValueError('; '.join(sized['reasons']))
                        fee = account['fee_bps']['USDT']/10000
                        current_risk = sum(max(0.,p['cost_basis']-p['quantity']*p['stop_price']*(1-fee)) for p in account['positions'].values())
                        if current_risk+sized['estimated_loss_at_stop']>_equity(account,'USDT')*self.policy['max_account_stop_risk_fraction']:
                            raise ValueError('account_stop_risk_limit')
                        order_id = f'survival-{decision.value}-{compact}'
                        row['fill'] = self.broker.submit(order_id,symbol,'buy',sized['quantity'],fresh,stamp,plan,self.exclusions)
                        row.update(status='paper_entered',chosen=choice,reasons=[])
            except Exception as error:
                if not history_ready: row['status']='waiting_data'
                row['reasons'].append(f'{type(error).__name__}: {error}')
        result['finished_at'] = now().isoformat()
        self.last_scan = result
        save(self.root/'scans'/f'{decision.value}.json',result)

    def publish(self):
        account = self.broker.account
        save(self.root/'status.json',dict(updated_at=now().isoformat(),paper_only=True,status='running',
             initial_usdt=self.initial,model=self.config['model'],minimum_history_minutes=self.predictor.metadata['minimum_history_minutes'],
             source=self.config['source'],policy=self.policy,grid=GRID,last_boundary=self.last_boundary,
             history_source=self.history_source,
             calibration_limitation='Model trained/calibrated on Wallex; proxy probabilities have not been validated for this distribution. Paper research experiment.' if self.config.get('history_mode','').endswith('_proxy') else None,
             scanning=bool(self.worker and self.worker.is_alive()),last_scan=self.last_scan,account=account,errors=self.errors,
             semantics='First trade-price touch probabilities by horizon. No-touch is not a win and has no assigned return. Ranking uses conditional-hit payoff only. Positions have no horizon expiry; exits require fresh executable bid and depth. Protection runs only while this process and source feed run.'))

    def run(self, seconds=None):
        start = time.monotonic()
        while seconds is None or time.monotonic()-start < seconds:
            try:
                self.protect(self.feed())
                self.errors.pop('feed',None)
                boundary = pd.Timestamp(now()).floor('5min')
                age=(pd.Timestamp(now())-boundary).total_seconds()
                retry=bool(self.last_scan and any(row['status']=='waiting_data' for row in self.last_scan['results'])
                           and (pd.Timestamp(now())-pd.Timestamp(self.last_scan['finished_at'])).total_seconds()>=10 and age<90)
                if age>=10 and (boundary.isoformat()!=self.last_boundary or retry) and not(self.worker and self.worker.is_alive()):
                    self.last_boundary = boundary.isoformat()
                    self.worker = threading.Thread(target=self.scan,args=(boundary,),daemon=True)
                    self.worker.start()
            except Exception as error:
                self.errors['feed'] = f'{type(error).__name__}: {error}'
            try:
                self.publish()
                self.errors.pop('status_publish',None)
            except PermissionError as error:
                # Windows readers may briefly block snapshots; keep protection running.
                self.errors['status_publish'] = str(error)
            time.sleep(5)


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Native survival paper account</title><style>body{font:16px system-ui;background:#101923;color:#dce8f4;max-width:1100px;margin:32px auto;padding:0 20px}h1{font-size:26px}a{color:#75cef3}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#182737;padding:20px;border-radius:8px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #365069}</style><h1>Native survival paper account</h1><p>Paper only. Independent account; same frozen initial capital and trading fees as the account-sized demo.</p><p><a href="http://127.0.0.1:8789">Original account-sized demo</a> · <a href="/status.json">Complete status and ledger</a></p><p id="summary">Loading…</p><table><thead><tr><th>Market</th><th>Decision</th><th>Reason</th></tr></thead><tbody id="rows"></tbody></table><p>No horizon expiry. No-touch is not a win. First-touch probabilities do not guarantee executable fills.</p><details><summary>Policy and model</summary><pre id="policy"></pre></details><script>async function update(){try{let s=await(await fetch('/status.json',{cache:'no-store'})).json();document.getElementById('summary').textContent=`Updated ${s.updated_at} | Starting ${s.initial_usdt.toFixed(8)} USDT | Cash ${s.account.cash.USDT.toFixed(8)} USDT | Fills ${s.account.fills.length} | Required history ${s.minimum_history_minutes} M1 | Predictions ${s.last_scan?.predictions_available??0} | ${s.scanning?'Scanning':'Waiting for next 5-minute decision'}`;document.getElementById('history').textContent=s.history_source+'. '+(s.calibration_limitation??'');let rows=document.getElementById('rows');rows.replaceChildren();for(let r of s.last_scan?.results??[]){let tr=document.createElement('tr');let best=[...(r.candidates??[])].sort((a,b)=>b.forecast.p_tp_first-a.forecast.p_tp_first)[0];for(let v of [r.symbol,r.status,(best?'Highest P(TP) '+best.forecast.p_tp_first.toFixed(3)+' at '+best.forecast.horizon_minutes+'m. ':'')+r.reasons.join('; ')]){let td=document.createElement('td');td.textContent=v;tr.append(td)}rows.append(tr)}document.getElementById('policy').textContent=JSON.stringify({model:s.model,policy:s.policy,grid:s.grid,errors:s.errors,semantics:s.semantics},null,2)}catch(e){document.getElementById('summary').textContent=String(e)}}update();setInterval(update,5000)</script></html>'''


HTML = HTML.replace('</h1>', '</h1><p id="history"></p>', 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--seconds',type=float)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('mode')!='paper': raise ValueError('Only paper mode is supported')
    root = Path(config['run_dir']); root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'runner.lock'):
        runner = Runner(config)
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ('/','/status.json'): self.send_error(404); return
                body = HTML.encode() if self.path=='/' else (root/'status.json').read_bytes()
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8' if self.path=='/' else 'application/json')
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def log_message(self,*args): pass
        runner.publish()
        server = ThreadingHTTPServer(('127.0.0.1',config['port']),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        runner.run(args.seconds)


if __name__=='__main__': main()
