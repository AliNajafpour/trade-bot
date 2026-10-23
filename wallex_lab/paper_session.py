"""Bounded multi-market paper session, with asynchronous analysis and a live ledger."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
import pandas as pd
from . import context as context_api
from .context import fetch_public, parse_feed, timestamp
from .data import PublicClient, aggregate_complete, load_market, normalize_history, utc_now
from .fees import official_public_rates
from .orchestrator import Orchestrator, fresh_quote
from .shadow import validate_quote
from .specialists import _news, _fundamentals, context_fingerprint

ROOT=Path(__file__).resolve().parents[1]


class MarginTrial:
    """Explicit paper ablation; forecasts and checkpoint evidence stay unchanged."""
    def __init__(self, bank, policy):
        self.bank, self.policy=bank, policy

    def predict(self, frame, symbol):
        result=self.bank.predict(frame,symbol)
        if result.get('model_ready'):
            result['validation_selected_margin']=result['entry_margin']
            result['entry_margin_policy']=self.policy
            if self.policy=='positive-net': result['entry_margin']=0.
        return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_seen(coordinator, scan, context, symbol, asof):
    if getattr(coordinator,'config',{}).get('review_mode')=='model_only':context={}
    key=hashlib.sha256((scan['fingerprint']+context_fingerprint(context,symbol,asof)).encode()).hexdigest()
    return key in coordinator.state['seen']


def entry_window_mode(now,start,deadline,allow,finish_plans):
    if allow:return 'running'
    if now<start:return 'waiting_start'
    if now>=deadline:return 'monitoring_existing' if finish_plans else 'closing'
    return 'entries_paused'


def continue_session(now,deadline,jobs,positions,plans,finish_plans=True,barrier_only=False):
    if barrier_only:return now<deadline or bool(jobs) or bool(positions)
    return now<deadline or bool(jobs) or (finish_plans and bool(positions or plans))


def restore_forecast_cache(scanner,directory,symbols):
    """Restore actual receipt identities/clocks; scanners still validate policy and expiry."""
    cache=getattr(scanner,'cache',None)
    if cache is None:cache=getattr(scanner,'results',None)
    if cache is None:return 0
    restored=0
    for symbol in symbols:
        forecasts=list((Path(directory)/symbol.replace('/','')).glob('*/forecast.json'))
        if not forecasts:continue
        path=max(forecasts,key=lambda p:p.stat().st_mtime_ns)
        cache[symbol]=(path.parent.name,json.loads(path.read_text(encoding='utf-8')))
        restored+=1
    return restored


def save(path, value, *, required=True):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    for attempt in range(21):
        try:
            temporary.replace(path)
            return True
        except PermissionError:
            # Windows readers briefly deny replacement; keep the previous complete snapshot intact.
            if attempt==20:
                if required:raise
                return False
            time.sleep(.05)


def merge_minutes(previous, new):
    """Keep newly observed revisions; old decision snapshots remain immutable."""
    previous=previous.set_index('timestamp')
    latest=new.set_index('timestamp')
    overlap=previous.index.intersection(latest.index)
    revisions=int((previous.loc[overlap].ne(latest.loc[overlap]).any(axis=1)).sum())
    merged=pd.concat([previous, latest]).reset_index().drop_duplicates('timestamp',keep='last').sort_values('timestamp').reset_index(drop=True)
    merged.attrs['interval_seconds']=60
    return merged, revisions


def recent_minute_ranges(frame,end):
    """Bridge a stopped collector's gap using the venue's working one-day API pages."""
    start=min(end-10800,int(pd.Timestamp(frame.timestamp.iloc[-1]).timestamp())-60)
    return [(cursor,min(cursor+86400,end)) for cursor in range(start,end,86400)]


def next_candle_refresh(now,interval_seconds,frames):
    """Retry delayed venue publication without treating incomplete buckets as bars."""
    boundary=int(now)//interval_seconds*interval_seconds
    normal=boundary+interval_seconds+3
    lagging=any(frame.empty or pd.Timestamp(frame.timestamp.iloc[-1]).timestamp()+interval_seconds<boundary for frame in frames)
    return min(normal,now+30) if lagging else normal


def mark_portfolio(account, quotes, asof, rates, initial=1000.):
    cash=float(account['cash']['USDT'])
    value=0.; positions=[]; unknown=[]
    for symbol, position in account['positions'].items():
        row=dict(symbol=symbol, **position)
        quote=quotes.get(symbol)
        if not fresh_quote(quote, asof):
            row.update(mark_price=None, unrealized_pnl=None)
            unknown.append(symbol)
        else:
            net=position['quantity']*quote['bid']*(1-rates['USDT']/10000)
            value+=net
            row.update(mark_price=quote['bid'], unrealized_pnl=net-position['cost_basis'])
        positions.append(row)
    realized=sum(fill['realized_pnl'] for fill in account['fills'] if fill['side']=='sell')
    equity=None if unknown else cash+value
    closed=[fill for fill in account['fills'] if fill['side']=='sell'];wins=sum(fill['realized_pnl']>0 for fill in closed);losses=sum(fill['realized_pnl']<0 for fill in closed)
    return dict(initial_equity=initial, marked_equity=equity, cash=cash, realized_pnl=realized,
        unrealized_pnl=None if unknown else sum(p['unrealized_pnl'] for p in positions),
        total_pnl=None if equity is None else equity-initial,
        return_pct=None if equity is None else (equity/initial-1)*100,
        closed_trades=len(closed),wins=wins,losses=losses,win_rate=wins/len(closed) if closed else None,
        open_positions=len(positions),currency='USDT', fees_paid=sum(f['fee'] for f in account['fills'])), positions, unknown


def record_entry_window_report(run,state,deadline):
    """First observed snapshot at/after the deadline; never invent an exact-time mark."""
    path=Path(run)/'six-hour-report.json';captured=timestamp(state['run']['updated_at']).timestamp()
    if captured<deadline or path.exists():return False
    packet=dict(state)
    packet['timing_receipt']=dict(scheduled_deadline=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),captured_at=state['run']['updated_at'],
        lag_seconds=captured-deadline,policy='First published tick at or after entry deadline; actual quote clocks retained; no backdated prices or data-end liquidation.',
        quote_clocks={r['symbol']:r.get('quote_at') for r in state['markets']},new_entries_disabled=True,
        open_positions_remain_open=True)
    save(path,packet);return True


def collect_context(directory, index, previous, assets=()):
    """Fetch news concurrently; a failed source does not turn old news into fresh news."""
    if Path(__file__).with_name('public_sources.py').is_file():
        from .public_sources import collect_sources
        output=collect_sources(directory,assets,previous=previous)
        path=directory/f'context-{index:04d}.json'
        save(path,output)
        return path,output
    output=dict(news=[], fundamentals=[], sources=[], errors=[], available_at=utc_now())
    seen={row['id']:row for row in previous.get('news', [])}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={name:pool.submit(fetch_public,url,directory,name,6) for name,url in context_api.FEEDS.items()}
        facts=pool.submit(fetch_public,'https://api.llama.fi/v2/chains',directory,'chain_facts',6)
        for name, future in jobs.items():
            raw, receipt=future.result(); output['sources'].append(receipt)
            try:
                if receipt.get('error'): raise ValueError(receipt['error'])
                for row in parse_feed(raw,receipt,seen): seen[row['id']]=row
            except Exception as error:
                output['errors'].append(dict(source=name,error=str(error)))
        raw, receipt=facts.result(); output['sources'].append(receipt)
        try:
            if receipt.get('error'): raise ValueError(receipt['error'])
            for row in json.loads(raw):
                asset=row.get('tokenSymbol')
                value=row.get('tvl')
                if not isinstance(asset,str) or type(value) not in (int,float) or not pd.notna(value) or not 0<=value<1e100: continue
                output['fundamentals'].append(dict(id=hashlib.sha256((str(row.get('name'))+receipt['sha256']).encode()).hexdigest(),
                    source_url=receipt['url'], assets=[asset], metric='defillama_chain_tvl_usd', value=value, unit='USD',
                    text='Vendor chain TVL level in USD; not an independently verified flow, trend or intraday price signal.',
                    observed_at=receipt['response_received_utc'], available_at=receipt['response_received_utc'],
                    quality=['Source omits a metric update clock; only local receipt availability is verified.'],source_receipt=receipt))
        except Exception as error:
            output['errors'].append(dict(source='chain_facts',error=str(error)))
    now=timestamp(utc_now())
    output['news']=[row for row in seen.values() if 0<=(now-timestamp(row['available_at'])).total_seconds()<86400]
    output['available_at']=utc_now()
    output['status']='partial' if output['errors'] else 'ok'
    path=directory/f'context-{index:04d}.json'
    save(path,output)
    return path, output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Preserve ledger, history and original deadline after a stopped process')
    parser.add_argument('--universe',type=Path,default=ROOT/'data/paper-universe-20261021/universe.json')
    parser.add_argument('--models',type=Path,default=ROOT/'runs/paper-models-20261021')
    parser.add_argument('--model-family',choices=('kronos','kronos-path','chronos2','first-touch','learned_path','hgb','barrier-live','ensemble'),default='kronos')
    parser.add_argument('--barrier-policy',type=Path)
    parser.add_argument('--initial-usdt',type=float,default=1000.)
    parser.add_argument('--review-mode',choices=('model_only','model_context','legacy'),default='legacy')
    parser.add_argument('--agent-engine',choices=('legacy','langgraph'),default='legacy',
        help='Specialist engine; model_only experiments bypass language inference')
    parser.add_argument('--market-transport',choices=('rest','mcp'),default='rest',
        help='Public market source; mcp connects to the official api.wallex.ir/mcp service')
    parser.add_argument('--entry-deadline',help='Aware common entry-window end; start is deadline minus --seconds')
    parser.add_argument('--analysis-budget',type=int)
    parser.add_argument('--market-evaluation-budget',type=int)
    parser.add_argument('--first-touch-policy',type=Path)
    parser.add_argument('--learned-path-policy',type=Path)
    parser.add_argument('--forecast-context',type=int,default=2048)
    parser.add_argument('--fundamental-manifest',type=Path,default=ROOT/'data/chronos-covariates-20261021/manifest.json')
    parser.add_argument('--kronos-family',choices=('small','base'),default='small')
    parser.add_argument('--kronos-checkpoint',type=Path)
    parser.add_argument('--kronos-paths',type=int,default=16)
    parser.add_argument('--kronos-horizon','--forecast-horizon',type=int,default=12)
    parser.add_argument('--seconds',type=int,default=3600)
    parser.add_argument('--close-at-session-end',dest='finish_plans',action='store_false',default=True,
        help='Legacy bounded test: cancel pending plans and liquidate at the entry-window deadline')
    parser.add_argument('--poll-seconds',type=float,default=10)
    parser.add_argument('--candle-minutes',type=int,choices=(5,15),default=5)
    parser.add_argument('--margin-policy',choices=('validation','positive-net'),default='validation')
    args=parser.parse_args(argv)
    barrier_only=args.model_family=='barrier-live'
    strict_entry_window=barrier_only or args.entry_deadline is not None
    evaluation_budget=args.analysis_budget if args.analysis_budget is not None else (400 if barrier_only else 36)
    market_budget=args.market_evaluation_budget if args.market_evaluation_budget is not None else (100 if barrier_only else 12)
    if not 1<=evaluation_budget<=10000 or not 1<=market_budget<=100:parser.error('analysis-budget [1,10000], market-evaluation-budget [1,100]')
    common_deadline=None
    if args.entry_deadline:
        try:
            stamp=timestamp(args.entry_deadline);common_deadline=stamp.timestamp()
        except (ValueError,TypeError):parser.error('entry-deadline must be an aware ISO timestamp')
    if not math.isfinite(args.initial_usdt) or args.initial_usdt<=0:parser.error('initial-usdt must be positive finite')
    if not 60<=args.seconds<=86400 or not 5<=args.poll_seconds<=60: parser.error('seconds [60,86400], polling [5,60]')
    if args.model_family=='hgb' and args.candle_minutes!=5:parser.error('Stored HGB paper models require their trained M5 cadence')
    if args.model_family=='first-touch' and (args.candle_minutes!=15 or args.first_touch_policy is None):
        parser.error('First-touch requires --candle-minutes 15 and --first-touch-policy')
    if args.model_family=='learned_path' and (args.candle_minutes!=15 or args.learned_path_policy is None):
        parser.error('Learned path requires --candle-minutes 15 and --learned-path-policy')
    if barrier_only and (args.candle_minutes!=15 or args.barrier_policy is None or not args.finish_plans):
        parser.error('Barrier live requires M15, --barrier-policy, and monitoring without session-end liquidation')
    interval_seconds=args.candle_minutes*60
    run=args.run_dir.resolve(); run.mkdir(parents=True,exist_ok=True)
    prior=None
    if (run/'session.json').exists():
        if not args.resume: raise ValueError('Use --resume or a new directory; existing ledgers are never reset')
        prior=json.loads((run/'dashboard.json').read_text(encoding='utf-8'))
        original=json.loads((run/'session.json').read_text(encoding='utf-8'))
        if prior['run']['status'].startswith('completed'): raise ValueError('Completed sessions cannot resume')
    elif args.resume: raise ValueError('No original session to resume')
    import torch
    torch.set_num_threads(2)
    selection=json.loads(args.universe.read_text(encoding='utf-8'))
    selected=[r for r in selection['selected'] if r.get('data_status')=='ready']
    if not selected or any(r['quote_currency']!='USDT' for r in selected): raise ValueError('This shared portfolio requires prepared USDT markets')
    rates=official_public_rates()
    initial=args.initial_usdt
    source_paths=[Path(__file__),ROOT/'wallex_lab/orchestrator.py',ROOT/'wallex_lab/spot.py',ROOT/'wallex_lab/specialists.py',
        ROOT/'wallex_lab/public_sources.py',ROOT/'wallex_lab/locking.py']
    source_paths.extend(ROOT/'wallex_lab'/name for name in
        (('ensemble_scanner.py','benchmark_signals.py','horizon_models.py','model_signals.py','ai_market.py','neural.py','features.py','fees.py','fast_market.py','opportunities.py','context.py') if args.model_family=='ensemble' else
         ('barrier_live_scanner.py','learned_path_scanner.py','first_touch_scanner.py','path_signal.py','path_hgb.py','path_tcn.py','supervised_dataset.py','forecast_scanner.py','first_touch.py','chronos_market.py','scaled_kronos.py','direct_kronos.py','fast_market.py','opportunities.py','features.py','fees.py','neural.py','kronos.py','data.py','context.py') if barrier_only else
         ('kronos_market.py','direct_kronos.py','forecast_scanner.py') if args.model_family=='kronos' else
         ('learned_path_scanner.py','path_signal.py','path_hgb.py','path_tcn.py','supervised_dataset.py','forecast_scanner.py','fast_market.py','opportunities.py','features.py','fees.py','neural.py','kronos.py','data.py','context.py') if args.model_family=='learned_path' else
         ('first_touch_scanner.py','first_touch.py','forecast_scanner.py','chronos_market.py','scaled_kronos.py','direct_kronos.py','fast_market.py','opportunities.py','features.py','fees.py') if args.model_family=='first-touch' else
         ('forecast_scanner.py','chronos_market.py','chronos_covariates.py') if args.model_family=='chronos2' else
         ('forecast_scanner.py','scaled_kronos.py','direct_kronos.py','path_policy.py') if args.model_family=='kronos-path' else ('paper_models.py',)))
    if args.agent_engine=='langgraph':
        source_paths.extend(ROOT/'wallex_lab'/name for name in ('graph_agents.py','providers.py','llm.py'))
    if args.market_transport=='mcp':
        source_paths.extend(ROOT/'wallex_lab'/name for name in ('mcp_client.py','wallex_mcp.py'))
    meta=dict(pid=os.getpid(), created_at=utc_now(), duration_seconds=args.seconds, universe_sha256=sha(args.universe),
        universe_path=str(args.universe.resolve()), virtual_initial_cash_usdt=initial,
        fee_bps_per_side=rates, margin_policy=args.margin_policy,model_family=args.model_family,candle_minutes=args.candle_minutes,review_mode=args.review_mode,
        analysis_budget=evaluation_budget,market_evaluation_budget=market_budget,entry_deadline=args.entry_deadline,
        kronos_settings={'family':args.kronos_family,'checkpoint':str(args.kronos_checkpoint) if args.kronos_checkpoint else None,
            'paths':args.kronos_paths,'horizon_bars':args.kronos_horizon,'context_bars':128} if args.model_family=='kronos' else None,
        forecast_settings=dict(context=args.forecast_context,horizon_bars=args.kronos_horizon,
            fundamental_manifest=str(args.fundamental_manifest) if args.model_family in ('chronos2','first-touch') else None,
            **({'first_touch_policy_path':str(args.first_touch_policy.resolve()),'first_touch_policy_sha256':sha(args.first_touch_policy)} if args.model_family=='first-touch' else {}),
            **({'learned_path_policy_path':str(args.learned_path_policy.resolve()),'learned_path_policy_sha256':sha(args.learned_path_policy)} if args.model_family=='learned_path' else {})),
        finish_existing_plans_after_entry_window=args.finish_plans,
        real_orders=0, paper_only=True, market_transport=args.market_transport, source_sha256={p.name:sha(p) for p in source_paths})
    if args.agent_engine!='legacy':meta['agent_engine']=args.agent_engine
    if barrier_only:meta['forecast_settings'].update(barrier_policy_path=str(args.barrier_policy.resolve()),barrier_policy_sha256=sha(args.barrier_policy),exit_mode='barriers_only')
    if prior:
        if args.candle_minutes!=original.get('candle_minutes',5):raise ValueError('Resume candle cadence differs')
        if args.agent_engine!=original.get('agent_engine','legacy'):raise ValueError('Resume specialist engine differs')
        if args.market_transport!=original.get('market_transport','rest'):raise ValueError('Resume market transport differs')
        for key in ('universe_sha256','fee_bps_per_side','model_family','margin_policy','kronos_settings'):
            if meta[key]!=original[key]: raise ValueError('Resume configuration differs: '+key)
        if initial!=original['virtual_initial_cash_usdt'] or args.review_mode!=original.get('review_mode','legacy'):raise ValueError('Resume initial cash/review mode differs')
        if evaluation_budget!=original.get('analysis_budget',36) or market_budget!=original.get('market_evaluation_budget',12) or args.entry_deadline!=original.get('entry_deadline'):
            raise ValueError('Resume deadline/evaluation budgets differ')
        if 'forecast_settings' in original and meta['forecast_settings']!=original['forecast_settings']:
            raise ValueError('Resume forecast settings differ')
        snapshot_dir=run/'resume-source-snapshots'/str(time.time_ns())
        snapshot_dir.mkdir(parents=True)
        for source in source_paths: shutil.copyfile(source,snapshot_dir/source.name)
        save(snapshot_dir/'resume.json',dict(previous_report_at=prior['run']['updated_at'],resumed_at=utc_now(),
            original_deadline=prior['run']['ends_at'],source_sha256=meta['source_sha256'],pid=os.getpid(),
            limitation='No order-book observations exist during the interruption; recent candles may be retrieved later.'))
    else:
        (run/'source-snapshots').mkdir()
        for source in source_paths: shutil.copyfile(source,run/'source-snapshots'/source.name)
        save(run/'session.json',meta)
    coordinators={}; clients={}; native={}; frames={}; rows={}; quote_cache={}; counts=Counter(); rejected=Counter(); failures=[]; equity=[]
    if args.model_family=='ensemble':
        from .ensemble_scanner import EnsembleScanner
        scanner=EnsembleScanner(args.models)
        receipt_path=run/'ensemble-artifact-receipt.json'
        if receipt_path.exists() and json.loads(receipt_path.read_text())!=scanner.artifact_receipt:raise ValueError('Resume ensemble artifacts differ')
        save(receipt_path,scanner.artifact_receipt)
    elif barrier_only:
        from .barrier_live_scanner import BarrierLiveScanner
        scanner=BarrierLiveScanner(run/'forecasts',args.barrier_policy)
        receipt_path=run/'barrier-live-artifact-receipt.json'
        if receipt_path.exists() and json.loads(receipt_path.read_text())!=scanner.artifact_receipt:
            scanner.close();raise ValueError('Resume barrier artifact metadata differs')
        save(receipt_path,scanner.artifact_receipt);save(run/'selected-barrier-policy.json',scanner.policy)
    elif args.model_family=='learned_path':
        from .learned_path_scanner import LearnedPathScanner
        scanner=LearnedPathScanner(run/'forecasts',args.learned_path_policy)
        receipt_path=run/'learned-path-artifact-receipt.json'
        if receipt_path.exists() and json.loads(receipt_path.read_text(encoding='utf-8'))!=scanner.artifact_receipt:
            scanner.close();raise ValueError('Resume learned path artifact metadata differs')
        save(receipt_path,scanner.artifact_receipt)
        save(run/'selected-learned-path-policy.json',scanner.policy)
    elif args.model_family=='first-touch':
        from .first_touch_scanner import FirstTouchScanner
        scanner=FirstTouchScanner(run/'forecasts',args.first_touch_policy,fundamental_manifest=args.fundamental_manifest)
        save(run/'selected-first-touch-policy.json',scanner.policy)
    elif args.model_family=='kronos':
        from .kronos_market import KronosMarket
        scanner=KronosMarket(run/'forecasts',family=args.kronos_family,checkpoint=args.kronos_checkpoint,
            paths=args.kronos_paths,horizon=args.kronos_horizon)
    elif args.model_family in ('chronos2','kronos-path'):
        from .forecast_scanner import ForecastScanner
        scanner=ForecastScanner(run/'forecasts',family=args.model_family,context=args.forecast_context,horizon=args.kronos_horizon,
            checkpoint=args.kronos_checkpoint,fundamental_manifest=args.fundamental_manifest if args.model_family=='chronos2' else None)
    else:
        from .paper_models import ReturnModelBank, PaperScanner
        bank=MarginTrial(ReturnModelBank(args.models),args.margin_policy)
        scanner=PaperScanner(bank,dict(max_candle_age_seconds=900))
    started=datetime.now(timezone.utc); deadline=started.timestamp()+args.seconds
    if common_deadline is not None:deadline=common_deadline;started=datetime.fromtimestamp(deadline-args.seconds,timezone.utc)
    if prior:started=datetime.fromisoformat(prior['run']['started_at']);deadline=datetime.fromisoformat(prior['run']['ends_at']).timestamp()
    client_type=PublicClient
    if args.market_transport=='mcp':
        from .mcp_client import MCPPublicClient
        client_type=MCPPublicClient
    for item in selected:
        symbol=item['symbol']; compact=item['compact']
        clients[symbol]=client_type(run/'raw-data'/compact,timeout=8)
        native[symbol]=load_market(item['data']['path_1m'],60)
        initial_bars,_=aggregate_complete(native[symbol],interval_seconds,60)
        initial_path=run/'initial-candles'/f'{compact}_{args.candle_minutes}m.csv'
        initial_path.parent.mkdir(exist_ok=True)
        if initial_path.exists():pd.testing.assert_frame_equal(load_market(initial_path,interval_seconds),initial_bars,check_dtype=False,rtol=1e-12)
        else:initial_bars.to_csv(initial_path,index=False)
        frames[symbol]=(str(initial_path.resolve()),initial_bars,utc_now())
        config=dict(mode='paper',account_path=str(run/'account.json'),initial_cash={'TMN':0.,'USDT':initial},
            max_evaluations=market_budget,max_open_positions=5,max_position_fraction=.15,risk_per_trade_fraction=.005,
            min_notional={'TMN':50000.,'USDT':float(item.get('metadata',{}).get('minNotional') or 1.)},
            max_spread_bps=80.,max_candle_age_seconds=max(900,2*interval_seconds),max_analysis_age_seconds=max(900,2*interval_seconds))
        if args.model_family in ('first-touch','learned_path'):config['minimum_net_rr']=scanner.minimum_net_rr
        if barrier_only:config.update(minimum_net_rr=scanner.minimum_net_rr,entry_window_ends_at=datetime.fromtimestamp(deadline,timezone.utc).isoformat())
        elif strict_entry_window:config['entry_window_ends_at']=datetime.fromtimestamp(deadline,timezone.utc).isoformat()
        if args.review_mode!='legacy':config['review_mode']=args.review_mode
        analyzer=None
        if args.agent_engine=='langgraph':
            from .graph_agents import analyze
            analyzer=analyze
        coordinators[symbol]=Orchestrator(run/'markets'/compact,config,scanner=scanner.evaluate,analyzer=analyzer)
        rows[symbol]=dict(symbol=symbol,model_family=args.model_family,decision='loading',reason='مدل و داده در حال آماده‌سازی است',news_status='unknown',fundamental_status='unknown')
    broker=next(iter(coordinators.values())).broker
    jobs={}; evaluations=0; context={}; context_path=None; context_job=None; context_index=0
    data_pool=ThreadPoolExecutor(max_workers=4)
    analysis_pool=ThreadPoolExecutor(max_workers=2)
    context_pool=ThreadPoolExecutor(max_workers=1)
    next_candles=0.; next_context=0.; tick=0
    mode=entry_window_mode(time.time(),started.timestamp(),deadline,started.timestamp()<=time.time()<deadline,args.finish_plans)
    if prior:
        counts.update(scanned=prior['funnel']['scanned'],eligible=prior['funnel']['eligible'],analyses=prior['funnel']['analyses'],entry=prior['funnel']['entries'])
        rejected.update(prior['funnel']['rejected']);evaluations=prior['funnel']['analyses']
        equity=prior.get('equity_history',[]);failures=prior.get('failures',[])
        failures.append(dict(stage='interruption',from_at=prior['run']['updated_at'],resumed_at=utc_now(),
            error='Previous process stopped without a final summary; missing book observations are not backfilled.'))
        rows.update({row['symbol']:row for row in prior['markets']})
        for row in rows.values():
            if row.get('model_forecasts'):row.setdefault('model_estimate_at',row.get('updated_at'))
            history=run/'markets'/row['symbol'].replace('/','')/'decisions.jsonl'
            if 'last_final_decision' not in row and history.exists():
                for line in reversed(history.read_text(encoding='utf-8').splitlines()):
                    try:decision=json.loads(line).get('analysis',{}).get('decision')
                    except json.JSONDecodeError:continue
                    if decision:
                        row['last_final_decision']=decision;break
        contexts=sorted((run/'context').glob('context-*.json'))
        if contexts:
            context_path=contexts[-1];context=json.loads(context_path.read_text(encoding='utf-8'))
            context_index=int(context_path.stem.split('-')[-1])+1
        inputs=list((run/'inputs').glob('*.json'))
        tick=max((int(p.stem.split('-')[0]) for p in inputs),default=-1)+1
        restore_forecast_cache(scanner,run/'forecasts',rows)
    equity_peak=max([initial]+[p['equity'] for p in equity]);max_drawdown=0.
    if prior:
        equity_peak=max(equity_peak,prior.get('metrics',{}).get('equity_peak',initial));max_drawdown=prior.get('metrics',{}).get('max_drawdown_pct',0.)
    if args.review_mode=='model_only':
        context=dict(news=[],fundamentals=[],sources=[],errors=[],status='model_only',available_at=utc_now(),scope='No news/fundamental review performed in this numeric-model-only account.')
        context_path=run/'context'/'model-only.json';save(context_path,context)

    def fetch_market(symbol, refresh):
        client=clients[symbol]
        if refresh:
            end=int(time.time())//60*60
            merged=native[symbol];revisions=0
            for start,stop in recent_minute_ranges(merged,end):
                payload, receipt=client.get('/v1/udf/history',dict(symbol=symbol.replace('/',''),resolution='1',**{'from':start,'to':stop-1}),'recent_minutes')
                new,_=normalize_history(payload,60,min(stop,timestamp(receipt['request_started_utc']).timestamp()))
                merged,changed=merge_minutes(merged,new);revisions+=changed
            frame,_=aggregate_complete(merged,interval_seconds,60)
            # Receipt-time snapshots keep a same-minute publication retry from replacing past evidence.
            path=run/'candles'/f'{int(time.time())}'/(symbol.replace('/','')+f'_{args.candle_minutes}m.csv')
            path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False)
            native[symbol]=merged; frames[symbol]=(str(path.resolve()),frame,utc_now())
            if revisions: save(path.with_suffix('.revisions.json'),dict(revised_minutes=revisions,available_at=utc_now(),receipt=receipt))
        payload, receipt=client.get('/v1/depth',dict(symbol=symbol.replace('/','')),'book')
        quote=validate_quote(payload,receipt,utc_now())
        quote.update(available_at=quote['observed_utc'],source_receipt=receipt)
        return symbol, quote

    def publish():
        nonlocal equity_peak,max_drawdown
        asof=utc_now(); account=broker.account
        metrics, positions, unknown=mark_portfolio(account,quote_cache,asof,rates,initial)
        if metrics['marked_equity'] is not None:
            equity.append(dict(timestamp=asof,equity=metrics['marked_equity']));equity_peak=max(equity_peak,metrics['marked_equity'])
            max_drawdown=max(max_drawdown,100*(1-metrics['marked_equity']/equity_peak))
        metrics.update(max_drawdown_pct=max_drawdown,equity_peak=equity_peak)
        positioned=set(account['positions'])
        for symbol, row in rows.items():
            row['position']=symbol in positioned
            row['news_data_count']=len(_news(context,symbol,asof))
            row['fundamental_data_count']=len(_fundamentals(context,symbol,asof))
        state=dict(run=dict(id=run.name,status=mode,started_at=started.isoformat(),ends_at=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),updated_at=asof,
            entries_paused=(run/'stop-entries.flag').exists() or time.time()>=deadline,pid=os.getpid(),margin_policy=args.margin_policy,model_family=args.model_family,candle_minutes=args.candle_minutes,
            review_mode=args.review_mode,market_transport=args.market_transport,exit_mode='barriers_only' if barrier_only else 'legacy',
            finish_existing_plans_after_entry_window=args.finish_plans),metrics=metrics,markets=list(rows.values()),
            positions=positions,trades=account['fills'],equity_history=equity[-5000:],
            funnel=dict(scanned=counts['scanned'],eligible=counts['eligible'],analyses=counts['analyses'],entries=counts['entry'],rejected=dict(rejected)),
            sources=dict(unknown_count=sum(r.get('fundamental_status')!='observed' for r in rows.values()),stale_count=len(unknown),
                summary=f"{len(context.get('news',[]))} خبر؛ {len(context.get('fundamentals',[]))} داده بنیادی؛ {len(selected)} بازار؛ سیاست {args.margin_policy}",errors=context.get('errors',[]),integrations=context.get('integrations',{})),
            active_jobs=len(jobs),failures=failures[-30:],evaluation_budget=evaluation_budget,known_limits=['Observed bid/ask paper fills, unknown additional impact and actual account fees.',
                'Original trained AI signal is an entry control only; its probabilities are not probabilities for the changed TP/SL or unlimited holding. No timed position exit.' if barrier_only else
                'Marginal extrema are not joint paths. TRAIN-fitted observed-outcome probabilities and timeout returns are estimates, not guaranteed profit.' if args.model_family=='learned_path' else
                'Forecasts/quantiles are uncalibrated; historical TVL publication availability is assumed48h, not verified.' if args.model_family!='hgb' else
                'HGB models use 14-day local training; selected validation settings are exploratory.'],filter_ablation='reports/filter-ablation.md')
        if not save(run/'dashboard.json',state,required=False):
            failures.append(dict(stage='dashboard_write_delayed',at=asof,error='Windows snapshot replacement locked; retained previous dashboard and will retry next tick.'))
        if strict_entry_window:record_entry_window_report(run,state,deadline)
        return state

    def consume(symbol, result):
        counts[result['event']]+=1
        if result.get('watch_conditions') is not None:rows[symbol]['watch_conditions']=result['watch_conditions']
        if result['event'] not in ('scan','position_monitored','analysis','entry','watch'): rejected[result.get('reason') or result['event']]+=1
        analysis=result.get('analysis')
        if analysis:
            rows[symbol].pop('watch_conditions',None)
            if analysis.get('decision'):rows[symbol]['last_final_decision']=analysis['decision']
            rows[symbol].update(news_status=analysis.get('sentiment',{}).get('status','unknown'),fundamental_status=analysis.get('fundamentals',{}).get('status','unknown'),
                decision=analysis.get('decision',{}).get('stance',result['event']),trigger=analysis.get('decision',{}).get('trigger'),
                reason=analysis.get('decision',{}).get('rationale') or str(analysis.get('errors',[])))
        elif result['event'] not in ('scan','position_monitored'): rows[symbol].update(decision=result['event'],reason=result.get('message'))
        with (run/'events.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')

    publish()
    fatal_error=False
    try:
        while continue_session(time.time(),deadline,jobs,broker.account['positions'],
                any(c.state['plans'] for c in coordinators.values()),args.finish_plans,strict_entry_window) or (strict_entry_window and not (run/'six-hour-report.json').exists()):
            began=time.monotonic(); now=time.time(); allow=started.timestamp()<=now<deadline and not (run/'stop-entries.flag').exists()
            mode=entry_window_mode(now,started.timestamp(),deadline,allow,args.finish_plans)
            if strict_entry_window and not allow:
                for symbol,future in list(jobs.items()):
                    if future.cancel():del jobs[symbol]
            for symbol,future in list(jobs.items()):
                if future.done():
                    try: consume(symbol,future.result())
                    except Exception as error: failures.append(dict(stage='analysis',symbol=symbol,error=str(error),at=utc_now()))
                    del jobs[symbol]
            if strict_entry_window and not allow:
                for symbol,coordinator in coordinators.items():
                    if symbol not in jobs and coordinator.state['plans'].get(symbol):
                        consume(symbol,coordinator.process(frames[symbol][1],symbol,utc_now(),None,context,allow_entries=False))
            if context_job and context_job.done():
                try: context_path,context=context_job.result()
                except Exception as error: failures.append(dict(stage='context',error=str(error),at=utc_now()))
                context_job=None
            if args.review_mode!='model_only' and now>=next_context and context_job is None and (now<deadline or args.finish_plans and not strict_entry_window):
                context_job=context_pool.submit(collect_context,run/'context',context_index,context,[item['base_asset'] for item in selected])
                context_index+=1; next_context=now+120
            refresh=now>=next_candles
            futures=[data_pool.submit(fetch_market,item['symbol'],refresh) for item in selected]
            for future in futures:
                try:
                    symbol,quote=future.result(); quote_cache[symbol]=quote
                except Exception as error: failures.append(dict(stage='market_data',error=str(error),at=utc_now()))
            if refresh:next_candles=next_candle_refresh(time.time(),interval_seconds,[item[1] for item in frames.values()])
            candidates=[]
            held=set(broker.account['positions'])
            for item in selected:
                symbol=item['symbol']; quote=quote_cache.get(symbol); asof=utc_now()
                if fresh_quote(quote,asof):rows[symbol].update(last_price=quote['bid'],updated_at=asof,quote_at=quote['available_at'])
                if symbol in jobs or not fresh_quote(quote,asof): continue
                path,frame,available=frames[symbol]
                coordinator=coordinators[symbol]
                if symbol in held or coordinator.state['plans'].get(symbol) or not allow:
                    finish_pending=not strict_entry_window and args.finish_plans and now>=deadline and bool(coordinator.state['plans'].get(symbol)) and not (run/'stop-entries.flag').exists()
                    can_enter=(started.timestamp()<=time.time()<deadline and not (run/'stop-entries.flag').exists()) if strict_entry_window else allow or finish_pending
                    consume(symbol,coordinator.process(frame,symbol,asof,quote,context,allow_entries=can_enter))
                    continue
                scan=scanner.evaluate(frame,symbol,asof,quote,{**coordinator.config,'fee_bps':rates['USDT']})
                counts['scanned']+=1
                summary=scan.get('summary',{})
                prediction=scan.get('model_forecasts',{})
                rows[symbol].update(last_price=quote['bid'],net_return=summary.get('model_net_return',summary.get('model_median_net_return')),
                    hgb_return=summary.get('hgb_gross_return'),horizon_minutes=summary.get('model_horizon_minutes'),
                    updated_at=asof,model_estimate_at=asof,model_forecasts=prediction)
                if args.model_family!='hgb':
                    rows[symbol].update(prediction_price=summary.get('predicted_terminal_close_median'),
                        take_profit_price=summary.get('predicted_target_price'),stop_loss_price=summary.get('predicted_stop_price'))
                if scan['status']!='setup':
                    rejected[scan['reason']]+=1
                    rows[symbol].update(decision=scan['status'],reason=scan['reason'])
                    continue
                if setup_seen(coordinator,scan,context,symbol,asof):
                    rows[symbol].update(decision='already_analyzed',reason='همین پیش‌بینی و شواهد قبلاً تحلیل شده‌اند')
                    continue
                counts['eligible']+=1
                rows[symbol].update(decision='eligible',reason=scan['reason'])
                candidates.append((float(summary.get('model_margin_after_cost',summary.get('model_net_return',0))),symbol))
            if allow and context_path:
                for _,symbol in sorted(candidates,reverse=True):
                    if len(jobs)>=2 or evaluations>=evaluation_budget or strict_entry_window and (time.time()>=deadline or (run/'stop-entries.flag').exists()): break
                    path,frame,available=frames[symbol]; quote=quote_cache[symbol]; asof=utc_now()
                    if not fresh_quote(quote,asof): continue
                    save(run/'inputs'/f'{tick}-{symbol.replace("/","")}.json',dict(symbol=symbol,asof=asof,candles=path,candles_sha256=sha(path),
                        frame_available_at=available,context=str(context_path.resolve()),context_sha256=sha(context_path),quote=quote))
                    jobs[symbol]=analysis_pool.submit(coordinators[symbol].process,frame,symbol,asof,quote,context,allow_entries=True)
                    rows[symbol].update(decision='analyzing',reason='ارزیابی عددی مدل و ریسک در حال اجرا است' if args.review_mode=='model_only' else 'تحلیل خبر و تصمیم نهایی در حال اجرا است')
                    evaluations+=1; counts['analyses']+=1
            tick+=1; publish()
            if not continue_session(time.time(),deadline,jobs,broker.account['positions'],
                    any(c.state['plans'] for c in coordinators.values()),args.finish_plans,strict_entry_window) and (not strict_entry_window or (run/'six-hour-report.json').exists()): break
            remaining=args.poll_seconds-(time.monotonic()-began)
            if remaining>0: time.sleep(min(remaining,max(.1,deadline-time.time()) if now<deadline else remaining))
    except KeyboardInterrupt:
        failures.append(dict(stage='interrupt',at=utc_now(),error='User interrupt; attempting fresh-quote paper liquidation'))
    except Exception as error:
        fatal_error=True
        failures.append(dict(stage='fatal',at=utc_now(),error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        analysis_pool.shutdown(wait=True)
        for symbol,future in list(jobs.items()):
            try: consume(symbol,future.result())
            except Exception as error: failures.append(dict(stage='final_analysis',error=str(error),at=utc_now()))
        jobs.clear()
        for symbol,position in ([] if strict_entry_window else list(broker.account['positions'].items())):
            try:
                _,quote=fetch_market(symbol,False); quote_cache[symbol]=quote
                fill=broker.submit('session-end-'+symbol,symbol,'sell',position['quantity'],quote,utc_now(),exclusions=coordinators[symbol].exclusions)
                consume(symbol,dict(event='exit',symbol=symbol,asof=utc_now(),reason='session_end',fill=fill))
            except Exception as error: failures.append(dict(stage='final_exit',symbol=symbol,error=str(error),at=utc_now()))
        data_pool.shutdown(wait=True); context_pool.shutdown(wait=True)
        if args.model_family!='hgb': scanner.close()
        mode=('failed_with_open_positions' if broker.account['positions'] else 'failed') if fatal_error else ('completed_with_open_positions' if broker.account['positions'] else 'completed')
        final=publish(); save(run/'summary.json',final)
        print(json.dumps(dict(status=mode,metrics=final['metrics'],fills=len(final['trades']),run_dir=str(run))),flush=True)


if __name__=='__main__': main()
