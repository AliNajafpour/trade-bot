"""Portable MCP-fed paper trading with LangGraph and configurable inference."""
import argparse
import asyncio
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

from .data import aggregate_complete, load_market, normalize_history, utc_now
from .spot import canonical_symbol


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict): raise ValueError('Configuration must be an object')
    if value.keys()-{'mode','initial_usdt','poll_seconds','context_seconds','symbols','scanner','mcp','risk','run_dir','models','policy'}:
        raise ValueError('Unknown configuration fields; keep credentials in environment variables')
    if not isinstance(value.get('risk',{}),dict) or not isinstance(value.get('mcp',{}),dict):
        raise ValueError('risk and mcp must be configuration objects')
    config = dict(mode='paper', initial_usdt=100., poll_seconds=15., context_seconds=300.,
                  symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], scanner='hgb', mcp={})
    config.update(value)
    if config['mode'] != 'paper': raise ValueError('Portable execution supports paper mode only')
    symbols = config['symbols']
    if not isinstance(symbols, list) or not 1 <= len(symbols) <= 100: raise ValueError('Use 1..100 symbols')
    config['symbols'] = [canonical_symbol(symbol) for symbol in symbols]
    if len(set(config['symbols'])) != len(symbols) or any(not x.endswith('/USDT') for x in config['symbols']):
        raise ValueError('Use distinct USDT spot markets in this shared portfolio')
    for key, low, high in (('initial_usdt', .01, 1e9), ('poll_seconds', 5, 300), ('context_seconds', 60, 3600)):
        number = config[key]
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError('Invalid ' + key)
    if config['scanner'] not in ('hgb', 'barrier-live', 'learned_path'): raise ValueError('Unknown scanner')
    for key in ('run_dir', 'models') + (('policy',) if config['scanner'] != 'hgb' else ()):
        raw = config.get(key)
        if not isinstance(raw, str) or not raw: raise ValueError('Explicit relative or absolute ' + key + ' required')
        config[key] = str((path.parent / raw).resolve())
    from .mcp_client import validate_config
    config['mcp'] = validate_config(config['mcp'])
    config['source_config'] = str(path)
    return config


def history_ranges(previous, end, scanner):
    days = 1 if scanner == 'hgb' else 7
    start = int(previous.timestamp.iloc[-1].timestamp()) - 60 if previous is not None and len(previous) else end-days*86400
    return [(cursor, min(cursor+86400, end)) for cursor in range(start, end, 86400)]


async def fetch_market(config, symbol):
    from .mcp_client import connect, call_read_only
    from .shadow import validate_quote
    compact = symbol.replace('/', '')
    directory = Path(config['run_dir']) / 'market-data' / compact
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / 'native.csv'
    native = load_market(cache, 60) if cache.exists() else None
    end = int(time.time()) // 60 * 60
    async with connect(config['mcp'], directory) as (session, _, tools):
        from .paper_session import merge_minutes
        quality=[]
        for start, stop in history_ranges(native, end, config['scanner']):
            history = await call_read_only(session, 'wallex_history', dict(symbol=compact, start=start, end=stop), tools)
            new, report = normalize_history(history['payload'], 60,
                min(stop, datetime.fromisoformat(history['receipt']['request_started_utc']).timestamp()))
            native = new if native is None else merge_minutes(native, new)[0]
            quality.append(report)
        if native is None or native.empty: raise ValueError('No real candles available')
        temporary=cache.with_suffix('.tmp')
        native.to_csv(temporary,index=False)
        temporary.replace(cache)
        interval = 300 if config['scanner'] == 'hgb' else 900
        frame, aggregation = aggregate_complete(native, interval, 60)
        snapshot = directory / str(time.time_ns())
        snapshot.mkdir(parents=True, exist_ok=True)
        native.to_csv(snapshot / 'native.csv', index=False)
        frame.to_csv(snapshot / 'candles.csv', index=False)
        (snapshot / 'quality.json').write_text(json.dumps(dict(native=quality, aggregation=aggregation)), encoding='utf-8')
        # Fetch executable quotes last so candle downloads cannot age them into entry.
        book = await call_read_only(session, 'wallex_orderbook', dict(symbol=compact), tools)
        quote = validate_quote(book['payload'], book['receipt'], utc_now())
        quote.update(available_at=quote['observed_utc'], source_receipt=book['receipt'])
    return frame, quote


def coordinators(config):
    from .graph_agents import analyze
    from .orchestrator import Orchestrator
    root = Path(config['run_dir'])
    if config['scanner'] == 'hgb':
        from .paper_models import PaperScanner, ReturnModelBank
        scanner = PaperScanner(ReturnModelBank(config['models']))
    elif config['scanner'] == 'learned_path':
        from .learned_path_scanner import LearnedPathScanner
        scanner = LearnedPathScanner(root / 'forecasts', config['policy'])
    else:
        from .barrier_live_scanner import BarrierLiveScanner
        scanner = BarrierLiveScanner(root / 'forecasts', config['policy'])
    settings = dict(max_evaluations=100, max_open_positions=5, max_candle_age_seconds=1800.,
                    minimum_net_rr=getattr(scanner, 'minimum_net_rr', 1.))
    settings.update(config.get('risk', {}))
    settings.update(mode='paper', account_path=str(root / 'account.json'), initial_cash={'USDT':config['initial_usdt'], 'TMN':0.})
    return {symbol:Orchestrator(root / 'markets' / symbol.replace('/', ''), settings,
                               analyzer=analyze, scanner=scanner.evaluate) for symbol in config['symbols']}


async def run_tick(config, agents, context, jobs=None):
    wait = jobs is None
    jobs = {} if jobs is None else jobs
    events = []
    async def collect(symbol):
        try:
            if symbol in jobs:
                if not jobs[symbol].done():
                    return dict(event='analysis_pending',symbol=symbol)
                result=jobs.pop(symbol).result()
                return result
            frame, quote = await fetch_market(config, symbol)
            task=asyncio.create_task(asyncio.to_thread(agents[symbol].process,frame,symbol,utc_now(),quote,context))
            if wait:return await task
            jobs[symbol]=task
            return dict(event='analysis_pending',symbol=symbol)
        except Exception as error:
            # Remote exception bodies can contain credentials; boundary receipts carry safe details.
            return dict(event='tick_failed', symbol=symbol, error_type=type(error).__name__)
    events=await asyncio.gather(*(collect(symbol) for symbol in config['symbols']))
    result = dict(updated_at=utc_now(), engine='langgraph', transport=config['mcp']['transport'], paper_only=True, events=events)
    root = Path(config['run_dir'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'ticks.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return result


async def run(config, seconds):
    from .public_sources import collect_sources
    agents = coordinators(config)
    deadline = time.monotonic() + seconds
    context, next_context, jobs = {}, 0., {}
    root = Path(config['run_dir'])
    while True:
        if time.monotonic() >= next_context:
            context = await asyncio.to_thread(collect_sources, root / 'context',
                [symbol.split('/')[0] for symbol in config['symbols']], context)
            (root / 'context' / 'current.json').write_text(json.dumps(context, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            next_context = time.monotonic() + config['context_seconds']
        result = await run_tick(config, agents, context, jobs)
        print(json.dumps(dict(updated_at=result['updated_at'], events=[{k:r[k] for k in ('event','symbol') if k in r} for r in result['events']]), ensure_ascii=False), flush=True)
        if time.monotonic() >= deadline: break
        await asyncio.sleep(min(config['poll_seconds'], max(0, deadline-time.monotonic())))
    if jobs:
        completed=await asyncio.gather(*jobs.values(),return_exceptions=True)
        for symbol,event in zip(jobs,completed):
            row=event if isinstance(event,dict) else dict(event='tick_failed',symbol=symbol,error_type=type(event).__name__)
            with (root/'ticks.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(dict(updated_at=utc_now(),events=[row]),ensure_ascii=False)+'\n')
    # Stop the collector only; positions stay recorded. Resume to continue exit monitoring.


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('probe', 'analyze', 'run'))
    parser.add_argument('--config', default='config/portable-paper.example.json')
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/portable-probe'))
    parser.add_argument('--seconds', type=int, default=60)
    args = parser.parse_args(argv)
    if not 0 <= args.seconds <= 86400: parser.error('seconds must be within 0..86400')
    config = load_config(args.config)
    if args.command == 'probe':
        from .mcp_client import probe
        result = asyncio.run(probe(config['mcp'], args.output, config['symbols'][0].replace('/', '')))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({key:result[key] for key in ('initialized','server','protocol','tools','read_only')}))
    elif args.command == 'analyze':
        from .graph_agents import analyze
        if args.packet is None: parser.error('analyze requires --packet')
        packet = json.loads(args.packet.read_text(encoding='utf-8'))
        result = analyze(packet['summary'], packet.get('context', {}), packet['symbol'], packet['asof'], args.output)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(status=result['status'], decision=result['decision'], graph=result['graph']), ensure_ascii=False))
    else:
        asyncio.run(run(config, args.seconds))


if __name__ == '__main__': main()
