"""Read current Wallex credit cash and create a paper-only USDT capital snapshot.

No order, transfer, activation or other state-changing tool is callable here.
Existing non-cash holdings are rejected rather than silently omitted or sold.
"""
import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def paper_seed(tmn, usdt, asks, fee_bps):
    tmn, usdt, fee_bps = map(float, (tmn, usdt, fee_bps))
    if not all(math.isfinite(x) and x >= 0 for x in (tmn, usdt, fee_bps)) or fee_bps >= 10000:
        raise ValueError('Invalid cash or conversion fee')
    remaining, quantity, fills = tmn/(1+fee_bps/10000), 0., []
    for row in sorted(asks, key=lambda x: float(x['price'])) if tmn else []:
        price, depth = float(row['price']), float(row['quantity'])
        if not math.isfinite(price) or not math.isfinite(depth) or price <= 0 or depth <= 0:
            raise ValueError('Invalid observed ask depth')
        notional = min(remaining, price*depth)
        if notional <= 0:
            break
        quantity += notional/price
        remaining -= notional
        fills.append(dict(price_tmn=price, quantity_usdt=notional/price, notional_tmn=notional))
    if remaining > max(1e-6, tmn*1e-12):
        raise ValueError('Insufficient observed USDT/TMN depth for the full virtual conversion')
    if usdt+quantity <= 0:
        raise ValueError('No available cash to mirror')
    notional = sum(row['notional_tmn'] for row in fills)
    return dict(paper_only=True, available_tmn=tmn, available_usdt=usdt, initial_usdt=usdt+quantity,
        conversion=dict(status='virtual_only_no_exchange_order', quantity_usdt=quantity,
            notional_tmn=notional, fee_tmn=notional*fee_bps/10000,
            spent_tmn=notional*(1+fee_bps/10000), fee_bps=fee_bps, fills=fills,
            fee_assumption='Same as the paper broker: fee charged on quote notional; full observed ask depth consumed. Fractional quantity is a paper proxy.'))


async def snapshot(output):
    from dotenv import load_dotenv
    from wallex_lab.mcp_client import connect, call_read_only
    from wallex_lab.shadow import validate_quote
    load_dotenv(REPO/'.env', override=True)
    cfg=dict(transport='streamable_http', url='https://api.wallex.ir/mcp', private_reads=True, timeout_seconds=35)
    output.mkdir(parents=True, exist_ok=True)
    async with connect(cfg, output/'discovery') as (session, _, tools):
        async def private(name):
            if name not in {'wallex_credit_sub_account_balances_details', 'wallex_basic_get_account_fees'}:
                raise ValueError('Only the two explicit account read tools are permitted')
            if name not in tools or 'STATE-CHANGING' in (tools[name].description or ''):
                raise ValueError('Required read-only tool unavailable')
            response=await session.call_tool(name, arguments={})
            if response.isError:
                raise ValueError('Private read failed; remote error text omitted')
            value=response.structuredContent
            if value is None:
                value=json.loads(next(b.text for b in response.content if getattr(b,'type',None)=='text'))
            if value.get('success') is not True or not isinstance(value.get('result'),dict):
                raise ValueError('Invalid private read response')
            return value['result']
        balances=await private('wallex_credit_sub_account_balances_details')
        noncash=[s for s,row in balances.items() if s not in ('TMN','USDT') and Decimal(str(row.get('total',0))) != 0]
        if noncash:
            raise ValueError('Non-cash holdings require a separate paper mirroring design; no conversion was executed')
        tmn=Decimal(str(balances.get('TMN',{}).get('available',0)))
        usdt=Decimal(str(balances.get('USDT',{}).get('available',0)))
        fees=await private('wallex_basic_get_account_fees')
        rates=[float(fees[s]['takerFeeRate'])*10000 for s in ('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT')]
        if len(set(rates)) != 1:
            raise ValueError('Per-market fees differ; quote-currency-only paper fees cannot represent this account')
        book=await call_read_only(session, 'wallex_orderbook', {'symbol':'USDTTMN'}, tools)
        now=datetime.now(timezone.utc).isoformat()
        quote=validate_quote(book['payload'],book['receipt'],now)
        result=paper_seed(tmn, usdt, book['payload']['result']['ask'], float(fees['USDTTMN']['takerFeeRate'])*10000)
        result.update(snapshot_at=now, balance_scope='selected credit sub-account, available cash only; never added to the main-wallet view',
            fee_bps={'USDT':rates[0], 'TMN':float(fees['USDTTMN']['takerFeeRate'])*10000},
            conversion_quote=dict(bid=quote['bid'],ask=quote['ask'],received_at=quote['observed_utc']),
            quote_receipt=book['receipt'], fee_source='authenticated wallex_basic_get_account_fees')
        encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
        (output/'capital-seed.json').write_text(encoded,encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('snapshot_at','available_tmn','available_usdt','initial_usdt','fee_bps','paper_only')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        asyncio.run(snapshot(args.output.resolve()))
    except Exception as error:
        # Credentials and remote bodies never enter CLI logs, even on failure.
        print('Read-only paper seed failed: '+type(error).__name__,file=sys.stderr)
        raise SystemExit(1) from None
