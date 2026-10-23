"""Fetch actual chain TVL and preserve separate assumed/verified availability datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from wallex_lab.context import fetch_public,save_json
from wallex_lab.data import utc_now
from wallex_lab.chronos_covariates import CHAINS,FUNDAMENTAL_COLUMNS,tvl_series,join_covariates,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/chronos-covariates-20261021'))
    args=parser.parse_args();root=args.output.resolve()
    if (root/'manifest.json').exists():raise ValueError('Use fresh output; existing datasets remain frozen')
    root.mkdir(parents=True,exist_ok=True)
    manifest=dict(status='downloading',created_at=utc_now(),symbols={},errors=[],fundamental_columns=FUNDAMENTAL_COLUMNS,
        historical_availability='UNKNOWN. Measurement date+48h is a conservative research assumption, not contemporaneous evidence.',
        source='https://api.llama.fi/v2/historicalChainTvl/{chain}',
        economic_scope='Native-asset chain DeFi TVL only; not asset valuation, protocol revenue or proven intraday alpha.',
        source_model_inputs='Actual Wallex USDT M5 candles; no gold data or current facts repeated through past history.',
        modes=['retrospective_assumed_48h','prospectively_verified'],
        source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path('wallex_lab/chronos_covariates.py').resolve()):sha('wallex_lab/chronos_covariates.py')})
    save_json(root/'manifest.json',manifest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(fetch_public,'https://api.llama.fi/v2/historicalChainTvl/'+chain,root,symbol+'_tvl',20):(symbol,chain) for symbol,chain in CHAINS.items()}
        for future in as_completed(jobs):
            symbol,chain=jobs[future];raw,receipt=future.result();facts=None
            try:
                if receipt.get('error'):raise ValueError(receipt['error'])
                facts=tvl_series(json.loads(raw),receipt)
                facts.to_csv(root/(symbol+'_tvl_facts.csv'),index=False)
            except Exception as error:manifest['errors'].append(dict(symbol=symbol,error=str(error)))
            source=Path('data/kronos-history-20261021/market')/(symbol+'_5m.csv')
            bars=pd.read_csv(source);outputs={}
            for mode in manifest['modes']:
                enriched=join_covariates(bars,facts,mode)
                destination=root/mode/(symbol+'_5m.csv');destination.parent.mkdir(exist_ok=True)
                enriched.to_csv(destination,index=False)
                outputs[mode]=dict(path=str(destination),sha256=sha(destination),rows=len(enriched),
                    fundamental_known_rows=int((enriched.fund_missing==0).sum()),missing_rows=int(enriched.fund_missing.sum()))
            manifest['symbols'][symbol]=dict(chain=chain,source_candles=str(source.resolve()),source_candles_sha256=sha(source),
                fact_rows=0 if facts is None else len(facts),receipt=receipt,outputs=outputs,
                usage='past_covariates_only; future TVL and news are unknown and must not be supplied as known future inputs')
            save_json(root/'manifest.json',manifest)
            print(json.dumps({'symbol':symbol,**outputs}),flush=True)
    manifest.update(status='partial' if manifest['errors'] else 'ready',completed_at=utc_now())
    save_json(root/'manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'errors':manifest['errors']}),flush=True)


if __name__=='__main__':main()
