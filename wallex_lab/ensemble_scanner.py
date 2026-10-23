"""Explicit HGB/GRU benchmark ensemble; only the three actually fitted USDT assets."""
from pathlib import Path
from .ai_market import AIMarket
from .benchmark_signals import BenchmarkBank


class EnsembleBank:
    def __init__(self,models):
        root=Path(models)
        self.banks={symbol:BenchmarkBank(root/symbol/'15m'/'h360',combo='mean_hgb_gru',
            device='cpu',margin=0,fee_profile='usdt_level1_taker') for symbol in ('BTCUSDT','ETHUSDT','SOLUSDT')}
        self.artifact_receipt={symbol:bank._load()['provenance'] for symbol,bank in self.banks.items()}

    def predict(self,frame,symbol):
        if symbol not in self.banks:
            return dict(status='unavailable',model_ready=False,models={},errors={'ensemble':'No fitted HGB+GRU asset: '+str(symbol)})
        return self.banks[symbol].predict(frame,symbol)


class EnsembleScanner(AIMarket):
    def __init__(self,models):
        super().__init__(bank=EnsembleBank(models))
        self.artifact_receipt=self.bank.artifact_receipt

    def close(self):
        pass
