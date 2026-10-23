"""Reversible price units prevent upstream's absolute epsilon dominating tiny prices."""
import math
from .direct_kronos import DirectKronos, causal_window
from .kronos import _sha256


class PriceUnits:
    def __init__(self,predictor): self.predictor=predictor

    def predict(self,frame,*args,**kwargs):
        last=float(frame.close.iloc[-1])
        if not math.isfinite(last) or last<=0: raise ValueError('Positive finite context close required')
        factor=100./last
        scaled=frame.copy()
        columns=['open','high','low','close']
        scaled[columns]=scaled[columns]*factor
        if 'amount' in scaled: scaled['amount']=scaled['amount']*factor
        output=self.predictor.predict(scaled,*args,**kwargs).copy()
        output[columns]=output[columns]/factor
        if 'amount' in output: output['amount']=output['amount']/factor
        return output


class ScaledKronos(DirectKronos):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.predictor=PriceUnits(self.predictor)

    def forecast(self,frame,asof,run_dir,**kwargs):
        window,_=causal_window(frame,asof,kwargs.get('context',128))
        last=float(window.close.iloc[-1])
        self.provenance={**self.provenance,'price_unit_transform':dict(
            rule='Last completed context CLOSE becomes100; OHLC and amount scale together; volume unchanged',
            multiplier=100./last,original_last_close=last,output_units='Original exchange quote units after exact inverse',
            scaler_fit_scope='Completed input context only; no future price or outcome used',adapter_sha256=_sha256(__file__))}
        return super().forecast(frame,asof,run_dir,**kwargs)
