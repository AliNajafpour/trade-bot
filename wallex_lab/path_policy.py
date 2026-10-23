"""Exploratory, model-derived CLOSE barrier selection with disjoint sampled paths.

No market outcomes, ATR levels or quota enter this rule. Sample frequencies and
payoffs are uncalibrated model estimates; selecting among nine pairs is still
sampling-sensitive and needs independent forward validation.
"""
import numpy as np


def select_path_policy(paths,ask,fee_bps=13.,close_index=3,minimum_net_rr=1.):
    """Derive barriers on first half; score first crossings on the other half.

    A crossing exits at that model CLOSE, including overshoot. Unresolved paths
    exit at their terminal CLOSE. Entry is actual ask plus fee; exit pays fee.
    Forecast CLOSE is not a guaranteed executable bid. Physical OHLC/volume
    validity is deliberately separate from finite-positive CLOSE validity.
    """
    result=dict(approved=False,reasons=[],selected=None,candidates=[],valid_close_paths=0)
    try:
        for name,value in (('ask',ask),('fee_bps',fee_bps),('minimum_net_rr',minimum_net_rr)):
            if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        ask=float(ask);fee=float(fee_bps)/10000;minimum_net_rr=float(minimum_net_rr)
        if ask<=0 or not 0<=fee<1 or minimum_net_rr<=0:raise ValueError('positive ask/RR and fee in [0,10000) bps required')
        values=np.asarray(paths,dtype=float)
        if values.ndim==3:
            if isinstance(close_index,bool) or not isinstance(close_index,(int,np.integer)) or not 0<=close_index<values.shape[2]:raise ValueError('invalid CLOSE column')
            values=values[:,:,close_index]
        if values.ndim!=2 or values.shape[1]<1:raise ValueError('paths must be [sample,step] CLOSE or [sample,step,column]')
        valid=np.isfinite(values).all(axis=1)&(values>0).all(axis=1)
        original_indices=np.flatnonzero(valid);values=values[valid]
        result['valid_close_paths']=len(values)
        if len(values)<8:raise ValueError('at least eight valid positive CLOSE paths required')
        half=len(values)//2;derive=values[:half];score=values[half:]
        result.update(derivation_path_indices=original_indices[:half].tolist(),scoring_path_indices=original_indices[half:].tolist(),
            fee_bps_per_side=float(fee_bps),ask=ask,
            caveats=['Sample-derived frequencies and expected payoffs are uncalibrated, not observed market edge',
                     'Half-path split prevents using the same draws for levels and scoring; selecting the best of nine pairs remains optimistic',
                     'CLOSE crossings ignore intrabar touches; future quoted exits, spread, depth and slippage are unknown'])
        targets=sorted(set(float(x) for x in np.quantile(derive.max(axis=1),[.5,.75,.9])))
        stops=sorted(set(float(x) for x in np.quantile(derive.min(axis=1),[.1,.25,.5])))
        result['skipped_stop_levels']=[dict(stop_loss_price=x,reason='model stop is not strictly below ask')
                                       for x in stops if not 0<x<ask]
        stops=[x for x in stops if 0<x<ask]
        if not stops:raise ValueError('no model-derived stop strictly below ask; no arbitrary stop substitution')
        for target in targets:
            reward=target*(1-fee)/(ask*(1+fee))-1
            if target<=ask or reward<=0:continue
            for stop in stops:
                loss=1-stop*(1-fee)/(ask*(1+fee))
                rr=reward/loss
                exit_prices=[];tp=sl=terminal=0
                for path in score:
                    hits=np.flatnonzero((path>=target)|(path<=stop))
                    if len(hits):
                        price=float(path[hits[0]])
                        if price>=target:tp+=1
                        else:sl+=1
                    else:price=float(path[-1]);terminal+=1
                    exit_prices.append(price)
                net=np.asarray(exit_prices)*(1-fee)/(ask*(1+fee))-1
                candidate=dict(take_profit_price=target,stop_loss_price=stop,
                    stop_fraction=1-stop/ask,target_fraction=target/ask-1,net_rr=float(rr),
                    expected_net_return=float(net.mean()),score_paths=len(score),tp_first_count=tp,sl_first_count=sl,
                    terminal_count=terminal,tp_first_frequency=tp/len(score),sl_first_frequency=sl/len(score),
                    terminal_frequency=terminal/len(score),scored_net_returns=net.tolist(),
                    eligible=bool(rr>=minimum_net_rr and net.mean()>0))
                result['candidates'].append(candidate)
        eligible=[c for c in result['candidates'] if c['eligible']]
        if eligible:
            result.update(approved=True,selected=max(eligible,key=lambda c:(c['expected_net_return'],c['net_rr'])))
        else:result['reasons'].append('no barrier pair has both positive held-out sampled post-fee payoff and required net reward/risk')
    except (ValueError,TypeError,IndexError) as error:result['reasons'].append(str(error))
    return result
