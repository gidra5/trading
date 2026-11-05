"""Replay logged account transitions independently of the planner kernels.

Uses plain NumPy arithmetic for Oracle's signed funding, received-amount fees,
crossing fills, aggregate caps and reservations. Checks the recorded actions;
it does not validate forecasts, Bellman optimality or the authenticity of prices.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def require(condition, message):
    if not condition:raise ValueError(message)


def close_flow(state, rules):
    value=state[1]*state[2];b=1-rules['taker']
    return b*value if value>=0 else value/b


def allowed(flow, rules):
    # Logged quote deltas are recovered by subtracting rounded cash balances.
    return flow==0 or rules['minimum']-1e-8<=abs(flow)<=rules['maximum']+1e-8


def market_fee(flow,rules):
    """Quote value lost on this received-amount market trade at execution price."""
    return -flow*rules['taker'] if flow<=0 else flow*rules['taker']/(1-rules['taker'])


def arrival_components(previous,price,rules,funding):
    """Additive marked-equity attribution, before the next rebalance.

    Price P&L uses the previous quantity; the signed funding change is valued at
    the new price. Limit execution improvement excludes maker fees. Fees paid
    in received asset are also valued at this tick's price, not the limit price.
    """
    parts=dict(heldPricePnl=float(previous[1]*(price-previous[2])),
        signedFundingPnl=float(previous[1]*(funding-1)*price),
        limitExecutionImprovement=0.,makerFeesMarkedAtTick=0.)
    before=round(previous[2]/rules['tick']);after=round(price/rules['tick'])
    for i in range(4,len(previous),3):
        side,limit,notional=previous[i:i+3];level=round(limit/rules['tick'])
        hit=(side>0 and after<before and after<=level<=before) or (side<0 and before<after and before<=level<=after)
        if not hit:continue
        parts['limitExecutionImprovement']+=side*notional*(price/limit-1)
        parts['makerFeesMarkedAtTick']+=rules['maker']*notional*(price/limit if side>0 else 1.)
    return parts


def arrival(previous, price, rules, funding):
    state=np.asarray(previous,dtype=float).copy();state[3]=state[2];state[2]=price;state[1]*=funding
    effective=close_flow(state,rules);raw=state[0]+state[1]*price;wealth=state[0]+effective
    require(raw>0 and wealth>0,'Arrival has nonpositive equity')
    require(rules['maintenance_lower']-1e-10<=effective/wealth<=rules['maintenance_upper']+1e-10,
        'Arrival violates maintenance')
    tick=rules['tick'];before=round(previous[2]/tick);after=round(price/tick);consumed=0.;fills=0
    for i in range(4,len(state),3):
        side,limit,notional=state[i:i+3];level=round(limit/tick)
        hit=(side>0 and after<before and after<=level<=before) or (side<0 and before<after and before<=level<=after)
        if not hit:continue
        flow=-notional if side>0 else (1-rules['maker'])*notional
        require(allowed(flow,rules),'Committed fill violates order size')
        state[0]+=flow
        state[1]+=(1-rules['maker'])*notional/limit if side>0 else -notional/limit
        consumed+=abs(flow);fills+=1;state[i:i+3]=0.
    require(consumed<=rules['maximum']+1e-8,'Committed fills exceed aggregate cap')
    return state,consumed,fills


def check_action(arrived, post, consumed, fills, rules):
    post=np.asarray(post,dtype=float)
    require(post.shape==arrived.shape and np.isfinite(post).all(),'Invalid post-state shape or numbers')
    require(np.array_equal(post[2:4],arrived[2:4]),'Action changed current or previous market price')
    flow=post[0]-arrived[0];b=1-rules['taker']
    require(allowed(flow,rules),'Market action violates order size')
    require(consumed+abs(flow)<=rules['maximum']+1e-8,'Market action exceeds remaining tick capacity')
    expected_asset=arrived[1]-(b*flow if flow<=0 else flow/b)/arrived[2]
    # Explicit valid full closes set the computed residual to exactly zero.
    tolerance=max(1e-8,abs(arrived[0])*1e-11)/arrived[2]
    require(abs(post[1]-expected_asset)<=tolerance,'Asset delta does not match fee-aware quote flow')
    wealth=post[0]+post[1]*post[2]
    require(wealth>0,'Post-decision equity is nonpositive')
    exposure=post[1]*post[2]/wealth
    if flow!=0 or fills:
        require(rules['lower']-1e-9<=exposure<=rules['upper']+1e-9,'Post-trade exposure violates entry limits')
    previous_orders=Counter(tuple(arrived[i:i+3]) for i in range(4,len(arrived),3) if arrived[i]!=0)
    reserve=0.;orders=0
    for i in range(4,len(post),3):
        side,limit,notional=post[i:i+3]
        if side==0:
            require(limit==0 and notional==0,'Inactive order has nonzero fields');continue
        orders+=1;key=tuple(post[i:i+3])
        require(side in (-1.,1.) and limit>0 and notional>0,'Invalid active order')
        require(abs(limit/rules['tick']-round(limit/rules['tick']))<1e-6,'Off-tick limit price')
        require(allowed(-notional if side>0 else (1-rules['maker'])*notional,rules),'Order violates size bound')
        if previous_orders[key]:previous_orders[key]-=1
        else:require(limit<post[2] if side>0 else limit>post[2],'New order is not strictly passive')
        reserve+=notional/rules['reserve_leverage']
    if orders:
        require(reserve+abs(post[1]*post[2])/rules['reserve_leverage']<=wealth+1e-8,'Orders reuse unavailable equity')
    return dict(quoteFlow=float(flow),wealth=float(wealth),exposure=float(exposure),orders=orders)


def audit(directory,price_source=None):
    from oracle_dense_checkpoint import initial_archive
    directory=Path(directory);plan=json.loads((directory/'plan.json').read_text())
    rules=plan['rules'];funding=plan['fundingMultiplier'];archive=initial_archive(directory,plan);initial=archive/'front.npz'
    with np.load(initial) as data:state=data['state'].copy()
    require(len(state)==4+3*plan['grid']['max_orders'],'Initial order capacity differs from plan')
    base=state.copy();base[0]=10000.;base[1]=0.;base[4:]=0.
    initial_action=check_action(base,state,0.,0,rules)
    attribution=dict(heldPricePnl=0.,signedFundingPnl=0.,limitExecutionImprovement=0.,makerFeesMarkedAtTick=0.,
        takerFees=market_fee(initial_action['quoteFlow'],rules))
    text=(directory/'updates.jsonl').read_text()
    lines=text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    rows=[json.loads(line) for line in lines if line.strip()]
    results_path=directory/'results.json';result=json.loads(results_path.read_text()) if results_path.exists() else None
    market_trades=int(state[0]!=base[0]);fill_ticks=0;individual_fills=0;last_update=0;terminal_checked=False
    for row in rows:
        update=row['update'];require(update==last_update+1,'Missing or repeated observed update')
        if 'horizon' not in row:
            require(result is not None and result['completed'],'Terminal row lacks a completed result')
            price=result['state'][2]
        else:price=row['price']
        require(price>0 and np.isfinite(price),'Invalid observed price')
        require(abs(price/rules['tick']-round(price/rules['tick']))<1e-6,'Off-tick observed price')
        arrived,consumed,fills=arrival(state,price,rules,funding)
        components=arrival_components(state,price,rules,funding)
        arrival_change=sum(components[k] for k in ('heldPricePnl','signedFundingPnl','limitExecutionImprovement'))-components['makerFeesMarkedAtTick']
        require(abs((arrived[0]+arrived[1]*price)-(state[0]+state[1]*state[2])-arrival_change)<=1e-7,
            'Arrival attribution does not reconcile equity')
        for key,value in components.items():attribution[key]+=value
        fill_ticks+=int(fills>0);individual_fills+=fills
        if 'horizon' not in row:
            flow=close_flow(arrived,rules)
            attribution['takerFees']+=market_fee(flow,rules)
            require(allowed(flow,rules) and consumed+abs(flow)<=rules['maximum']+1e-8,'Invalid terminal close')
            expected=arrived.copy();expected[0]+=flow;expected[1]=0.;expected[4:]=0.
            expected[3]=price  # Runner canonicalizes the completed cash state.
            require(abs(arrived[0]+arrived[1]*price-row['markedEquity'])<=1e-8,'Wrong terminal pre-close mark')
            require(np.allclose(expected,result['state'],rtol=0,atol=1e-8),'Terminal result differs from executed close')
            require(abs(expected[0]-result['terminalWealth'])<=1e-8,'Incorrect terminal wealth')
            market_trades+=int(arrived[1]!=0);terminal_checked=True;state=expected
        else:
            require('state' in row,'Replay predates complete account-state logging')
            check=check_action(arrived,row['state'],consumed,fills,rules)
            attribution['takerFees']+=market_fee(check['quoteFlow'],rules)
            require(abs(check['quoteFlow']-row['tradeQuote'])<=1e-8,'Logged quote delta differs')
            require(abs(check['wealth']-row['markedEquity'])<=1e-8,'Logged equity differs')
            require(abs(check['exposure']-row['exposure'])<=1e-10,'Logged exposure differs')
            require(check['orders']==row['orders'] and bool(fills)==row['filled'],'Logged order/fill counts differ')
            expected_horizon=min(plan['arguments']['horizon'],plan['arguments']['seconds']-update)
            require(row['horizon']==expected_horizon,'Wrong planning endpoint')
            market_trades+=int(check['quoteFlow']!=0);state=np.asarray(row['state'])
        last_update=update
    if result and result['completed']:
        require(terminal_checked and last_update==plan['arguments']['seconds'],'Completed result lacks final execution')
        require(market_trades==result['marketTrades'] and fill_ticks==result['fillTicks'],'Aggregate trade/fill counts differ')
    gross=sum(attribution[k] for k in ('heldPricePnl','signedFundingPnl','limitExecutionImprovement'))
    fees=attribution['makerFeesMarkedAtTick']+attribution['takerFees']
    marked=float(state[0]+state[1]*state[2]);reconciled=10000.+gross-fees
    require(abs(marked-reconciled)<=max(1e-7,abs(marked)*1e-10),'Cumulative attribution does not reconcile equity')
    attribution.update(grossPnlForExecutedQuantities=gross,totalMarkedFees=fees,netPnl=marked-10000.,
        reconciliationError=marked-reconciled,
        scope='Additive attribution of the actual executed quantities. Removing these fees is not a counterfactual fee-free policy replay.')
    price_check=None
    if price_source is not None:
        from export_stochastic_price_forecast import load_close_prices
        with np.load(archive/'market.npz') as cloud:start_time=float(cloud['origin_seconds'])
        with np.load(initial) as data:observed_prices=[float(data['state'][2])]
        for row in rows:
            if 'horizon' in row:
                require(row['originSeconds']==start_time+row['update'],'Wrong observed timestamp')
                observed_prices.append(row['price'])
            else:observed_prices.append(result['state'][2])
        times=np.rint((start_time+np.arange(len(observed_prices)))*1000).astype(np.int64)
        actual=load_close_prices(times,Path(price_source))
        require(np.array_equal(actual,np.asarray(observed_prices)),'Logged prices differ from immutable candle closes')
        price_check=dict(source=str(price_source),checkedPrices=len(actual),maximumPriceDifference=0.)
    return dict(checkedUpdates=last_update,marketTrades=market_trades,fillTicks=fill_ticks,individualFills=individual_fills,
        terminalChecked=terminal_checked,lastState=state.tolist(),
        equityAttribution=attribution,
        sourcePriceCheck=price_check,
        scope='Independent logged execution arithmetic, plus optional timestamp/immutable-close comparison. No forecast, policy-optimality or unlogged-step certificate.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--price-source',type=Path);args=parser.parse_args()
    result=audit(args.directory,args.price_source);(args.directory/'ledger-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
