import { equity, type OracleAction, type OracleOrder } from './oracle-execution.js';
import { type ActionMenu } from './oracle-crra-policy.js';
import { ZERO, rat, gt, mul, add, serializeRat, type Rational } from './oracle-rational.js';

/** Exhaustive rational refinement for a fixed finite observation tree.
 * As level grows, enumerate every rational signed quote, grid order price,
 * rational order notional, cancellation subset and ordered placement list
 * allowed by the finite book bound. Execution, not enumeration, filters
 * feasibility. This becomes expensive quickly; a small level is not an
 * epsilon certificate for continuous actions.
 */
export function exhaustiveOracleMenu(level:number):ActionMenu {
  if(!Number.isSafeInteger(level)||level<1)throw new Error('Positive enumeration level required');
  const fractions=new Map<string,Rational>();
  for(let d=1;d<=level;d++)for(let n=1;n<=level;n++){
    const q=rat(`${n}/${d}`);fractions.set(serializeRat(q),q);
  }
  return (arrival,_tree,rules)=>{
    const state=arrival.state,w=equity(state);
    if(!gt(w,ZERO))return [{id:'zero',action:{marketQuote:ZERO,cancelIds:[],place:[]}}];
    const sizes=[...fractions.values()].map(f=>mul(w,f));
    const market=[ZERO,...sizes,...sizes.map(q=>mul(rat(-1),q))];
    const cancel:string[][]=[[]];
    for(const o of state.orders)for(const s of [...cancel])cancel.push([...s,o.id]);
    const orderTypes:Omit<OracleOrder,'id'>[]=[];
    for(const side of ['buy','sell'] as const)for(let offset=1;offset<=level;offset++){
      const price=add(state.price,mul(rat(side==='buy'?-offset:offset),rules.priceStep));
      if(!gt(price,ZERO))continue;
      for(const notional of sizes)orderTypes.push({side,price,notional,placedAt:state.time});
    }
    const result:{id:string;action:OracleAction}[]=[];
    for(const cancelIds of cancel){
      const slots=rules.maxOrders-(state.orders.length-cancelIds.length);
      const visit=(place:readonly OracleOrder[])=>{
        for(const marketQuote of market){
          const id=JSON.stringify([serializeRat(marketQuote),cancelIds,place.map(o=>o.id)]);
          result.push({id,action:{marketQuote,cancelIds,place}});
        }
        if(place.length===slots)return;
        for(const o of orderTypes){
          const id=`t${state.time}:${o.side}:${serializeRat(o.price)}:${serializeRat(o.notional)}:slot${place.length}`;
          visit([...place,{...o,id}]);
        }
      };
      if(slots>=0)visit([]);
    }
    return result;
  };
}
