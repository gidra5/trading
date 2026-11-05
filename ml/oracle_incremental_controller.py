"""Resident sampler using the shared, validated Oracle account lifecycle."""
from oracle_dense_controller import DenseController
from oracle_dense_rollout import roll_market_cloud
from oracle_incremental_table import IncrementalTable,TableBudget
from oracle_incremental_observation import observe


class IncrementalController(DenseController):
    def __init__(self,current,model,state,config,*,budget=TableBudget(),root_draws=None,reference=None,**kwargs):
        if config.backend!='cuda': raise ValueError('Resident incremental planning requires CUDA')
        table = current if isinstance(current,IncrementalTable) else IncrementalTable(
            current,config.seed,config.candidates,budget)
        self.root_draws = table.B if root_draws is None else root_draws
        self.reference = reference
        if isinstance(self.root_draws,bool) or not isinstance(self.root_draws,int) or not 1<=self.root_draws<=table.B:
            raise ValueError('Invalid root draw budget')
        if table.seed!=config.seed or table.candidates!=config.candidates or table.budget!=budget:
            raise ValueError('Restored table and controller sampling settings disagree')
        super().__init__(table,model,state,config,**kwargs)
        if table.update!=self.update:
            raise ValueError('Restored sampler and account update disagree')

    def _initial_front(self,state,consumed,filled):
        front,self.initial_root = self.current.choose(state,consumed,filled,self.root_draws)
        return front

    def _plan(self,f,p,t,consumed,filled):
        table,cfg = self.current,self.config
        observed = observe(table,self.front,float(p[-1]),f,p,self.model.feature_std.cpu().numpy())
        remaining = min(self.maximum_horizon,int(cfg.terminal_seconds-t[-1]))
        kwargs = {} if self.predict is None else dict(predict=self.predict)
        cloud,market = roll_market_cloud(table.cloud,self.model,f,p,t,
            cfg.seed+cfg.forecast_origin_index+self.update,append_tail=remaining==table.H,**kwargs)
        if len(cloud.counts)-1!=remaining: raise AssertionError('Incorrect terminal horizon')
        run = table.advance(cloud)
        reference,stats = (None,None) if self.reference is None else self.reference(table,self.update)
        front,root = table.choose(self.state,consumed,filled,self.root_draws,reference)
        return table,front,dict(horizon=remaining,sweepSeconds=run['seconds'],table=run,root=root,
            market=market,observed=observed,reference=stats,fullRefresh=False,errorAllowance=None,
            approximation='Bounded incremental finite-table estimates; no global Bellman error certificate')
