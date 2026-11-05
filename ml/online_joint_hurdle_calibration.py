"""Joint causal input/output adaptation through a frozen log-hurdle predictor.

The adapter has 118 input parameters (one gain/shift per feature, shared by
the two observed ticks and all three experts), and four output offsets. One
likelihood gradient updates both ends. Neural weights and normalization stay
fixed, but autograd must traverse the predictor to reach the input adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from torch import nn

from log_hurdle_sign_size_forecast import (MINIMUM_MEAN, MAXIMUM_UP_MEAN_RATIO,
    log_hurdle_nll_elements)

CONTRACT = "causal-joint-input-output-log-hurdle-v1"


@dataclass(frozen=True)
class JointCalibrationConfig:
    horizon_seconds: int
    input_enabled: bool = True
    cadence_seconds: int = 4
    update_seconds: int = 60
    window_seconds: int = 3600
    half_life_seconds: float = 3600.
    batch_size: int = 128
    input_learning_rate: float = .001
    output_learning_rate: float = .01
    input_precision: float = 32.
    output_precision: float = 8.

    def __post_init__(self):
        for key in ('horizon_seconds','cadence_seconds','update_seconds','window_seconds','batch_size'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value <= 0:
                raise ValueError('positive integer '+key+' required')
        if self.update_seconds % self.cadence_seconds:
            raise ValueError('updates must align with canonical issuance')
        for key in ('half_life_seconds','input_learning_rate','output_learning_rate','input_precision','output_precision'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:
                raise ValueError('finite positive '+key+' required')


class JointHurdleAdapter(nn.Module):
    def __init__(self, base: nn.Module, input_mean: torch.Tensor,
                 input_std: torch.Tensor, input_enabled: bool = True):
        super().__init__()
        if input_mean.shape!=(59,) or input_std.shape!=(59,) or not bool((input_std>0).all()) \
                or not bool(torch.isfinite(input_mean).all() & torch.isfinite(input_std).all()):
            raise ValueError('finite 59-feature normalization required')
        self.base=base.eval().requires_grad_(False)
        self.input_enabled=input_enabled
        self.register_buffer('input_mean',input_mean.detach().float().clone())
        self.register_buffer('input_std',input_std.detach().float().clone())
        self.input_log_gain=nn.Parameter(torch.zeros_like(self.input_mean),requires_grad=input_enabled)
        self.input_shift=nn.Parameter(torch.zeros_like(self.input_mean),requires_grad=input_enabled)
        self.output_offset=nn.Parameter(torch.zeros(4,dtype=torch.float64,device=input_mean.device))

    def input_transform(self,x):
        # Algebraically normalize -> affine -> denormalize, preserving exact
        # identity without a subtract/divide/multiply/add round trip.
        if not self.input_enabled: return x
        return x+(x-self.input_mean)*torch.expm1(self.input_log_gain)+self.input_std*self.input_shift

    def output_transform(self,raw):
        c=self.output_offset
        up=raw[...,2]*c[2].exp()/(1+(raw[...,2]/raw[...,4])*torch.expm1(c[2]))
        up=torch.minimum(up,MAXIMUM_UP_MEAN_RATIO*raw[...,4]).clamp_min(MINIMUM_MEAN)
        down=(raw[...,3]*c[3].exp()).clamp_min(MINIMUM_MEAN)
        return torch.stack((raw[...,0]+c[0],raw[...,1]+c[1],up,down,raw[...,4],raw[...,5]),dim=-1)

    def forward(self,x): return self.output_transform(self.base(self.input_transform(x)))

    def prior(self,config):
        return .5*(config.input_precision*(self.input_log_gain.square().sum()+self.input_shift.square().sum())
            +config.output_precision*self.output_offset.square().sum())

    @torch.no_grad()
    def project(self):
        self.input_log_gain.clamp_(-math.log(2),math.log(2))
        self.input_shift.clamp_(-1.,1.)
        self.output_offset[:2].clamp_(-2.,2.)
        self.output_offset[2:].clamp_(-math.log(2),math.log(2))

    def adapter_parameters(self):
        return (self.input_log_gain,self.input_shift,self.output_offset) if self.input_enabled else (self.output_offset,)


def matured_window(stamps: np.ndarray, now: int, issued: int, config: JointCalibrationConfig):
    """Choose deterministic midpoint representatives of already mature bins.

Every eligible observation contributes to its bin's weight. At most batch_size
representatives approximate the decayed window objective; there is no random
sampling, future label access, or selection based on the scoring query set.
"""
    matured=int(np.searchsorted(stamps[:issued]+config.horizon_seconds*1000,now,side='right'))
    lo=int(np.searchsorted(stamps[:matured]+config.horizon_seconds*1000,
                            now-config.window_seconds*1000,side='left'))
    if lo==matured: return np.empty(0,dtype=np.int64),np.empty(0),matured
    edges=np.linspace(lo,matured,min(config.batch_size,matured-lo)+1,dtype=np.int64)
    indices=(edges[:-1]+edges[1:]-1)//2
    all_weights=np.exp(-math.log(2)*(now-stamps[lo:matured]-config.horizon_seconds*1000)/(1000*config.half_life_seconds))
    all_weights*=min(1.,config.cadence_seconds/config.horizon_seconds)
    weights=np.add.reduceat(all_weights,edges[:-1]-lo)
    return indices,weights,matured


class OnlineJointHurdleCalibrator:
    """An explicit event clock owns adaptation; forecast queries are read-only."""
    def __init__(self,adapter: JointHurdleAdapter,config: JointCalibrationConfig):
        if adapter.input_enabled!=config.input_enabled: raise ValueError('adapter/config mismatch')
        self.adapter,self.config=adapter,config
        groups=[{'params':[adapter.output_offset],'lr':config.output_learning_rate}]
        if config.input_enabled:
            groups.append({'params':[adapter.input_log_gain,adapter.input_shift],'lr':config.input_learning_rate})
        self.optimizer=torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8)
        self.last_update_ms=None
        self.updates=0

    def update(self,now:int,inputs:torch.Tensor,targets:torch.Tensor,
               origins_ms:np.ndarray,weights:torch.Tensor,*,raw:torch.Tensor|None=None):
        if self.last_update_ms is not None and now<=self.last_update_ms:
            raise ValueError('update clock must advance')
        if not len(origins_ms) or np.any(origins_ms+self.config.horizon_seconds*1000>now):
            raise ValueError('only already-matured labels may enter the loss')
        if np.any(np.diff(origins_ms)<=0) or inputs.shape[0]!=len(origins_ms) \
                or targets.shape!=(len(origins_ms),1) or weights.shape!=(len(origins_ms),):
            raise ValueError('ordered matching update arrays required')
        if not bool(torch.isfinite(inputs).all() & torch.isfinite(targets).all() & torch.isfinite(weights).all()) \
                or not bool((weights>0).all()): raise ValueError('finite positive-weight observations required')
        self.optimizer.zero_grad(set_to_none=True)
        if raw is not None and self.config.input_enabled: raise ValueError('input adaptation needs live neural forward')
        prediction=self.adapter(inputs) if raw is None else self.adapter.output_transform(raw)
        nll=log_hurdle_nll_elements(prediction,targets).reshape(-1)
        loss=((weights*nll).sum()+self.adapter.prior(self.config))/(weights.sum()+self.config.output_precision)
        loss.backward()
        gradients=[p.grad for p in self.adapter.adapter_parameters()]
        if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError('missing or nonfinite end-to-end gradient')
        inorm=0. if not self.config.input_enabled else float(torch.sqrt(sum(g.square().sum() for g in gradients[:2])))
        onorm=float(self.adapter.output_offset.grad.norm())
        norm=torch.nn.utils.clip_grad_norm_(self.adapter.adapter_parameters(),10.)
        if not bool(torch.isfinite(loss) & torch.isfinite(norm)): raise ValueError('nonfinite online objective')
        self.optimizer.step();self.adapter.project()
        self.last_update_ms=now;self.updates+=1
        return {'timestamp':now,'latestMaturityUsed':int(origins_ms[-1]+self.config.horizon_seconds*1000),
            'loss':float(loss.detach()),'inputGradientNorm':inorm,'outputGradientNorm':onorm,
            'effectiveWeight':float(weights.sum()),'representatives':len(origins_ms),'updates':self.updates}

    @torch.no_grad()
    def predict(self,now:int,inputs:torch.Tensor,*,raw:torch.Tensor|None=None):
        if self.last_update_ms is not None and now<self.last_update_ms:
            raise ValueError('query precedes incorporated observations')
        if raw is not None and self.config.input_enabled: raise ValueError('adapted inputs invalidate raw cache')
        return self.adapter(inputs) if raw is None else self.adapter.output_transform(raw)

    def state_dict(self):
        return {'contract':CONTRACT,'config':asdict(self.config),'lastUpdateTimestamp':self.last_update_ms,
            'updates':self.updates,'adapter':{name:value.detach().clone() for name,value in (
                ('input_log_gain',self.adapter.input_log_gain),('input_shift',self.adapter.input_shift),
                ('output_offset',self.adapter.output_offset))},'optimizer':self.optimizer.state_dict()}

    def load_state_dict(self,state):
        if state['contract']!=CONTRACT or state['config']!=asdict(self.config):
            raise ValueError('online state/config mismatch')
        with torch.no_grad():
            for name,value in state['adapter'].items():
                target=getattr(self.adapter,name)
                if target.shape!=value.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError('invalid adapter checkpoint')
                target.copy_(value)
        self.optimizer.load_state_dict(state['optimizer'])
        self.last_update_ms,self.updates=state['lastUpdateTimestamp'],state['updates']


def replay_joint(calibrator,features,stamps,targets_at,raw,*,progress=None):
    """Replay a fixed update clock, batching forecasts between updates.

Each row's predictor is independent of other rows in its inference batch.
Features of later rows cannot change an earlier row or the current adapter.
The target callback is invoked only for indices whose endpoints are observed.
"""
    config=calibrator.config
    if len(stamps)!=len(features) or raw.shape!=(len(stamps),1,6) \
            or np.any(np.diff(stamps)!=1000*config.cadence_seconds):
        raise ValueError('matching canonical four-second streams required')
    width=config.update_seconds//config.cadence_seconds
    result=np.empty((len(stamps),6),dtype=np.float64)
    trace={'coefficients':np.empty((len(stamps),4)),
        'inputLogGain':np.empty((len(stamps),59),dtype=np.float32),
        'inputShift':np.empty((len(stamps),59),dtype=np.float32),
        'latestMaturityUsed':np.full(len(stamps),-1,dtype=np.int64),
        'maturedObservations':np.zeros(len(stamps),dtype=np.int64),
        'lastUpdateTimestamp':np.full(len(stamps),-1,dtype=np.int64)}
    updates=[];latest=-1
    for start in range(0,len(stamps),width):
        stop=min(start+width,len(stamps));now=int(stamps[start])
        chosen,weights,matured=matured_window(stamps,now,start,config)
        if len(chosen):
            y=np.asarray(targets_at(chosen),dtype=np.float64)
            event=calibrator.update(now,features[chosen],torch.as_tensor(y[:,None],device=features.device),
                stamps[chosen],torch.as_tensor(weights,device=features.device),
                raw=None if config.input_enabled else raw[chosen])
            event.update({'firstRepresentative':int(chosen[0]),'lastRepresentative':int(chosen[-1]),
                'availableMaturedObservations':matured})
            updates.append(event);latest=max(latest,event['latestMaturityUsed'])
        # Before the first adaptation, reuse the exact archived base forecast.
        if not calibrator.updates:
            result[start:stop]=raw[start:stop,0].cpu().numpy()
        else:
            result[start:stop]=calibrator.predict(now,features[start:stop],
                raw=None if config.input_enabled else raw[start:stop]).cpu().numpy()[:,0]
        a=calibrator.adapter
        trace['coefficients'][start:stop]=a.output_offset.detach().cpu().numpy()
        trace['inputLogGain'][start:stop]=a.input_log_gain.detach().cpu().numpy()
        trace['inputShift'][start:stop]=a.input_shift.detach().cpu().numpy()
        trace['latestMaturityUsed'][start:stop]=latest
        trace['maturedObservations'][start:stop]=matured
        trace['lastUpdateTimestamp'][start:stop]=-1 if calibrator.last_update_ms is None else calibrator.last_update_ms
        if progress and start%(width*120)==0: progress(start,len(stamps),len(updates))
    if not np.isfinite(result).all() or np.any(trace['latestMaturityUsed']>stamps):
        raise ValueError('invalid output or future label access')
    return {'parameters':result,'snapshots':trace,'updates':updates,
        'canonicalIssues':len(stamps),'onlineOptimizerSteps':len(updates),
        'finalState':{k:values[-1].tolist() for k,values in trace.items()},
        'forecastBatchIndependence':'each observed two-tick row is evaluated independently with a fixed adapter between updates'}
