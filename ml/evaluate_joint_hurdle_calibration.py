"""Prespecified causal input/output calibration experiment on frozen models."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from online_joint_hurdle_calibration import (CONTRACT,JointCalibrationConfig,JointHurdleAdapter,
    OnlineJointHurdleCalibrator,replay_joint)
from train_per_horizon_log_hurdle import load_combined,verified
from train_neural_sign_size_forecast import HORIZONS,sha,write_json,now,select_origins
from stochastic_price_forecast import load_timeline_split
from evaluate_log_hurdle_forecast import array_sha,score_log_law,export_stream
from evaluate_online_hurdle_calibration import CachedStream

VARIANTS=('output-only-calibrated','joint-calibrated')


def save_result(root,split,h,variant,stream,simulation,calibrator):
    prefix=root/'cache'/f'{split}-{variant}-h{h}'
    prefix.parent.mkdir(parents=True,exist_ok=True)
    file=prefix.with_suffix('.npz');trace=prefix.with_suffix('.trace.npz')
    events=prefix.with_suffix('.updates.json');state=prefix.with_suffix('.state.pt')
    np.savez_compressed(file,parameters=simulation['parameters'],origins=stream.origins,
        timestamps=stream.timestamps,accuracyIndices=stream.accuracy_indices)
    np.savez_compressed(trace,origins=stream.origins,timestamps=stream.timestamps,**simulation['snapshots'])
    write_json(events,{'contract':CONTRACT,'events':simulation['updates']})
    torch.save(calibrator.state_dict(),state)
    return {'parametersFile':str(file),'parametersSha256':sha(file),'traceFile':str(trace),'traceSha256':sha(trace),
        'updatesFile':str(events),'updatesSha256':sha(events),'stateFile':str(state),'stateSha256':sha(state),
        'canonicalIssues':len(stream.origins),'onlineOptimizerSteps':simulation['onlineOptimizerSteps'],
        'finalState':simulation['finalState']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('data/backtests/oracle-joint-calibration'))
    p.add_argument('--baseline',type=Path,default=Path('data/backtests/oracle-per-horizon-log/evaluation.json'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'evaluation.json').exists(): raise ValueError('completed experiment already exists')
    cache=Path('data/training/cache/torchinductor').resolve()
    os.environ['TORCHINDUCTOR_CACHE_DIR'],os.environ['TRITON_CACHE_DIR']=str(cache),str(cache/'triton')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    # Four independent triplets, grad/no-grad paths and early small batches
    # need more than Dynamo's default eight specializations of this method.
    torch._dynamo.config.recompile_limit=64
    torch._dynamo.config.fail_on_recompile_limit_hit=True
    hashes={}
    def checked(path,digest=None):
        actual=sha(Path(path))
        if digest is not None and actual!=digest: raise ValueError('artifact changed: '+str(path))
        hashes[str(path)]=actual;return Path(path)
    def read(path,digest=None):return json.loads(checked(path,digest).read_text())
    baseline=read(args.baseline)
    for file,digest in baseline['artifactHashes'].items():checked(file,digest)
    source_file=Path(baseline['selectionFile']);source=read(source_file,baseline['selectionSha256'])
    dataset=Path(source['plan']['dataset'])
    for file in (Path(__file__),Path('ml/online_joint_hurdle_calibration.py'),
            Path('ml/log_hurdle_sign_size_forecast.py'),Path('ml/evaluate_log_hurdle_forecast.py')):checked(file)
    configs={str(h):{v:asdict(JointCalibrationConfig(h,input_enabled=v=='joint-calibrated')) for v in VARIANTS} for h in HORIZONS}
    plan={'contract':CONTRACT,'createdAt':now(),'baseSelectionFile':str(source_file),
        'baseSelectionSha256':sha(source_file),'baselineFile':str(args.baseline),'baselineSha256':sha(args.baseline),
        'configs':configs,'testUsedForSelection':False,'hyperparameterSearch':False,'settingsFrozenBeforeTest':True,
        'optimizer':'Adam; one end-to-end full log-hurdle NLL plus identity priors per fixed 60-second update',
        'inputs':'diagonal affine in training-normalized feature units, shared over the two observed ticks and three experts; independent per horizon',
        'frozen':'all predictor weights, shapes and training normalization; no detached predictor in the joint gradient path',
        'window':'last 3600 seconds of matured labels, exp half-life3600; at most128 deterministic midpoint representatives with total bin weights',
        'overlapWeight':'min(1,4/horizonSeconds), retained heuristic for effective information',
        'comparison':'same end-to-end optimizer with input adapter disabled, plus archived Fisher output calibration and frozen raw forecasts',
        'splits':'identity adapters and new optimizer per variant/horizon/split; validation does not initialize test',
        'numerics':'float32/high matmul frozen neural network, float64 probability law and losses; compiled experts',
        'compilerRecompileLimit':64,
        'sourceHashes':dict(hashes)}
    plan_file=args.output/'plan.json'
    if plan_file.exists():
        old=json.loads(plan_file.read_text());plan['createdAt']=old['createdAt']
        if old!=plan: raise ValueError('prespecified plan changed')
    else:write_json(plan_file,plan)
    checked(plan_file)
    correction=args.output/'runtime-correction.json'
    if correction.exists():
        for file,digest in read(correction)['preservedFiles'].items():checked(file,digest)
    replay={**read(args.baseline.parent/'replay-plan.json'),'contract':'oracle-joint-calibration-comparison-v1',
        'runsRoot':str(args.output),'reportScript':'scripts/report-oracle-joint-calibration.mjs',
        'variants':[{'name':v,'feeScales':[1,0]} for v in VARIANTS],
        'certificateCoverage':'Every decision in all 12 replays',
        'selection':'fixed calibration protocol on frozen independently selected log models',
        'onlineAdaptation':plan['optimizer']+'; both adapters start at identity and use only mature labels',
        'limitations':'Previously inspected historical period; hyperparameters not selected on test. Approximate online optimization, overlap weighting and quadrature; finite-menu Lean decision checks.'}
    write_json(args.output/'replay-plan.json',replay);checked(args.output/'replay-plan.json')
    selection={'contract':CONTRACT,'planFile':str(plan_file),'planSha256':sha(plan_file),
        'baseSelectionFile':str(source_file),'baseSelectionSha256':sha(source_file),'configs':configs,
        'testLoaded':False,'testUsedForSelection':False,'allSettingsFrozen':True}
    selection_file=args.output/'selection.json'
    write_json(selection_file,selection);checked(selection_file)
    started=time.monotonic();results=[];exports=[]
    def emit(value):
        line=json.dumps(value,allow_nan=False);print(line,flush=True)
        with (args.output/'progress.jsonl').open('a') as f:f.write(line+'\n')
    for split in ('validation','test'):
        if split=='test':
            write_json(args.output/'validation-complete.json',{'selectionSha256':sha(selection_file),
                'settingsChanged':False,'results':[r for r in results if r['split']=='validation']})
            checked(args.output/'validation-complete.json')
        timeline=load_timeline_split(dataset,split)
        common=select_origins(timeline,4,None)
        if len(common)!=(34560 if split=='validation' else 36134):raise ValueError('population mismatch')
        for j,h in enumerate(HORIZONS):
            old=next(r for r in baseline['results'] if r['split']==split and r['horizonSeconds']==h)
            manifest=old['oldCanonicalCacheManifest'];entry=source['horizons'][j]
            with np.load(checked(manifest['marketFile'],manifest['marketSha256'])) as z:
                prices=z['prices'];first=int(z['firstPriceOrigin']);axis=int(z['axisStartMs'])
            raw_record=old['models']['finetunedRaw']['savedStream']
            with np.load(checked(raw_record['parametersFile'],raw_record['parametersSha256'])) as z:
                raw=z['parameters'];origins=z['origins'];stamps=z['timestamps'];accuracy=z['accuracyIndices']
            np.testing.assert_array_equal(origins[accuracy],common)
            offsets=common-first;targets=prices[offsets+h]/prices[offsets]-1
            sample_hashes={'origins':array_sha(common),'timestamps':array_sha(stamps[accuracy]),
                'simpleTargets':array_sha(targets),'logTargets':array_sha(np.log1p(targets))}
            if sample_hashes!=old['sampleHashes']:raise ValueError('target hash mismatch')
            stream=CachedStream(origins,stamps,raw,prices,first,accuracy,axis,h)
            features=torch.as_tensor(np.asarray(timeline.timeline[origins[:,None]+np.array([-1,0])],dtype=np.float32),device='cuda')
            raw_tensor=torch.as_tensor(raw[:,None],device='cuda')
            def targets_at(indices):
                offsets=origins[indices]-first
                return np.log1p(prices[offsets+h]/prices[offsets]-1)
            ck=verified(entry['finetuned']);base=load_combined(entry['finetuned'],'cuda')
            checked(entry['finetuned']['checkpoint'],entry['finetuned']['sha256'])
            for expert in (base.zero,base.sign,base.size):
                expert.forward=torch.compile(expert.forward,dynamic=True,options={'triton.cudagraphs':False})
            row={'split':split,'horizonSeconds':h,'examples':len(common),'activeExamples':int((targets!=0).sum()),
                'sampleHashes':sample_hashes,'constants':entry['constants'],'marketManifest':manifest,
                'models':{'frozenRaw':old['models']['finetunedRaw'],'fisherOutputOnly':old['models']['finetunedCalibrated']}}
            for variant in VARIANTS:
                config=JointCalibrationConfig(**configs[str(h)][variant])
                artifact=args.output/'cache'/f'{split}-{variant}-h{h}.result.json'
                if artifact.exists():
                    metrics=read(artifact)
                    for name in ('parameters','trace','updates','state'):
                        checked(metrics['savedStream'][name+'File'],metrics['savedStream'][name+'Sha256'])
                    row['models'][variant]=metrics
                    emit({'event':'resumeCompleted','split':split,'horizonSeconds':h,'variant':variant})
                    if split=='test' and h!=1:
                        out=read(args.output/f'{variant}-h{h}.manifest.json')
                        checked(out['forecastFile'],out['forecastSha256']);exports.append(out)
                    continue
                adapter=JointHurdleAdapter(base,ck['inputMean'].cuda(),ck['inputStd'].cuda(),config.input_enabled)
                calibrator=OnlineJointHurdleCalibrator(adapter,config)
                phase=time.monotonic()
                def progress(index,count,updates):emit({'event':'online','split':split,'horizonSeconds':h,
                    'variant':variant,'issues':index,'total':count,'updates':updates,'seconds':time.monotonic()-phase})
                simulation=replay_joint(calibrator,features,stamps,targets_at,raw_tensor,progress=progress)
                # Base includes buffers; verify all weights/statistics remain exact.
                for name,value in base.state_dict().items():
                    torch.testing.assert_close(value.cpu(),ck['modelState'][name],rtol=0,atol=0)
                metrics=score_log_law(simulation['parameters'][accuracy],targets,entry['constants'])
                saved=save_result(args.output,split,h,variant,stream,simulation,calibrator)
                metrics.update({'savedStream':saved,'config':asdict(config),'frozenPredictorVerified':True,
                    'seconds':time.monotonic()-phase,'inputNonzeroGradientUpdates':sum(e['inputGradientNorm']>0 for e in simulation['updates']),
                    'outputNonzeroGradientUpdates':sum(e['outputGradientNorm']>0 for e in simulation['updates'])})
                for name in ('parameters','trace','updates','state'):checked(saved[name+'File'],saved[name+'Sha256'])
                if split=='test' and h!=1:
                    out=export_stream(args.output,stream,simulation['parameters'],variant,source_file,
                        {**source,'finetuned':entry['finetuned']},'finetuned',saved,replay,{'config':asdict(config)},simulation)
                    out.update({'adaptationContract':CONTRACT,'adaptationSelectionFile':str(selection_file),
                        'adaptationSelectionSha256':sha(selection_file),'inputCalibration':config.input_enabled,
                        'jointEndToEndTraining':config.input_enabled,'updateCadenceSeconds':config.update_seconds})
                    write_json(Path(out['forecastFile']).with_suffix('.manifest.json'),out)
                    checked(out['forecastFile'],out['forecastSha256']);checked(Path(out['forecastFile']).with_suffix('.manifest.json'))
                    exports.append(out)
                write_json(artifact,metrics);checked(artifact);row['models'][variant]=metrics
                emit({'event':'scored','split':split,'horizonSeconds':h,'variant':variant,
                    'logMseSkill':metrics['logReturns']['allTickDeployedMseSkillVsZero'],
                    'logNll':metrics['logReturnNegativeLogLikelihood'],'seconds':metrics['seconds']})
                del calibrator,adapter,simulation
            results.append(row);write_json(args.output/f'{split}-progress.json',{'results':[r for r in results if r['split']==split]})
            del base,features,raw_tensor;torch.cuda.empty_cache()
    write_json(args.output/'exports.json',{'manifests':exports});checked(args.output/'exports.json')
    for file,digest in hashes.items():
        if sha(Path(file))!=digest:raise ValueError('artifact changed during experiment: '+file)
    result={'contract':CONTRACT,'planFile':str(plan_file),'planSha256':sha(plan_file),
        'selectionFile':str(selection_file),'selectionSha256':sha(selection_file),'baselineFile':str(args.baseline),
        'testUsedForSelection':False,'results':results,'artifactHashes':hashes,'seconds':time.monotonic()-started}
    write_json(args.output/'evaluation.json',result)
    emit({'event':'complete','seconds':result['seconds'],'evaluationSha256':sha(args.output/'evaluation.json')})


if __name__=='__main__':main()
