"""Independent per-horizon checkpoint, initializer and target provenance audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_neural_sign_size_forecast import HORIZONS,examples,select_origins,sha,write_json
from stochastic_price_forecast import load_timeline_split
from train_per_horizon_log_hurdle import CONTRACT,load_combined,verified
from train_ternary_sign_size_forecast import singleton_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('data/training/runs/stochastic-crra-per-horizon-log-v1'))
    parser.add_argument('--output',type=Path,default=Path('data/backtests/oracle-per-horizon-log/training-diagnostics.json'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    selected=json.loads((args.root/'selection.json').read_text())
    plan=json.loads((args.root/'plan.json').read_text())
    metadata=json.loads((args.root/'metadata.json').read_text())
    result=json.loads((args.root/'training-result.json').read_text())
    if selected['contract']!=CONTRACT or selected['plan']!=plan or sha(args.root/'plan.json')!=selected['planSha256'] \
            or result['selectionSha256']!=sha(args.root/'selection.json') or selected['testLoaded'] or result['testLoaded'] \
            or selected['testUsedForSelection'] or not selected['allHeadSelectionsFrozen'] or selected['expertBackbones']!=12:
        raise ValueError('frozen independent-horizon provenance mismatch')
    for path,digest in ((selected['sourceSelectionFile'],selected['sourceSelectionSha256']),
            (selected['onlineSelectionFile'],selected['onlineSelectionSha256'])):
        if sha(Path(path))!=digest: raise ValueError('retained selection changed')
    source=json.loads(Path(selected['sourceSelectionFile']).read_text())
    dataset=Path(plan['dataset'])
    if sha(dataset/'manifest.json')!=plan['datasetManifestSha256']: raise ValueError('dataset changed')
    populations={}
    for split in ('train','validation'):
        timeline=load_timeline_split(dataset,split)
        origins=select_origins(timeline,4,256000 if split=='train' else None)
        data=examples(timeline,origins)
        y=np.log1p(data.targets)
        np.testing.assert_array_equal(np.sign(y),np.sign(data.targets))
        populations[split]={'examples':len(y),'firstTimestamp':int(data.axis_start_ms+origins[0]*1000),
            'lastEndpointTimestamp':int(data.axis_start_ms+(origins[-1]+7200)*1000),
            'logTargetsSha256':hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()}
        if split=='train':
            mean=data.features.reshape(-1,59).mean(0,dtype=np.float64).astype(np.float32)
            std=np.maximum(data.features.reshape(-1,59).std(0,dtype=np.float64),1e-10).astype(np.float32)
            np.testing.assert_array_equal(mean,np.asarray(metadata['inputMean'],dtype=np.float32))
            np.testing.assert_array_equal(std,np.asarray(metadata['inputStd'],dtype=np.float32))
            for j,c in enumerate(selected['constants']):
                up,down=y[:,j]>0,y[:,j]<0
                np.testing.assert_equal(c['upMeanLog'],y[up,j].mean())
                np.testing.assert_equal(c['downMeanLog'],-y[down,j].mean())
                np.testing.assert_equal(c['upTrainingExamples'],up.sum())
                np.testing.assert_equal(c['downTrainingExamples'],down.sum())
    if populations['train']['examples']!=247901 or populations['validation']['examples']!=34560 \
            or populations['train']['lastEndpointTimestamp']>=populations['validation']['firstTimestamp']:
        raise ValueError('population or split-boundary mismatch')
    logs=[json.loads(s) for s in (args.root/'training.jsonl').read_text().splitlines()]
    size_epochs=[r for r in logs if r['task']=='size' and r['event']=='epoch']
    summaries=[];parameter_total=0
    for j,entry in enumerate(selected['horizons']):
        h=entry['horizonSeconds']
        if h!=HORIZONS[j]: raise ValueError('horizon order changed')
        size=entry['initializers']['size']
        best=min(size_epochs,key=lambda r:r['validationScores'][j])
        if size['epoch']!=best['epoch'] or size['validationScore']!=best['validationScores'][j]:
            raise ValueError('size selected globally instead of per horizon')
        saved_size=verified(size)
        if saved_size['targetCoordinate']!='log-return' or saved_size['contract']!=CONTRACT: raise ValueError('wrong size target')
        train_epochs=[r for r in logs if r['task']=='combined' and r['horizonsSeconds']==[h] and r['event']=='epoch']
        fine=entry['finetuned']
        winner=min(train_epochs,key=lambda r:r['validationScores'][0])
        if fine['epoch']!=winner['epoch'] or fine['validationScore']!=winner['validationScores'][0]:
            raise ValueError('fine checkpoint does not minimize its own horizon NLL')
        if fine['completedEpochs']!=train_epochs[-1]['epoch'] or fine['completedEpochs']<12 \
                or (fine['completedEpochs']<64 and fine['completedEpochs']-fine['epoch']<12):
            raise ValueError('registered independent stopping rule violated')
        pretrained=verified(entry['pretrained']); final=verified(fine)
        for saved in (pretrained,final):
            if saved['horizonsSeconds']!=[h] or saved['contract']!=CONTRACT or saved['planSha256']!=selected['planSha256']:
                raise ValueError('not a singleton-horizon checkpoint')
        for task,width in (('zero',1),('sign',1),('size',2)):
            record=entry['initializers'][task]
            if task!='size' and record!=source['horizons'][j]['pretrained'][task]: raise ValueError('original classifier initializer changed')
            initial=verified(record)
            expected=singleton_state(initial['modelState'],j,width)
            for name,value in expected.items():
                torch.testing.assert_close(pretrained['modelState'][task+'.'+name],value,rtol=0,atol=0)
        if fine['epoch']==0:
            for name,value in pretrained['modelState'].items():
                torch.testing.assert_close(final['modelState'][name],value,rtol=0,atol=0)
        # Retained final simple-return checkpoint also remains untouched.
        verified(source['horizons'][j]['finetuned'])
        model=load_combined(fine,torch.device('cpu'))
        parameters=sum(p.numel() for p in model.parameters())
        if parameters!=entry['parameters']: raise ValueError('parameter count changed')
        parameter_total+=parameters
        summaries.append({'horizonSeconds':h,'sizeEpoch':size['epoch'],'fineEpoch':fine['epoch'],
            'completedFineEpochs':fine['completedEpochs'],'parameters':parameters,
            'originalClassifierStatesPreserved':True,'sizeTargetCoordinate':'log-return'})
        del model
    if parameter_total!=selected['parameters'] or parameter_total!=69315656: raise ValueError('retained per-horizon architecture differs')
    output={'allChecksPassed':True,'testLoadedByAudit':False,'neuralInference':False,
        'selectionSha256':sha(args.root/'selection.json'),'trainingPlanSha256':selected['planSha256'],
        'populations':populations,'horizons':summaries,'parameters':parameter_total,
        'scope':'Independent target/statistic reconstruction; own-horizon minima, copied initializer states and original checkpoint hashes; no test or model inference'}
    write_json(args.output,output)
    print(json.dumps(output),flush=True)


if __name__=='__main__': main()
