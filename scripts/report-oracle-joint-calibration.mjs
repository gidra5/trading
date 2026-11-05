/** Check the causal-adapter exports, common tape and complete decision coverage. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
const root=process.cwd(), oi=process.argv.indexOf('--output');
const output=path.resolve(root,oi<0?'data/backtests/oracle-joint-calibration':process.argv[oi+1]);
const json=async(file)=>JSON.parse(await readFile(path.resolve(root,file),'utf8'));
const sha=(x)=>createHash('sha256').update(x).digest('hex');
const fileSha=async(file)=>sha(await readFile(path.resolve(root,file)));
const lines=(x)=>x.split(/\r?\n/).filter(x=>x.trim());
const plan=await json(path.join(output,'replay-plan.json'));
const selected=await json(path.join(output,'selection.json'));
const evaluation=await json(path.join(output,'evaluation.json'));
assert.equal(plan.contract,'oracle-joint-calibration-comparison-v1');
assert.equal(selected.contract,'causal-joint-input-output-log-hurdle-v1');
assert.equal(selected.testLoaded,false);assert.equal(selected.testUsedForSelection,false);
assert.equal(selected.allSettingsFrozen,true);assert.equal(evaluation.testUsedForSelection,false);
assert.equal(plan.riskAversion,0);assert.deepEqual(plan.horizonsSeconds,[900,1800,7200]);
assert.equal(await fileSha(selected.planFile),selected.planSha256);
assert.equal(await fileSha(evaluation.selectionFile),evaluation.selectionSha256);
assert.equal(await fileSha(selected.baseSelectionFile),selected.baseSelectionSha256);
const source=await json(selected.baseSelectionFile);
for(const [file,digest] of Object.entries(evaluation.artifactHashes))assert.equal(await fileSha(file),digest);
const previous=await json(path.join(path.dirname(evaluation.baselineFile),'comparison.json'));
let commonTapeSha256;const horizons=[],certificates=[];
for(const h of plan.horizonsSeconds){
  const count=plan.decisionCountsByHorizon[h],entry=source.horizons.find(r=>r.horizonSeconds===h);
  assert.equal(await fileSha(entry.finetuned.checkpoint),entry.finetuned.sha256);
  const variants=[];
  for(const variant of plan.variants){
    const file=path.join(output,`${variant.name}-h${h}.jsonl`);
    const manifest=await json(file.replace(/\.jsonl$/,'.manifest.json'));
    const config=selected.configs[h][variant.name];
    assert.equal(manifest.adaptationContract,selected.contract);
    assert.equal(manifest.adaptationSelectionSha256,evaluation.selectionSha256);
    assert.equal(manifest.selectionSha256,selected.baseSelectionSha256);
    assert.equal(manifest.checkpointSha256,entry.finetuned.sha256);
    assert.deepEqual(manifest.frozenCalibration.config,config);
    assert.equal(manifest.inputCalibration,config.input_enabled);
    assert.equal(manifest.jointEndToEndTraining,config.input_enabled);
    assert.equal(manifest.calibrationShapeUpdates,false);
    assert.equal(manifest.testUsedForSelection,false);
    assert.equal(manifest.onlineLabelUpdates,true);
    assert.equal(manifest.headSelectionsFrozenBeforeTest,true);
    assert.equal(manifest.calibrationSettingsFrozenBeforeTest,true);
    for(const key of ['parameters','trace','updates','state'])
      assert.equal(await fileSha(manifest.savedStream[key+'File']),manifest.savedStream[key+'Sha256']);
    const forecastText=await readFile(file,'utf8'),forecastSha=sha(forecastText),records=lines(forecastText);
    assert.equal(manifest.forecastSha256,forecastSha);assert.equal(records.length,count);
    assert.equal(manifest.firstTimestamp,plan.firstTimestamp);
    assert.equal(manifest.lastRealizedTimestamp,plan.terminalTimestamp);
    const tape=[];
    for(const [i,line] of records.entries()){
      const row=JSON.parse(line),stamp=plan.firstTimestamp+i*h*1000;
      assert.equal(row.timestamp,stamp);assert.equal(row.horizonSeconds,h);
      assert.equal(row.realizedNextTimestamp,stamp+h*1000);
      assert.equal(row.realizedPrices.length,h);assert.equal(row.realizedNextPrice,row.realizedPrices.at(-1));
      assert.equal(manifest.replayCanonicalIndices[i],i*h/4);
      const state=manifest.replayStateSamples[i];
      assert.equal(state.timestamp,stamp);
      assert(state.latestMaturityUsed===null||state.latestMaturityUsed<=stamp);
      // Information is incorporated on the fixed minute clock, independently
      // of which forecast origins are queried by the trading replay.
      const boundary=plan.firstTimestamp+Math.floor((stamp-plan.firstTimestamp)/(config.update_seconds*1000))*config.update_seconds*1000;
      const matured=Math.max(0,Math.floor((boundary-plan.firstTimestamp-h*1000)/4000)+1);
      assert.equal(state.maturedObservations,matured);
      const bounds=[2,2,Math.log(2),Math.log(2)];
      state.coefficients.forEach((v,k)=>{assert(Number.isFinite(v)&&Math.abs(v)<=bounds[k]);if(i===0)assert.equal(Math.abs(v),0);});
      if(i===0)tape.push(row.observedPrice);else assert.equal(row.observedPrice,tape.at(-1));
      for(const price of row.realizedPrices){assert(Number.isFinite(price)&&price>0);tape.push(price);}
      const children=row.tree.children;
      assert.equal(children.length,3);
      assert(children.every(c=>Number.isFinite(c.price)&&c.price>0&&Number.isFinite(c.probability)&&c.probability>=0));
      assert(Math.abs(children.reduce((s,c)=>s+c.probability,0)-1)<1e-12);
      assert(children[0].price<=row.observedPrice&&children[1].price===row.observedPrice&&children[2].price>=row.observedPrice);
    }
    assert.equal(tape.length,plan.comparisonObservedSeconds+1);
    const tapeSha=sha(JSON.stringify(tape));
    if(commonTapeSha256)assert.equal(tapeSha,commonTapeSha256);
    commonTapeSha256=tapeSha;assert.equal(tapeSha,previous.commonTapeSha256);
    const replays=[];
    for(const fee of variant.feeScales){
      const dir=path.join(output,`${variant.name}-h${h}-fees${fee}`),report=await json(path.join(dir,'report.json'));
      assert.equal(path.resolve(report.input),file);assert.equal(report.forecastSha256,forecastSha);
      assert.equal(report.contract,'oracle-long-cash-endpoint-mpc-v1');assert.equal(report.gamma,0);
      assert.equal(report.feeScale,String(fee));assert.equal(report.horizonSeconds,h);
      assert.equal(report.firstTimestamp,plan.firstTimestamp);assert.equal(report.terminalTimestamp,plan.terminalTimestamp);
      assert.equal(report.decisionCount,count);assert.equal(report.observedSeconds,plan.comparisonObservedSeconds);
      const decisions=lines(await readFile(path.join(dir,'decisions.jsonl'),'utf8'));
      assert.equal(decisions.length,count);assert.equal(report.decisionCertificates,count);
      certificates.push(...decisions);replays.push(report);
    }
    variants.push({name:variant.name,manifest,forecastSha256:forecastSha,replays});
  }
  horizons.push({horizonSeconds:h,variants});
}
assert.equal(certificates.length,1092);
const text=certificates.join('\n')+'\n';
await writeFile(path.join(output,'decisions.jsonl'),text);
await writeFile(path.join(output,'comparison.json'),JSON.stringify({plan,horizons,commonTapeSha256,
  artifactChecksPassed:true,certificateRecords:certificates.length,certificatesSha256:sha(text),
  scope:'Same supplied scenario/menu controller and tape; Lean checks decision certificates, not learned forecast quality or unrestricted optimality.'},null,2)+'\n');
console.log(JSON.stringify({artifactChecksPassed:true,certificateRecords:certificates.length}));
