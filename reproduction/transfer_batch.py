"""Execute the fixed held-out graph protocol; accuracy jobs may overlap."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, json, hashlib, subprocess, sys, time
from transfer_study import PROTOCOL, graph_pair, forbid_label_loading

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    out=ROOT/'transfer_results'
    out.mkdir(exist_ok=True)
    logs=out/'job_logs'
    logs.mkdir(exist_ok=True)
    for ds in ['acm','dblp']:
        with forbid_label_loading():
            _,_,manifest=graph_pair(ds)
        (out/(ds+'_graph_manifest.json')).write_text(json.dumps(manifest,indent=2))
    jobs=[]
    for ds in ['acm','dblp']:
        for variant in ['full','heco','mp_aug']:
            for mode in ['source_transfer','target_only_reference']:
                for seed in [1,2,3]:
                    jobs.append(dict(dataset=ds,variant=variant,mode=mode,seed=seed))
        for mode in ['random_encoder','raw_features']:
            for seed in [1,2,3]:
                jobs.append(dict(dataset=ds,variant='full',mode=mode,seed=seed))
    for spec in jobs:
        spec['key']=f"{PROTOCOL}__{spec['dataset']}__{spec['variant']}__{spec['mode']}__seed{spec['seed']}"
    plan=dict(protocol=PROTOCOL,status='running',jobs=jobs,trained_encoder_count=36,
              untrained_encoder_controls=6,raw_feature_probe_groups=6,workers=args.workers,
              encoder_seeds=[1,2,3],probe_initializations_per_encoder=5,
              checkpoint='minimum SSL loss; source-transfer never uses target SSL loss or labels',
              hyperparameter_selection='none; fixed dataset base learning rate and registered graph-local positive ranking',
              execution='independent accuracy jobs can overlap positive study; not latency evidence',
              script_sha256={f:sha(ROOT/f) for f in ['transfer_study.py','transfer_batch.py','transfer_protocol.txt']})
    def save():
        p=out/'study_plan.json'; tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(plan,indent=2));tmp.replace(p)
    save()
    def run(spec):
        target=out/(spec['key']+'.json')
        if target.exists():
            existing=json.loads(target.read_text())
            expected_manifest=json.loads((out/(spec['dataset']+'_graph_manifest.json')).read_text())
            assert existing['protocol']==PROTOCOL and existing['split_hash']==expected_manifest['split_hash']
            assert existing['args']['epochs']==600 and existing['args']['probes']==5
            print('TRANSFER_REUSE',spec['key'],flush=True)
            return spec['key']
        command=[sys.executable,'-u',str(ROOT/'transfer_study.py')]
        for k in ['dataset','variant','mode','seed']:
            command.extend(['--'+k,str(spec[k])])
        print('TRANSFER_START',spec['key'],flush=True)
        with (logs/(spec['key']+'.log')).open('w') as f:
            subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('TRANSFER_FINISH',spec['key'],flush=True)
        return spec['key']
    t=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        completed=list(executor.map(run,jobs))
    plan.update(status='complete',completed_keys=completed,wall_seconds=time.perf_counter()-t)
    save()
    hashes={p.name:sha(p) for p in sorted(out.glob('*.json'))}
    (out/'evidence_sha256.json').write_text(json.dumps(dict(protocol=PROTOCOL,files=hashes),indent=2))
    (out/'STUDY_COMPLETE').write_text('complete\n')
    print('TRANSFER_STUDY_COMPLETE',len(completed),'groups; 36 trained encoders',flush=True)

if __name__=='__main__':
    main()
