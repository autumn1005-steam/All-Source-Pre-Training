"""Dense/exact-block equivalence, full-model scaling, and pairwise kernels."""
import json,time,copy,gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from experiment import ROOT,load_dataset,Model,pos_indices,sparse_tensor,normalized_adj,MassContrast,BlockContraNorm,seed_all
from extended_controls import run_extended
run_extended()
def induced(base,n):
 data=dict(base);cfg=copy.deepcopy(base['cfg']);cfg['types'][0]=n;data['cfg']=cfg
 data['y']=base['y'][:n];data['raw']=[a[:n,:n] for a in base['raw']];data['mps']=[sparse_tensor(normalized_adj(a)) for a in data['raw']]
 data['feats']=[None]*len(base['feats']);data['dims']=[n]
 new=[]
 for t,nei in enumerate(base['neis']):
  used=np.unique(np.concatenate([np.asarray(v) for v in nei[:n]]));mapping=np.full(base['cfg']['types'][t+1],-1,dtype='int64');mapping[used]=np.arange(len(used))
  new.append([mapping[np.asarray(v)] for v in nei[:n]]);cfg['types'][t+1]=len(used);data['dims'].append(len(used))
 data['neis']=new;data['pos']=base['pos'][:n,:n].tocsr();return data
def measured(fn,warmup=2,repeats=5):
 for _ in range(warmup):fn()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
 for _ in range(repeats):
  torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();times.append(time.perf_counter()-t)
 return dict(median_seconds=float(np.median(times)),individual_seconds=times,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
seed_all(20261010);benchmark=[]
plan=json.loads((ROOT/'study_plan.json').read_text())
for ds in ['acm','dblp','aminer','freebase']:
 data=load_dataset(ds);idx,mask=pos_indices(data['pos'])
 for variant in ['full','heco']:
  seed_all(0);model=Model(data,variant=variant).cuda();opt=torch.optim.Adam(model.parameters(),lr=data['cfg']['lr']*plan['primary'][ds][variant]['lrscale'])
  def trainstep():
   model.train();opt.zero_grad(set_to_none=True);loss=model(data,idx,mask);loss.backward();opt.step()
  result=measured(trainstep)
  model.eval()
  def inference():
   with torch.no_grad():return model.embeds(data)
  retained=sum(p.numel() for p in model.fc[0].parameters())+sum(p.numel() for p in model.mp.parameters())
  if variant=='heco':active=sum(p.numel() for m in [model.fc,model.mp,model.ng,model.contrast] for p in m.parameters())
  else:active=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in model.ln.parameters())
  result.update(dataset=ds,variant=variant,n=len(data['y']),inference=measured(inference,repeats=10),active_training_parameters=active,retained_inference_parameters=retained)
  benchmark.append(result);print('BENCHMARK',ds,variant,result['median_seconds'],result['peak_allocated_mib'],flush=True)
  del opt,model;gc.collect();torch.cuda.empty_cache()
 del data,idx,mask;gc.collect();torch.cuda.empty_cache()
base=load_dataset('aminer');model_scaling=[]
for n in [512,1024,2048,4096,6564]:
 data=induced(base,n);idx,mask=pos_indices(data['pos'])
 for block in [0,256]:
  seed_all(0);model=Model(data,block=block).cuda();opt=torch.optim.Adam(model.parameters(),lr=.003)
  def trainstep():
   model.train();opt.zero_grad(set_to_none=True);loss=model(data,idx,mask);loss.backward();opt.step()
  result=measured(trainstep)
  model.eval()
  def inference():
   with torch.no_grad():return model.embeds(data)
  result.update(dataset='AMiner induced target prefixes',n=n,encoded_nodes=sum(data['cfg']['types']),block=block,inference=measured(inference,repeats=10),parameters=sum(p.numel() for p in model.parameters()))
  model_scaling.append(result);print('MODEL',n,block,result['median_seconds'],result['peak_allocated_mib'],flush=True)
  del opt,model;gc.collect();torch.cuda.empty_cache()
 del data,idx,mask;gc.collect();torch.cuda.empty_cache()
del base;gc.collect();torch.cuda.empty_cache()
kernel_scaling=[]
for n in [2048,4096,8192,16384,32768]:
 for block in [0,256]:
  if n==32768 and block==0:continue
  seed_all(0);a=torch.randn(n,64,device='cuda',requires_grad=True);b=torch.randn(n,64,device='cuda',requires_grad=True)
  idx=torch.stack([(torch.arange(n,device='cuda')+k)%n for k in range(9)],1);mask=torch.ones_like(idx,dtype=torch.bool)
  contrast=MassContrast(64,.8,block).cuda();cn=BlockContraNorm(64,block).cuda()
  def step():
   a.grad=None;b.grad=None;contrast.zero_grad(set_to_none=True);cn.zero_grad(set_to_none=True)
   loss=contrast.pair(a,cn(b),idx,mask);loss.backward()
  try:result=measured(step,warmup=1,repeats=3);result.update(n=n,block=block,kind='synthetic pairwise objective plus ContraNorm kernels; no graph benchmark');kernel_scaling.append(result);print('KERNEL',n,block,result['median_seconds'],result['peak_allocated_mib'],flush=True)
  except torch.cuda.OutOfMemoryError:kernel_scaling.append(dict(n=n,block=block,out_of_memory=True));print('KERNEL_OOM',n,block,flush=True)
  del a,b,idx,mask,contrast,cn;gc.collect();torch.cuda.empty_cache()
(ROOT/'results/efficiency.json').write_text(json.dumps(dict(benchmark=benchmark,model_scaling=model_scaling,kernel_scaling=kernel_scaling,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),indent=2));print('EFFICIENCY_COMPLETE',flush=True)
