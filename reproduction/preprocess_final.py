"""Audit and measure the actual available preprocessing boundary."""
import time,json,importlib.util,types,copy
import numpy as np
import torch
from pathlib import Path
from experiment import ROOT,SRC,CONFIG,load_dataset,normalized_adj,seed_all,Model
import scipy.sparse as sp
def relation(p,name,shape):
 e=np.loadtxt(p/(name+'.txt'),dtype=np.int64)
 return sp.csr_matrix((np.ones(len(e),dtype=bool),(e[:,0],e[:,1])),shape=shape)
def build(ds):
 p=SRC/'data'/ds
 if ds=='acm':
  pa=relation(p,'pa',(4019,7167));ps=relation(p,'ps',(4019,60));adjs=[pa@pa.T,ps@ps.T]
 elif ds=='dblp':
  pa=relation(p,'pa',(14328,4057));pc=relation(p,'pc',(14328,20));pt=relation(p,'pt',(14328,7723));apc=pa.T@pc;apt=pa.T@pt
  adjs=[pa.T@pa,apc@apc.T,apt@apt.T]
 elif ds=='aminer':
  pa=relation(p,'pa',(6564,13329));pr=relation(p,'pr',(6564,35890));adjs=[pa@pa.T,pr@pr.T]
 else:
  md=relation(p,'md',(3492,2502));ma=relation(p,'ma',(3492,33401));mw=relation(p,'mw',(3492,4459));adjs=[md@md.T,ma@ma.T,mw@mw.T]
 return [a.astype('float32').tocsr() for a in adjs]
seed_all(20261010);out=[]
for ds in CONFIG:
 build(ds);construct=[];norm=[];loaded=[]
 for _ in range(5):
  t=time.perf_counter();a=build(ds);construct.append(time.perf_counter()-t)
  t=time.perf_counter();aa=[normalized_adj(v) for v in a];norm.append(time.perf_counter()-t)
 checks=[{'path':name,'nnz':v.nnz,'different_entries':(v!=sp.load_npz(SRC/'data'/ds/(name+'.npz'))).nnz} for name,v in zip(CONFIG[ds]['mps'],a)]
 for _ in range(5):
  torch.cuda.synchronize();t=time.perf_counter();data=load_dataset(ds);torch.cuda.synchronize();loaded.append(time.perf_counter()-t)
  del data;torch.cuda.empty_cache()
 out.append(dict(dataset=ds,construction_median_seconds=float(np.median(construct)),normalization_median_seconds=float(np.median(norm)),released_loading_median_seconds=float(np.median(loaded)),construction_repeats=construct,normalization_repeats=norm,loading_repeats=loaded,metapath_checks=checks,upstream_positive_generation_cost='unavailable: released positive files reused; no SimRank generator supplied'))
 print('PREPROCESS',ds,out[-1],flush=True)
(ROOT/'results/preprocessing_final.json').write_text(json.dumps(out,indent=2))
inference_equivalence={}
for ds in CONFIG:
 data=load_dataset(ds);model=Model(data).cuda().eval()
 with torch.no_grad():
  legacy=model.mp(model.project(data['feats'],False)[0],data['mps'])
  repeated=model.mp(model.project(data['feats'],False)[0],data['mps'])
  retained=model.embeds(data)
  difference=(legacy-retained).abs().max().item()
  repeat_difference=(legacy-repeated).abs().max().item()
 assert difference<1e-5
 inference_equivalence[ds]={'retained_max_abs_difference':difference,
  'repeated_legacy_max_abs_difference':repeat_difference,
  'tolerance':1e-5,'note':'CUDA sparse aggregation can vary at floating-point accumulation precision even for repeated unchanged forwards.'}
 del data,model,legacy,repeated,retained
 torch.cuda.empty_cache()
(ROOT/'results/inference_numerical_equivalence.json').write_text(json.dumps(inference_equivalence,indent=2))
# Compare the original AdaMEOW scorer with broadcast and exact row-block forms
# on the same small tensors, including encoder-input and scorer gradients.
from adameow_runner import prepare,exact_weighted_loss
prepare();from ada_impl.adameow import AdaMEOW
from adameow_runner import pathsim_topk,reference_normalize
from ada_impl.preprocess import pathsim,normalize_adj
test=sp.csr_matrix(np.array([[4,2,1,1],[2,7,1,0],[1,1,3,1],[1,0,1,5]],dtype='float64'))
original=pathsim([test],[2])[0].tocsr();adapted=pathsim_topk(test,2)
assert (original!=adapted).nnz==0
assert (original-original.T).nnz>0 # Exercise directed top-K selection.
normalization_difference=(torch.as_tensor(reference_normalize(adapted).toarray(),device='cuda').float()-normalize_adj(original)).abs().max().item()
assert normalization_difference<1e-6
def check_form(block):
 torch.manual_seed(987);a=torch.randn(17,9,device='cuda',requires_grad=True);b=torch.randn(17,9,device='cuda',requires_grad=True)
 dummy=types.SimpleNamespace(tau=.9,MLP=torch.nn.Sequential(torch.nn.Linear(9,16),torch.nn.Tanh(),torch.nn.Linear(16,1),torch.nn.Sigmoid()).cuda(),loss_block=block,training=True)
 loss=exact_weighted_loss(dummy,a,b) if block is not None else AdaMEOW.weight_InfoNce(dummy,a,b)
 loss.backward();return loss.item(),[a.grad,b.grad]+[p.grad for p in dummy.MLP.parameters()]
reference=check_form(None);result={}
for block in [0,5]:
 current=check_form(block);result[str(block)]={'loss_abs_difference':abs(reference[0]-current[0]),'gradient_max_abs_difference':max((a-b).abs().max().item() for a,b in zip(reference[1],current[1]))}
 assert result[str(block)]['loss_abs_difference']<1e-5 and result[str(block)]['gradient_max_abs_difference']<1e-5
result['pathsim_different_entries']=int((original!=adapted).nnz)
result['normalization_max_abs_difference']=normalization_difference
(ROOT/'results/adameow_numerical_equivalence.json').write_text(json.dumps(result,indent=2));print('ADAMEOW_EQUIVALENCE',result,flush=True)
