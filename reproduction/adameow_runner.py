"""AdaMEOW upstream architecture with common features and unified evaluation.

Only execution adapters are changed: identity features stay sparse, the
pairwise MLP is evaluated in exact recomputed row blocks, and checkpoints and
probes use the same label-isolated protocol as TriView-HCL.
"""
import sys, time, json, random, types, argparse, shutil, importlib.util
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from experiment import ROOT,SRC,load_dataset,sparse_tensor,normalized_adj,seed_all,probe,cluster

def prepare():
 src=ROOT/'baselines/MEOW-main/code_adameow/module';out=ROOT/'ada_impl';out.mkdir(exist_ok=True)
 (out/'__init__.py').write_text('')
 for f in src.glob('*.py'):
  if f.name=='__init__.py':continue
  text=f.read_text()
  if f.name=='adameow.py':
   text=text.replace('torch.stack(adjs).mean(axis=0)','sum(adjs[1:], adjs[0]) / len(adjs)')
   text=text.replace('i.to_dense().sum(axis=1).unsqueeze(axis=1).reshape(adjs[0].size(0),1)','torch.sparse.sum(i,dim=1).to_dense().reshape(adjs[0].size(0),1)')
  (out/f.name).write_text(text)

def row_normalize(x):
 x=x.tocsr().astype('float32');degree=np.asarray(x.sum(1)).ravel();inv=np.zeros_like(degree);ok=degree!=0;inv[ok]=1/degree[ok]
 return sp.diags(inv)@x

def reference_normalize(x):
 # Upstream normalizes with row degrees, then transposes the selected graph.
 # Selection can be directed; omitting this transpose changes the method.
 x=x.astype('float64').tocsr();degree=np.asarray(x.sum(1)).ravel()
 inv=np.zeros_like(degree);ok=degree>0;inv[ok]=degree[ok]**-.5
 return sp.diags(inv)@x.T@sp.diags(inv)

def pathsim_topk(a,k):
 # The official implementation ranks with PathSim, then keeps the ORIGINAL
 # meta-path multiplicities as edge weights (not PathSim scores).
 a=a.astype('float64').tocsr();a.eliminate_zeros();diag=a.diagonal();rows=np.repeat(np.arange(a.shape[0]),np.diff(a.indptr));den=diag[rows]+diag[a.indices]
 values=np.divide(2*a.data,den,out=np.zeros_like(a.data),where=den>0)
 ii=[];jj=[];vv=[]
 for i in range(a.shape[0]):
  lo,hi=a.indptr[i:i+2];ranking=np.zeros(a.shape[1],dtype='float64');ranking[a.indices[lo:hi]]=values[lo:hi]
  # Row-wise dense sorting exactly matches the upstream NumPy tie policy,
  # without materializing the complete N x N dense matrix.
  selected=np.argsort(ranking)[::-1][:k];selected=np.sort(selected)
  weights=a[i,selected].toarray().ravel();keep=weights!=0;ids=selected[keep];vs=weights[keep].astype('int32')
  ii.extend([i]*len(ids));jj.extend(ids);vv.extend(vs)
 return sp.csr_matrix((vv,(ii,jj)),shape=a.shape)

def mask_sparse(x,rate):
 x=x.coalesce();v=x.values().clone();count=int(rate*len(v));idx=torch.randperm(len(v),device=v.device)[:count];v[idx]=0
 return torch.sparse_coo_tensor(x.indices(),v,x.shape,device=v.device).coalesce()

def exact_weighted_loss(self,a,b):
 if getattr(self,'representation_only',False):return torch.zeros((),device=a.device)
 n=len(a);block=getattr(self,'loss_block',0) or n
 def part(aa,bb,lo):
  dots=torch.exp(aa@bb.T/self.tau)
  weight=self.MLP((aa[:,None,:]+bb[None,:,:]).reshape(-1,aa.shape[-1])).reshape(len(aa),n)
  numerator=dots[torch.arange(len(aa),device=a.device),torch.arange(lo,lo+len(aa),device=a.device)]
  return (-torch.log(numerator/(dots*weight).sum(1))).sum()
 total=0
 for lo in range(0,n,block):
  aa=a[lo:lo+block]
  if self.training and self.loss_block:total=total+checkpoint(lambda x,y,off=lo:part(x,y,off),aa,b,use_reentrant=False)
  else:total=total+part(aa,b,lo)
 return total/n

def run(args):
 seed_all(args.seed);t=time.perf_counter();data=load_dataset(args.dataset);p=ROOT/'baselines/MEOW-main/data'/args.dataset
 # The published dataset package has exactly the same labels and public splits.
 assert np.array_equal(data['y'],np.load(p/'labels.npy').reshape(-1).astype('int64'))
 cfgs={'acm':dict(k=[690,690],lr=.00005,wd=1e-5,drop=.3,tau=1.,fm=.3,em=.2),
       'dblp':dict(k=[25,200,40],lr=.0001,wd=.001,drop=.2,tau=.9,fm=.1,em=.3),
       'aminer':dict(k=[8,31],lr=.0001,wd=.0001,drop=0.,tau=.9,fm=0.,em=0.)}
 cfg=cfgs[args.dataset]
 feats=[]
 if args.dataset=='dblp':
  dims=[data['dims'][0],14328,7723,20]
  feats=[data['feats'][0]]+[sparse_tensor(sp.eye(n,format='csr')) for n in dims[1:]]
  names=['nei_ap','nei_apc','nei_apcp','nei_apt','nei_aptp']
 else:
  dims=data['dims'];feats=[data['feats'][i] if data['feats'][i] is not None else sparse_tensor(sp.eye(n,format='csr')) for i,n in enumerate(dims)]
  names=['nei_a','nei_s'] if args.dataset=='acm' else ['nei_a','nei_r']
 neis=[sparse_tensor(sp.load_npz(p/(s+'.npz'))) for s in names]
 reference_paths=[sp.load_npz(p/(name+'.npz')) for name in data['cfg']['mps']]
 for supplied,reference in zip(data['raw'],reference_paths):
  assert (supplied.astype(bool)!=reference.astype(bool)).nnz==0
 raw=[pathsim_topk(a,k) for a,k in zip(reference_paths,cfg['k'])];adjs=[sparse_tensor(reference_normalize(a)) for a in raw]
 masked=[]
 for a in raw:
  a=a.tocoo();keep=a.row!=a.col;rr=a.row[keep];cc=a.col[keep];pick=np.random.permutation(len(rr))[int(cfg['em']*len(rr)):]
  aa=sp.csr_matrix((np.ones(len(pick)),(rr[pick],cc[pick])),shape=a.shape)+sp.eye(a.shape[0])
  masked.append(sparse_tensor(reference_normalize(aa)))
 target=feats[0];mf=mask_sparse(target,cfg['fm'])
 prepare();from ada_impl.adameow import AdaMEOW
 model=AdaMEOW(dims,len(adjs),64,64,cfg['tau'],adjs,cfg['drop'],neis,target,args.dataset)
 if args.dataset=='aminer':
  # Upstream uses per-meta-path positional features of identical width. In the
  # common-feature experiment each path instead receives the same node identity.
  feats[0]=[target]*len(adjs);mf=[mf]*len(adjs)
  model.fc_list=nn.ModuleList([nn.Linear(d,64) for d in dims[1:]])
 model.weight_InfoNce=types.MethodType(exact_weighted_loss,model);model.loss_block=args.block;model=model.cuda()
 pre=time.perf_counter()-t;opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']*args.lrscale,weight_decay=cfg['wd'])
 key=f'{args.dataset}__adameow__common__lr{args.lrscale}__seed{args.seed}';path=ROOT/'results'/(key+'.json')
 if path.exists() and not args.force:print('EXISTS',key);return
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();best=float('inf');state=None;wait=0;history=[]
 for epoch in range(args.epochs):
  model.train();opt.zero_grad();loss=model(feats,mf,adjs,masked);lv=loss.item();history.append(lv)
  if not torch.isfinite(loss):raise ValueError('Nonfinite AdaMEOW loss')
  if lv<best:
   best=lv;wait=0;bestepoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else:wait+=1
  loss.backward();opt.step()
  if epoch%50==0:print(key,epoch,lv,flush=True)
  if epoch>=100 and wait>=40:break
 torch.cuda.synchronize();train=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()/2**20
 model.load_state_dict(state);model.eval();model.representation_only=True;t=time.perf_counter()
 with torch.no_grad():model(feats,mf,adjs,masked);z=model.get_embeds()
 torch.cuda.synchronize();infer=time.perf_counter()-t
 classification={k:probe(z,data['y'],s,args.seed,args.probes,200) for k,s in data['splits'].items()};clustering=cluster(z,data['y'],args.seed)
 result=dict(key=key,args=vars(args),config=cfg,classification=classification,clustering=clustering,epochs=epoch+1,best_epoch=bestepoch,best_loss=best,preprocess_seconds=pre,train_seconds=train,inference_seconds=infer,peak_training_mib=peak,parameters=sum(p.numel() for p in model.parameters()),history=history,feature_protocol='common features; node identities replace native auxiliary attributes and AMiner metapath2vec features; native count-preserving PathSim and transposed normalization',feature_protocol_version='native_pathsim_common_features_v3',environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda))
 path.write_text(json.dumps(result,indent=2));np.savez_compressed(path.with_suffix('.npz'),embedding=z.cpu().numpy());print('DONE',key,classification[20]['mean'],clustering['mean'],train,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['acm','dblp','aminer'],default='acm');p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=600);p.add_argument('--probes',type=int,default=5);p.add_argument('--lrscale',type=float,default=1.);p.add_argument('--block',type=int,default=0);p.add_argument('--force',action='store_true');run(p.parse_args())
