"""Official NS4GC encoder/objective on the target-node projected graph."""
import sys,time,json,argparse,importlib.util
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from experiment import ROOT,load_dataset,sparse_tensor,seed_all,probe,cluster

def normalize(a):
 a=(a+sparse_tensor(sp.eye(a.shape[0],format='csr'))).coalesce()
 deg=torch.sparse.sum(a,dim=0).to_dense().clamp(min=1).pow(-.5)
 idx=a.indices();v=a.values()*deg[idx[0]]*deg[idx[1]]
 return torch.sparse_coo_tensor(idx,v,a.shape,device='cuda').coalesce()
def augment(a,x):
 # Same feature-column and independent directed-edge dropout as upstream.
 keep=torch.bernoulli(torch.full((a._nnz(),),.7)).bool().cuda();idx=a.indices()[:,keep];v=a.values()[keep]
 aa=torch.sparse_coo_tensor(idx,v,a.shape,device='cuda').coalesce()
 mask=torch.rand(x.shape[1],device='cuda')<.2
 v=x.values().clone();v[mask[x.indices()[1]]]=0
 xx=torch.sparse_coo_tensor(x.indices(),v,x.shape,device='cuda').coalesce()
 return normalize(aa),xx
def run(args):
 seed_all(args.seed);t=time.perf_counter();data=load_dataset(args.dataset);n=len(data['y'])
 a=sum(data['raw']).astype(bool).astype('float32').tocsr();a.setdiag(0);a.eliminate_zeros();a=sparse_tensor(a)
 x=data['feats'][0] if data['feats'][0] is not None else sparse_tensor(sp.eye(n,format='csr'))
 spec=importlib.util.spec_from_file_location('ns4_encoder',ROOT/'baselines/NS4GC-main/encoder.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 model=m.GCN(x.shape[1],[256,64]).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001*args.lrscale,weight_decay=1e-5)
 src,dst=a.indices();mask=torch.ones(n,n,device='cuda',dtype=torch.bool);mask[src,dst]=False;mask.fill_diagonal_(False)
 key=f'{args.dataset}__ns4gc__projected__lr{args.lrscale}__seed{args.seed}';path=ROOT/'results'/(key+'.json')
 if path.exists() and not args.force:print('EXISTS',key);return
 pre=time.perf_counter()-t;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();best=float('inf');state=None;wait=0;history=[]
 for epoch in range(args.epochs):
  model.train();a1,x1=augment(a,x);a2,x2=augment(a,x);z1,z2=model(a1,a2,x1,x2);s=z1@z2.T
  loss=-s.diagonal().mean()-s[src,dst].mean()+torch.sigmoid((s[mask]-.6)/.1).mean()
  lv=loss.item();history.append(lv)
  if lv<best:
   best=lv;bestepoch=epoch;wait=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else:wait+=1
  opt.zero_grad();loss.backward();opt.step()
  if epoch%50==0:print(key,epoch,lv,flush=True)
  if epoch>=100 and wait>=40:break
 torch.cuda.synchronize();train=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()/2**20;model.load_state_dict(state);model.eval();t=time.perf_counter()
 with torch.no_grad():z=model.embed(normalize(a),x)
 torch.cuda.synchronize();infer=time.perf_counter()-t
 classification={k:probe(z,data['y'],s,args.seed,args.probes,200) for k,s in data['splits'].items()};clustering=cluster(z,data['y'],args.seed)
 result=dict(key=key,args=vars(args),config=dict(lr=.001*args.lrscale,weight_decay=1e-5,layers=[256,64],edge_drop=.3,feature_drop=.2,s=.6,tau=.1,lam=1.,gamma=1.),classification=classification,clustering=clustering,epochs=epoch+1,best_epoch=bestepoch,best_loss=best,preprocess_seconds=pre,train_seconds=train,inference_seconds=infer,peak_training_mib=peak,parameters=sum(p.numel() for p in model.parameters()),history=history,feature_protocol='common target-node features and binary union of released meta-path graphs',environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda))
 path.write_text(json.dumps(result,indent=2));np.savez_compressed(path.with_suffix('.npz'),embedding=z.cpu().numpy());print('DONE',key,classification[20]['mean'],clustering['mean'],train,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='acm');p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=400);p.add_argument('--probes',type=int,default=5);p.add_argument('--lrscale',type=float,default=1.);p.add_argument('--force',action='store_true');run(p.parse_args())
