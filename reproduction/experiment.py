"""Reproducible, label-isolated experiments for the TriView-HCL revision.

Original modules are preserved. This harness corrects paths, avoids dense identity
features, uses one validation checkpoint for all probe metrics, and records every
encoder seed separately. Evaluation labels never enter encoder optimization.
"""
import os, sys, json, time, copy, random, argparse, importlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'original/TriView-HCL-main'
sys.path.insert(0, str(SRC / 'code'))
from module.mp_encoder import Mp_encoder
from module.ng_encoder import Ng_encoder
from module.sm_encoder import Sm_encoder,dropout_feat,add_gaussian_noise,edge_drop
from module.contra_norm import ContraNorm

CONFIG = {
 'acm': dict(types=[4019,7167,60],mps=['pap','psp'],neis=['nei_a','nei_s'],feat='p_feat',sample=[7,1],drop=.3,attn=.5,lr=.0008,tau=.8),
 'dblp': dict(types=[4057,14328],mps=['apa','apcpa','aptpa'],neis=['nei_p'],feat='a_feat',sample=[6],drop=.3,attn=.5,lr=.0008,tau=.9),
 'aminer': dict(types=[6564,13329,35890],mps=['pap','prp'],neis=['nei_a','nei_r'],feat=None,sample=[3,8],drop=.5,attn=.5,lr=.003,tau=.5),
 'freebase': dict(types=[3492,2502,33401,4459],mps=['mdm','mam','mwm'],neis=['nei_d','nei_a','nei_w'],feat=None,sample=[1,18,2],drop=.1,attn=.3,lr=.001,tau=.5),
}
def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 torch.backends.cuda.matmul.allow_tf32 = False
 torch.set_num_threads(4)

def sparse_tensor(x, device='cuda'):
 x=x.tocoo().astype('float32')
 return torch.sparse_coo_tensor(np.vstack([x.row,x.col]),x.data,x.shape,device=device).coalesce()

def normalized_adj(x):
 x=x.tocsr().astype('float32'); degree=np.asarray(x.sum(1)).ravel()
 inv=np.zeros_like(degree); ok=degree>0; inv[ok]=degree[ok]**-.5
 return sp.diags(inv)@x@sp.diags(inv)

def load_dataset(name):
 cfg=CONFIG[name]; p=SRC/'data'/name
 y=np.load(p/'labels.npy',allow_pickle=False).astype('int64').reshape(-1)
 raw=[sp.load_npz(p/(s+'.npz')).tocsr().astype('float32') for s in cfg['mps']]
 mps=[sparse_tensor(normalized_adj(a)) for a in raw]
 feats=[None]*len(cfg['types']); dims=list(cfg['types'])
 if cfg['feat']:
  x=sp.load_npz(p/(cfg['feat']+'.npz')).tocsr().astype('float32')
  degree=np.asarray(x.sum(1)).ravel(); inv=np.zeros_like(degree);ok=degree!=0;inv[ok]=1/degree[ok]
  x=sp.diags(inv)@x; feats[0]=sparse_tensor(x);dims[0]=x.shape[1]
 neis=[np.load(p/(s+'.npy'),allow_pickle=True) for s in cfg['neis']]
 # Archive contains ordinary NumPy object arrays; every member is checked as integer indices.
 for t,arr in enumerate(neis):
  for a in arr:
   a=np.asarray(a)
   if a.ndim!=1 or a.dtype.kind not in 'iu' or (a.size and (a.min()<0 or a.max()>=cfg['types'][t+1])):
    raise ValueError('Invalid neighbor indices')
 splits={k:{s:np.load(p/f'{s}_{k}.npy',allow_pickle=False).astype('int64') for s in ['train','val','test']} for k in [20,40,60]}
 for k,s in splits.items():
  for a,b in [('train','val'),('train','test'),('val','test')]:
   assert len(np.intersect1d(s[a],s[b]))==0
 pos=sp.load_npz(p/'pos.npz').tocsr()
 return dict(name=name,cfg=cfg,y=y,feats=feats,dims=dims,mps=mps,raw=raw,neis=neis,splits=splits,pos=pos)

def pos_indices(pos):
 n=pos.shape[0]; counts=np.diff(pos.indptr); maxk=int(counts.max())
 idx=np.zeros((n,maxk),dtype='int64'); valid=np.zeros((n,maxk),dtype='bool')
 for i in range(n):
  v=pos.indices[pos.indptr[i]:pos.indptr[i+1]];idx[i,:len(v)]=v;valid[i,:len(v)]=True
 return torch.from_numpy(idx).cuda(),torch.from_numpy(valid).cuda()

def candidate_pool(raw, cap=128):
 n=raw[0].shape[0]; affinity=sum(normalized_adj(a) for a in raw).tocsr()
 idx=np.zeros((n,cap),dtype='int64'); val=np.zeros((n,cap),dtype='float32'); mask=np.zeros((n,cap),dtype='bool')
 for i in range(n):
  lo,hi=affinity.indptr[i:i+2];ids=affinity.indices[lo:hi];vs=affinity.data[lo:hi]
  keep=ids!=i;ids=ids[keep];vs=vs[keep]
  order=np.argsort(-vs,kind='stable')[:cap];ids=ids[order];vs=vs[order]
  idx[i,:len(ids)]=ids;val[i,:len(ids)]=vs;mask[i,:len(ids)]=True
 return [torch.from_numpy(x).cuda() for x in (idx,val,mask)]

def supplied_pool(pos):
 p=pos.copy();p.setdiag(0);p.eliminate_zeros();idx,valid=pos_indices(p)
 return idx,torch.ones_like(idx,dtype=torch.float32),valid

def static_positive(data, mode='structure', k=16):
 idx,aff,valid=candidate_pool(data['raw']);n=len(idx)
 score=aff.clone()
 if mode in ['cosine','combined'] and data['feats'][0] is not None:
  x=data['feats'][0].to_dense();x=F.normalize(x,dim=1)
  cosine=[]
  for lo in range(0,n,128): cosine.append((x[lo:lo+128,None,:]*x[idx[lo:lo+128]]).sum(-1))
  cosine=torch.cat(cosine);score=cosine if mode=='cosine' else cosine*aff
 elif mode=='cosine':
  # Identity features give zero off-diagonal cosine: do not silently call arbitrary ties semantic neighbors.
  eye=sp.eye(n,format='csr');return pos_indices(eye)
 score=score.masked_fill(~valid,float('-inf'));sel=score.topk(min(k,score.shape[1]),dim=1).indices
 ids=idx.gather(1,sel);mask=valid.gather(1,sel)
 return torch.cat([torch.arange(n,device='cuda')[:,None],ids],1),torch.cat([torch.ones(n,1,device='cuda',dtype=torch.bool),mask],1)

class MassContrast(nn.Module):
 def __init__(self,dim,tau,block=0):
  super().__init__();self.tau=tau;self.block=block
  self.proj=nn.Sequential(nn.Linear(dim,dim),nn.ELU(),nn.Linear(dim,dim))
  for m in self.proj:
   if isinstance(m,nn.Linear):nn.init.xavier_normal_(m.weight,gain=1.414)
 def direction(self,a,b,idx,mask):
  a=F.normalize(a,dim=1);b=F.normalize(b,dim=1);n=len(a);bs=self.block or n
  def part(aa,bb,ii,mm):
   logits=(aa@bb.T)/self.tau
   positive=logits.gather(1,ii).masked_fill(~mm,float('-inf'))
   return (torch.logsumexp(logits,1)-torch.logsumexp(positive,1)).sum()
  total=0
  for lo in range(0,n,bs):
   args=(a[lo:lo+bs],b,idx[lo:lo+bs],mask[lo:lo+bs])
   total=total+(checkpoint(part,*args,use_reentrant=False) if self.block and self.training else part(*args))
  return total/n
 def pair(self,a,b,idx,mask):
  a=self.proj(a);b=self.proj(b)
  return self.direction(a,b,idx,mask)+self.direction(b,a,idx,mask)

class BlockContraNorm(ContraNorm):
 def __init__(self,dim,block=0):
  super().__init__(dim=dim,scale=.1,temp=.8,dual_norm=False,pre_norm=False,learnable=False);self.block=block
 def forward(self,x):
  if not self.block:return super().forward(x)
  xn=F.normalize(x,dim=1)
  def part(q,k,v):return F.softmax(q@k.T/self.temp,dim=1)@v
  chunks=[]
  for lo in range(0,len(x),self.block):
   args=(xn[lo:lo+self.block],xn,x)
   chunks.append(checkpoint(part,*args,use_reentrant=False) if self.training else part(*args))
  return self.layernorm(x-self.scale*torch.cat(chunks))

class Model(nn.Module):
 def __init__(self,data,variant='full',dim=64,block=0,lam=.5):
  super().__init__();cfg=data['cfg'];self.variant=variant;self.cfg=cfg;self.lam=lam;self.dim=dim
  self.fc=nn.ModuleList([nn.Linear(d,dim) for d in data['dims']])
  for m in self.fc:nn.init.xavier_normal_(m.weight,gain=1.414)
  self.drop=nn.Dropout(cfg['drop'])
  if variant=='heco':
   sys.path.insert(0,str(ROOT/'baselines/HeCo-main/code'))
   # Loaded as a distinct package to avoid collisions with original module names.
   from heco_impl.mp_encoder import Mp_encoder as HeCoMp
   from heco_impl.sc_encoder import Sc_encoder
   self.mp=HeCoMp(len(data['mps']),dim,cfg['attn']);self.ng=Sc_encoder(dim,cfg['sample'],len(data['neis']),cfg['attn'])
  else:
   self.mp=Mp_encoder(len(data['mps']),dim,cfg['attn']);self.ng=Ng_encoder(dim,cfg['sample'],len(data['neis']),cfg['attn'])
  self.sm=Sm_encoder(dim,cfg['types'][0],1,.2,.2)
  self.cn=BlockContraNorm(dim,block);self.ln=nn.LayerNorm(dim,eps=1e-6)
  self.contrast=MassContrast(dim,cfg['tau'],block)
  if variant=='mean_anchor':
   # Change only the semantic summary; preserve architecture, initial tensors,
   # three-view objective, normalization, and all stochastic operations.
   import types
   def mean_attention(attention,embeds):
    current=attention.attn_drop(attention.att)
    scores=torch.cat([(attention.tanh(attention.fc(x))@current.T).mean(0) for x in embeds]).view(-1)
    weights=F.softmax(scores,dim=-1)
    return sum(x*weights[i] for i,x in enumerate(embeds))
   self.mp.att.forward=types.MethodType(mean_attention,self.mp.att)
 def project(self,feats,drop=True):
  hs=[]
  for x,fc in zip(feats,self.fc):
   h=fc.weight.T+fc.bias if x is None else torch.sparse.mm(x,fc.weight.T)+fc.bias
   hs.append(F.elu(self.drop(h) if drop else h))
  return hs
 def views(self,data):
  h=self.project(data['feats']);mp=self.mp(h[0],data['mps'])
  if self.variant=='mp_only':return mp,self.mp(self.project(data['feats'])[0],data['mps']),None
  if self.variant=='mp_aug':
   h2=add_gaussian_noise(dropout_feat(self.project(data['feats'])[0],.2),std=.1)
   noisy=[]
   for a in data['mps']:
    a=a.coalesce();ii,vv=edge_drop(a.indices(),a.values(),.2);noisy.append(torch.sparse_coo_tensor(ii,vv,a.shape,device=a.device).coalesce())
   return mp,self.mp(h2,noisy),None
  if self.variant=='no_global':
   self.ng.simplified_gate.forward=lambda local,global_context:local
  ng=self.ng(h,data['neis'])
  sm=None if self.variant in ['heco','mp_ng'] else self.sm(h[0],data['mps'])
  if self.variant not in ['no_cn','heco','mp_sm','mp_only']:
   ng=self.ln(ng) if self.variant=='layernorm' else self.cn(ng)
  return mp,ng,sm
 def forward(self,data,idx,mask):
  a,b,c=self.views(data)
  if self.variant=='heco':return .5*self.contrast.pair(a,b,idx,mask)
  if self.variant in ['mp_only','mp_aug','mp_ng']:return self.contrast.pair(a,b,idx,mask)
  if self.variant=='mp_sm':return self.contrast.pair(a,c,idx,mask)
  return self.lam*self.contrast.pair(a,b,idx,mask)+(1-self.lam)*self.contrast.pair(a,c,idx,mask)
 @torch.no_grad()
 def embeds(self,data):
  # The deployed representation requires only the target projection and MP
  # encoder. Do not execute discarded auxiliary-type projections.
  x=data['feats'][0];fc=self.fc[0]
  h=fc.weight.T+fc.bias if x is None else torch.sparse.mm(x,fc.weight.T)+fc.bias
  return self.mp(F.elu(h),data['mps']).detach()

@torch.no_grad()
def refine_positive(model,data,pool,k=16):
 was=model.training;model.eval()
 npstate=np.random.get_state();tstate=torch.get_rng_state();cstate=torch.cuda.get_rng_state_all()
 saved=(model.sm.pf,model.sm.pe,model.sm.noise_std)
 np.random.seed(20261010);model.sm.pf=0.;model.sm.pe=0.;model.sm.noise_std=0.
 a,b,c=model.views(data);z=[F.normalize(x,dim=1) for x in [a,b,c]]
 model.sm.pf,model.sm.pe,model.sm.noise_std=saved
 np.random.set_state(npstate);torch.set_rng_state(tstate);torch.cuda.set_rng_state_all(cstate)
 idx,aff,valid=pool;n=len(idx);scores=[]
 for lo in range(0,n,128):
  agree=torch.ones_like(aff[lo:lo+128])
  for x in z:agree*=((x[lo:lo+128,None,:]*x[idx[lo:lo+128]]).sum(-1).clamp(min=0))
  scores.append(agree.pow(1/3)*aff[lo:lo+128])
 score=torch.cat(scores).masked_fill(~valid,float('-inf'));sel=score.topk(min(k,score.shape[1]),dim=1).indices
 out=(torch.cat([torch.arange(n,device='cuda')[:,None],idx.gather(1,sel)],1),torch.cat([torch.ones(n,1,device='cuda',dtype=torch.bool),valid.gather(1,sel)],1))
 model.train(was);return out

def probe(z,y,split,seed=0,repeats=5,epochs=200):
 """Independent batched linear probes; choose one epoch by validation CE."""
 z=z.detach();n,d=z.shape;c=int(y.max())+1;y=torch.from_numpy(y).long().cuda()
 tr,va,te=[torch.as_tensor(split[s],device='cuda') for s in ['train','val','test']]
 torch.manual_seed(seed+10000)
 w=nn.Parameter(torch.empty(repeats,d,c,device='cuda'));b=nn.Parameter(torch.zeros(repeats,1,c,device='cuda'))
 nn.init.xavier_uniform_(w);opt=torch.optim.Adam([w,b],lr=.01,weight_decay=0.)
 best=torch.full((repeats,),float('inf'),device='cuda');preds=[None]*repeats;probs=[None]*repeats;bestepoch=[None]*repeats;valpred=[None]*repeats
 for epoch in range(epochs):
  opt.zero_grad();logits=torch.einsum('nd,rdc->rnc',z[tr],w)+b
  loss=F.cross_entropy(logits.reshape(-1,c),y[tr].repeat(repeats));loss.backward();opt.step()
  with torch.no_grad():
   vl=torch.einsum('nd,rdc->rnc',z[va],w)+b
   score=F.cross_entropy(vl.reshape(-1,c),y[va].repeat(repeats),reduction='none').reshape(repeats,-1).mean(1)
   improved=score<best;best=torch.minimum(best,score)
   if improved.any():
    tl=torch.einsum('nd,rdc->rnc',z[te],w)+b;pp=F.softmax(tl,dim=-1).cpu().numpy()
    for r in improved.nonzero().ravel().tolist():
     probs[r]=pp[r];preds[r]=pp[r].argmax(-1);bestepoch[r]=epoch;valpred[r]=vl[r].argmax(-1).cpu().numpy()
 yy=y[te].cpu().numpy();metrics=[]
 for pred,p in zip(preds,probs):
  acc=float((pred==yy).mean());micro=f1_score(yy,pred,average='micro');assert abs(acc-micro)<1e-12
  metrics.append(dict(acc=acc,micro=micro,macro=f1_score(yy,pred,average='macro'),auc=roc_auc_score(yy,p,multi_class='ovr')))
 return {'mean':{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},'sd':{k:float(np.std([m[k] for m in metrics],ddof=1)) for k in metrics[0]},'individual':metrics,'selected_epochs':bestepoch,'validation_macro':float(np.mean([f1_score(y[va].cpu().numpy(),v,average='macro') for v in valpred])),'validation_ce':float(best.mean())}

def cluster(z,y,seed):
 z=z.detach().cpu().numpy();out=[]
 for k in range(3):
  pred=KMeans(n_clusters=len(np.unique(y)),random_state=seed*100+k,n_init=10).fit_predict(z)
  out.append(dict(nmi=normalized_mutual_info_score(y,pred),ari=adjusted_rand_score(y,pred)))
 return {'mean':{k:float(np.mean([m[k] for m in out])) for k in out[0]},'individual':out}

def run(args):
 seed_all(args.seed);t=time.perf_counter();data=load_dataset(args.dataset)
 pool=None
 if args.positive=='supplied':idx,mask=pos_indices(data['pos'])
 elif args.positive in ['refine_supplied','self_added','random_supplied']:
  base=(data['pos']+sp.eye(len(data['y']),format='csr')).astype('bool').astype('float32')
  idx,mask=pos_indices(base)
  if args.positive in ['refine_supplied','random_supplied']:pool=supplied_pool(data['pos'])
  if args.positive=='random_supplied':
   pass # Same warm-up targets as refinement; random target updates start below.
 elif args.positive=='self':idx,mask=pos_indices(sp.eye(len(data['y']),format='csr'))
 else:idx,mask=static_positive(data,'structure' if args.positive=='refine' else args.positive,args.k)
 if args.positive=='refine':pool=candidate_pool(data['raw'])
 pre=time.perf_counter()-t;model=Model(data,args.variant,block=args.block,lam=args.lam).cuda()
 opt=torch.optim.Adam(model.parameters(),lr=data['cfg']['lr']*args.lrscale,weight_decay=0.)
 out=ROOT/'results';out.mkdir(exist_ok=True)
 key=f'{args.dataset}__{args.variant}__{args.positive}__k{args.k}__lam{args.lam}__lr{args.lrscale}__seed{args.seed}'
 path=out/(key+'.json')
 if path.exists() and not args.force:print('EXISTS',key,flush=True);return
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();best=float('inf');beststate=None;wait=0;bestepoch=0;history=[]
 for epoch in range(args.epochs):
  if pool is not None and epoch>=args.warmup and (epoch-args.warmup)%args.refresh==0:
   if args.positive=='random_supplied':
    ci,aff,valid=pool;generator=torch.Generator(device='cuda');generator.manual_seed(args.seed*100000+epoch)
    score=torch.rand(ci.shape,device='cuda',generator=generator).masked_fill(~valid,float('-inf'));sel=score.topk(min(args.k,ci.shape[1]),dim=1).indices
    idx=torch.cat([torch.arange(len(ci),device='cuda')[:,None],ci.gather(1,sel)],1);mask=torch.cat([torch.ones(len(ci),1,device='cuda',dtype=torch.bool),valid.gather(1,sel)],1)
   else:idx,mask=refine_positive(model,data,pool,args.k)
   # Positive targets change; reset loss-based stopping across objective phases.
   best=float('inf');wait=0
  model.train();opt.zero_grad();loss=model(data,idx,mask)
  if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
  lv=loss.item();history.append(lv)
  if lv<best:
   best=lv;bestepoch=epoch;wait=0;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else:wait+=1
  loss.backward();opt.step()
  if epoch%25==0:print(key,'epoch',epoch,'loss',round(lv,5),flush=True)
  if epoch+1>=args.min_epochs and wait>=args.patience:break
 torch.cuda.synchronize();train_seconds=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()/2**20
 model.load_state_dict(beststate);model.eval();torch.cuda.synchronize();t=time.perf_counter();z=model.embeds(data);torch.cuda.synchronize();infer=time.perf_counter()-t
 eval_t=time.perf_counter();classification={k:probe(z,data['y'],s,args.seed,args.probes,args.probe_epochs) for k,s in data['splits'].items()}
 clustering=cluster(z,data['y'],args.seed)
 ii=idx.cpu().numpy();mm=mask.cpu().numpy();rows=np.broadcast_to(np.arange(len(ii))[:,None],ii.shape);off=mm&(ii!=rows)
 diagnostics=dict(valid_positive_mean=float(mm.sum(1).mean()),valid_positive_min=int(mm.sum(1).min()),valid_positive_max=int(mm.sum(1).max()),self_pairs=int((mm&(ii==rows)).sum()),offdiagonal_label_agreement=float((data['y'][ii[off]]==data['y'][rows[off]]).mean()) if off.any() else None,label_use='post-training diagnostic only; never used for selection or encoder training')
 result=dict(key=key,args=vars(args),config=data['cfg'],classification=classification,clustering=clustering,epochs=epoch+1,best_epoch=bestepoch,best_loss=best,preprocess_seconds=pre,train_seconds=train_seconds,inference_seconds=infer,evaluation_seconds=time.perf_counter()-eval_t,peak_training_mib=peak,parameters=sum(p.numel() for p in model.parameters()),history=history,gate_sigmoid=float(model.ng.simplified_gate.gate.sigmoid().detach()) if args.variant!='heco' else None,positive_diagnostics=diagnostics,environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda))
 path.write_text(json.dumps(result,indent=2));np.savez_compressed(out/(key+'.npz'),embedding=z.cpu().numpy(),positive_indices=ii,positive_mask=mm)
 print('DONE',key,json.dumps({k:v['mean'] for k,v in classification.items()}),clustering['mean'],round(train_seconds,2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=CONFIG,default='acm');p.add_argument('--variant',default='full');p.add_argument('--positive',default='supplied');p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=400);p.add_argument('--min-epochs',type=int,default=60);p.add_argument('--patience',type=int,default=40);p.add_argument('--probes',type=int,default=5);p.add_argument('--probe-epochs',type=int,default=200);p.add_argument('--block',type=int,default=0);p.add_argument('--k',type=int,default=16);p.add_argument('--warmup',type=int,default=60);p.add_argument('--refresh',type=int,default=40);p.add_argument('--lam',type=float,default=.5);p.add_argument('--lrscale',type=float,default=1.);p.add_argument('--force',action='store_true');run(p.parse_args())
