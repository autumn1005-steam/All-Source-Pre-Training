import json, time
import torch
from torch import nn
from experiment import MassContrast,BlockContraNorm,seed_all
seed_all(123)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
n,d=71,64
idx=torch.stack([torch.arange(n),torch.roll(torch.arange(n),1),torch.roll(torch.arange(n),2)],1).cuda()
mask=torch.ones_like(idx,dtype=torch.bool)
a=torch.randn(n,d,device='cuda',requires_grad=True);b=torch.randn(n,d,device='cuda',requires_grad=True)
c=MassContrast(d,.8,0).cuda();ch=MassContrast(d,.8,13).cuda();ch.load_state_dict(c.state_dict())
l=c.pair(a,b,idx,mask);g=torch.autograd.grad(l,(a,b)+tuple(c.parameters()))
ll=ch.pair(a,b,idx,mask);gg=torch.autograd.grad(ll,(a,b)+tuple(ch.parameters()))
errors={'contrast_value_absolute':abs(float(l-ll)),'contrast_gradient_max_absolute':max(float((x-y).abs().max()) for x,y in zip(g,gg))}
x=torch.randn(n,d,device='cuda',requires_grad=True);cn=BlockContraNorm(d,0).cuda();cb=BlockContraNorm(d,13).cuda();cb.load_state_dict(cn.state_dict())
target=torch.randn(n,d,device='cuda');u=cn(x);v=cb(x)
ga=torch.autograd.grad((u*target).sum(),x)[0];gb=torch.autograd.grad((v*target).sum(),x)[0]
errors.update(contranorm_value_max_absolute=float((u-v).abs().max()),contranorm_gradient_max_absolute=float((ga-gb).abs().max()))
assert errors['contrast_gradient_max_absolute']<1e-5 and errors['contranorm_gradient_max_absolute']<1e-5
from pathlib import Path
Path('results/numerical_equivalence.json').write_text(json.dumps(errors,indent=2));print(errors)
