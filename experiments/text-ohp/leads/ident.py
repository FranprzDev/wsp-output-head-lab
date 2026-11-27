import numpy as np, torch, pandas as pd, feats_text as ft
from transformers import AutoTokenizer, AutoModelForCausalLM
M="HuggingFaceTB/SmolLM2-135M-Instruct"; tok=ft.tok; model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
d=np.load('feats.npz'); YES,NO=int(d['YES']),int(d['NO']); Htr,ytr=d['train_H'],d['train_y']
mu=Htr[:100].mean(0); yc=(ytr[:100]-ytr[:100].mean())/ytr[:100].std(); C=Htr[:100]-mu
w=np.linalg.solve(C.T@C+100*np.eye(576),C.T@yc); wt=torch.tensor(w,dtype=torch.float32)
te=pd.read_parquet('/tmp/lf/test.parquet'); errs=[]; shift=[]
W0=model.lm_head.weight.detach().clone(); model.lm_head.weight=torch.nn.Parameter(W0.clone())
orig=[];pat=[];hs=[]
for i in range(8):
    b=tok(ft.prompt(te.iloc[i]),return_tensors='pt')
    with torch.no_grad(): o=model(**b); h=model.model(**b).last_hidden_state[0,-1]
    orig.append((o.logits[0,-1,YES]-o.logits[0,-1,NO]).item()); hs.append(h)
with torch.no_grad():
    model.lm_head.weight[YES]+=wt/2; model.lm_head.weight[NO]-=wt/2
    for i in range(8):
        b=tok(ft.prompt(te.iloc[i]),return_tensors='pt'); o=model(**b)
        pat.append((o.logits[0,-1,YES]-o.logits[0,-1,NO]).item())
errs=[abs(pat[i]-(orig[i]+float(hs[i]@wt))) for i in range(8)]
print('max err single-prompt identity',max(errs)); print('batched-vs-single z diff', max(abs(orig[i]-float(d['test_z'][i])) for i in range(8)))
