import numpy as np, pandas as pd, torch, json
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
d=np.load('feats.npz'); YES,NO=int(d['YES']),int(d['NO'])
Htr,ytr,ztr=d['train_H'],d['train_y'],d['train_z']; Hv,yv,zv=d['valid_H'],d['valid_y'],d['valid_z']; Hte,yte,zte=d['test_H'],d['test_y'],d['test_z']
def ridge(H,y,reg):
    mu=H.mean(0); yc=(y-y.mean())/y.std(); C=H-mu
    return np.linalg.solve(C.T@C+reg*np.eye(H.shape[1]),C.T@yc)
def acc_best(z,y,thr): return float(((z>thr)==y).mean())
res={}
res['base_auc_test']=roc_auc_score(yte,zte); res['base_auc_valid']=roc_auc_score(yv,zv)
rng=np.random.RandomState(1)
for N in [60,100,200,1000]:
    rows=[]
    for rep in range(10 if N<1000 else 1):
        idx=rng.choice(len(ytr),N,replace=False) if N<1000 else np.arange(1000)
        best=None
        for reg in [1,10,100,1000,10000]:
            w=ridge(Htr[idx],ytr[idx],reg); a=roc_auc_score(yv,Hv@w)  # alpha scales score; AUC independent of alpha direction scaling only w/ base z: use alpha grid
            for alpha in [0.1,0.3,1,3,10,30]:
                s=roc_auc_score(yv,zv+alpha*(Hv@w))
                if best is None or s>best[0]: best=(s,reg,alpha,w)
        s,reg,alpha,w=best
        zt=zte+alpha*(Hte@w)
        # threshold = median-free: choose on train subset
        zs=ztr[idx]+alpha*(Htr[idx]@w); thr=np.median(zs)
        rows.append((roc_auc_score(yte,zt),acc_best(zt,yte,thr),roc_auc_score(yte,Hte@w),reg,alpha))
    r=np.array(rows)[:,:3]
    res[f'OHP_N{N}']={'test_auc_mean':float(r[:,0].mean()),'test_auc_min':float(r[:,0].min()),'test_auc_max':float(r[:,0].max()),'test_acc_mean':float(r[:,1].mean()),'ridge_only_auc':float(r[:,2].mean()),'reps':len(rows)}
# baselines: tabular logistic regression on raw columns, and logistic on hidden feats
tr=pd.read_parquet('/tmp/lf/train.parquet').loc[d['train_idx']]; va=pd.read_parquet('/tmp/lf/valid.parquet'); te=pd.read_parquet('/tmp/lf/test.parquet')
cols=['industry','region','employee_band','estimated_revenue_band','process_maturity_band','role_function','seniority','buyer_role','lead_source','touch_count','inbound_touch_count','outbound_touch_count','session_count','pricing_page_views','demo_page_views','total_session_duration_seconds','touches_days_0_7','days_since_first_touch','days_since_last_touch']
def X(df): 
    x=df[cols].copy()
    for c in cols:
        if x[c].dtype==object or str(x[c].dtype).startswith('string'): x[c]=x[c].astype(str)
        else: x[c]=x[c].astype(float)
    return x
al=pd.concat([X(tr),X(va),X(te)]); al=pd.get_dummies(al,dtype=float).fillna(-1)
a,b,c=al.iloc[:len(tr)].values,al.iloc[len(tr):len(tr)+len(va)].values,al.iloc[len(tr)+len(va):].values
for N in [100,1000]:
    g=GradientBoostingClassifier(random_state=0).fit(a[:N],ytr[:N]); res[f'tabular_GBM_N{N}_auc']=roc_auc_score(yte,g.predict_proba(c)[:,1])
    m=LogisticRegression(max_iter=2000,C=0.1).fit((a[:N]-a[:N].mean(0))/(a[:N].std(0)+1e-9),ytr[:N]); res[f'tabular_logreg_N{N}_auc']=roc_auc_score(yte,m.predict_proba((c-a[:N].mean(0))/(a[:N].std(0)+1e-9))[:,1])
    l=LogisticRegression(max_iter=3000,C=0.01).fit(Htr[:N],ytr[:N]); res[f'hidden_logreg_N{N}_auc']=roc_auc_score(yte,l.decision_function(Hte))
# identity check on real model with untied lm_head
from transformers import AutoTokenizer, AutoModelForCausalLM
M="HuggingFaceTB/SmolLM2-135M-Instruct"; model=AutoModelForCausalLM.from_pretrained(M).eval()
print('tied',model.config.tie_word_embeddings)
w=ridge(Htr[:100],ytr[:100],100.0); alpha=1.0
W0=model.lm_head.weight.detach().clone(); emb0=model.get_input_embeddings().weight.detach().clone()
model.lm_head.weight=torch.nn.Parameter(W0.clone())  # untie: copy
with torch.no_grad():
    model.lm_head.weight[YES]+=alpha*torch.tensor(w,dtype=torch.float32)/2; model.lm_head.weight[NO]-=alpha*torch.tensor(w,dtype=torch.float32)/2
chg=(model.lm_head.weight!=W0).any(1).sum().item(); emb_same=bool(torch.equal(model.get_input_embeddings().weight,emb0))
tok=AutoTokenizer.from_pretrained(M)
import feats_text as ft
res['rows_changed']=chg; res['input_embeddings_unchanged']=emb_same
errs=[]
for i in range(8):
    p=ft.prompt(te.iloc[i]); b=tok(p,return_tensors='pt')
    with torch.no_grad(): o=model(**b).logits[0,-1]
    zp=(o[YES]-o[NO]).item(); pred=zte[i]+alpha*float(Hte[i]@w); errs.append(abs(zp-pred))
res['identity_max_abs_err_8_prompts']=max(errs)
res['n_params_changed']=2*len(w)
print(json.dumps(res,indent=1,default=float))
json.dump(res,open("results.json","w"),indent=1,default=float)
