import numpy as np, sys, pandas as pd, json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
def ridge(H,y,reg):
    mu=H.mean(0); sd=H.std(0)+1e-6; C=(H-mu)/sd; yc=y-y.mean()
    return np.linalg.solve(C.T@C+reg*np.eye(H.shape[1]),C.T@yc),mu,sd
def score(m,H): w,mu,sd=m; return ((H-mu)/sd)@w
REGS=[10,100,1000,10000]
def inner_pick(Htr,ytr,layers):
    best=None; sk=StratifiedKFold(4,shuffle=True,random_state=0)
    for L in layers:
        for reg in REGS:
            s=np.zeros(len(ytr))
            for a,b in sk.split(Htr[:,L],ytr): s[b]=score(ridge(Htr[a,L],ytr[a],reg),Htr[b,L])
            auc=roc_auc_score(ytr,s)
            if best is None or auc>best[0]: best=(auc,L,reg)
    return best[1],best[2]
def run(tag,target):
    f=np.load(f'f_{tag}.npz'); H=f['H']; q=f['q']; z=f['z']
    if target=='hi_vs_lo': keep=q!=3; y=(q>=4).astype(int)
    else: keep=np.ones(len(q),bool); y=(q>=4).astype(int)
    H,y,z=H[keep],y[keep],z[keep]; nl=H.shape[1]
    rk=RepeatedStratifiedKFold(n_splits=5,n_repeats=4,random_state=1)
    r={'final':[],'any':[],'base':[]}; picks=[]
    for tr,te in rk.split(H,y):
        r['base'].append(roc_auc_score(y[te],z[te]))
        L,reg=inner_pick(H[tr],y[tr],[nl-1]); r['final'].append(roc_auc_score(y[te],score(ridge(H[tr,L],y[tr],reg),H[te,L])))
        L,reg=inner_pick(H[tr],y[tr],list(range(nl//3,nl,2))); picks.append(L); r['any'].append(roc_auc_score(y[te],score(ridge(H[tr,L],y[tr],reg),H[te,L])))
    return {k:(round(float(np.mean(v)),3),round(float(np.std(v)),3)) for k,v in r.items()}|{'n':len(y),'pos':int(y.sum()),'layers_chosen':sorted(set(picks))}
for tag in sys.argv[1:]:
    for t in ['hi_vs_lo','hi_vs_rest']: print(tag,t,run(tag,t),flush=True)
