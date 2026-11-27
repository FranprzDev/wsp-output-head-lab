import pandas as pd, numpy as np, torch, time, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(1)
M="HuggingFaceTB/SmolLM2-135M-Instruct"
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.float32).eval()
def f(v,u=""): return "unknown" if pd.isna(v) else (f"{v:.0f}{u}" if isinstance(v,(int,float,np.floating)) else str(v))
def text(r):
    return (f"B2B lead. Industry: {r.industry}. Region: {r.region}. Company size: {r.employee_band} employees, revenue {r.estimated_revenue_band}. "
    f"Process maturity: {r.process_maturity_band}. Contact: {r.role_function}, {r.seniority}, {r.buyer_role}. Source: {r.lead_source}. "
    f"Touches: {f(r.touch_count)} (inbound {f(r.inbound_touch_count)}, outbound {f(r.outbound_touch_count)}). Sessions: {f(r.session_count)}. "
    f"Pricing page views: {f(r.pricing_page_views)}. Demo page views: {f(r.demo_page_views)}. Time on site: {f(r.total_session_duration_seconds,'s')}. "
    f"Touches in first 7 days: {f(r.touches_days_0_7)}. Days since first touch: {f(r.days_since_first_touch)}. Days since last touch: {f(r.days_since_last_touch)}.")
def prompt(r):
    m=[{"role":"user","content":text(r)+"\nWill this lead convert to a customer within 90 days? Answer Yes or No."}]
    return tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
yes=[tok.encode(x,add_special_tokens=False) for x in ["Yes","No"]]; print(yes); YES,NO=yes[0][0],yes[1][0]
rng=np.random.RandomState(0)
sets={'train':pd.read_parquet('/tmp/lf/train.parquet').sample(1000,random_state=0),'valid':pd.read_parquet('/tmp/lf/valid.parquet'),'test':pd.read_parquet('/tmp/lf/test.parquet')}
tok.padding_side="left"
out={}
for k,d in sets.items():
    H=[];Z=[];t=time.time();ps=[prompt(r) for r in d.itertuples()]
    for i in range(0,len(ps),4):
        b=tok(ps[i:i+4],return_tensors="pt",padding=True)
        with torch.no_grad():
            h=model.model(**b).last_hidden_state[:,-1,:]  # post final norm
            W=model.lm_head.weight
            z=h@(W[YES]-W[NO])
        H.append(h.numpy());Z.append(z.numpy())
        if i%200==0: print(k,i,time.time()-t,flush=True)
    out[k+'_H']=np.concatenate(H);out[k+'_z']=np.concatenate(Z);out[k+'_y']=d.converted_within_90_days.astype(int).values
    out[k+'_idx']=d.index.values
np.savez('/tmp/ohp/feats.npz',**out,YES=YES,NO=NO)
print('done')
