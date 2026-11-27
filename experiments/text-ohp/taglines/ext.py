import pandas as pd, numpy as np, torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(1)
M=sys.argv[1]; tag=sys.argv[2]
d=pd.read_json('t.jsonl',lines=True)
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
Y=tok.encode("Yes",add_special_tokens=False)[0]; N=tok.encode("No",add_special_tokens=False)[0]
H=[];Z=[]
for t in d.tagline:
    m=[{"role":"user","content":f'SaaS landing page tagline: "{t}"\nIs this a high-quality tagline: clear, differentiated, memorable, and about value rather than just a product category? Answer Yes or No.'}]
    p=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
    b=tok(p,return_tensors='pt')
    with torch.no_grad(): o=model.model(**b,output_hidden_states=True)
    hs=o.hidden_states  # layers+1; last one is post-norm
    H.append(torch.stack([h[0,-1] for h in hs]).float().numpy())
    W=model.lm_head.weight.float(); Z.append(float(o.last_hidden_state[0,-1].float()@(W[Y]-W[N])))
np.savez(f'f_{tag}.npz',H=np.stack(H),z=np.array(Z),q=d.quality.values)
print('done',tag,np.stack(H).shape)
