import pandas as pd, numpy as np
from transformers import AutoTokenizer
M='HuggingFaceTB/SmolLM2-135M-Instruct'
tok=AutoTokenizer.from_pretrained(M)
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
