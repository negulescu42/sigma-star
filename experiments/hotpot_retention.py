#!/usr/bin/env python
"""Characterize HotpotQA examples RETAINED vs DROPPED by the 512-token
supporting-fact-survival filter used in hotpot_cert.py. Selection-bias audit.
Same tokenizer, MAXLEN, and retention rule as the main experiment."""
import json, re, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

SEED=0; MAXLEN=512
tok=AutoTokenizer.from_pretrained('bert-base-uncased')
ds=load_dataset('hotpot_qa','distractor',split='validation')
# same ordering/scope as the experiment: it oversampled N_EVAL*OVERSAMPLE=3600 then kept first 300.
# For a representative selection-bias audit we replay the retention decision over the SAME 3600-index pool.
idx=np.random.RandomState(SEED).permutation(len(ds))[:300*12]

def norm(s): return re.sub(r'\s+',' ',s.strip().lower())

def features_and_retention(ex):
    q=ex['question']; ans=ex['answer']
    titles=ex['context']['title']; sents=ex['context']['sentences']
    supp=set(zip(ex['supporting_facts']['title'], ex['supporting_facts']['sent_id']))
    ctx=''; sent_spans=[]
    for t,slist in zip(titles,sents):
        for si,s in enumerate(slist):
            st=len(ctx); ctx+=s; en=len(ctx)
            sent_spans.append((st,en,(t,si) in supp))
    yn = ans.lower() in ('yes','no')
    enc=tok(q, ctx, truncation='only_second', max_length=MAXLEN,
            return_offsets_mapping=True, return_tensors='pt')
    off=enc['offset_mapping'][0].tolist(); seqids=enc.sequence_ids(0)
    n_ctx_tok=sum(1 for s in seqids if s==1)
    # answer-span survival (span answers)
    ans_ok=True
    if not yn:
        pos=ctx.lower().find(ans.lower())
        if pos<0: ans_ok=False
        else:
            aend=pos+len(ans); asl=ael=None
            for i,(o,sq) in enumerate(zip(off,seqids)):
                if sq!=1 or o[1]<=o[0]: continue
                if o[0]<=pos<o[1] and asl is None: asl=i
                if o[0]<aend<=o[1]: ael=i
            if asl is None or ael is None or ael<asl: ans_ok=False
    # supporting-sentence survival
    supp_present=set()
    for i,(o,sq) in enumerate(zip(off,seqids)):
        if sq!=1: continue
        c=o[0]
        for k,(st,en,iss) in enumerate(sent_spans):
            if st<=c<en:
                if iss: supp_present.add(k)
                break
    n_supp_total=sum(1 for (_,_,iss) in sent_spans if iss)
    retained = ans_ok and (len(supp_present)>=n_supp_total)
    feats=dict(
        retained=bool(retained),
        yesno=bool(yn),
        n_supp=int(n_supp_total),
        ctx_chars=len(ctx),
        ctx_tok_full=int(len(tok(ctx, add_special_tokens=False)['input_ids'])),
        ctx_tok_trunc=int(n_ctx_tok),
        truncated=bool(len(tok(q,ctx,add_special_tokens=True)['input_ids'])>MAXLEN),
        q_type=ex['type'],  # 'comparison' or 'bridge'
        level=ex['level'],  # 'easy'/'medium'/'hard'
        ans_len_words=len(ans.split()),
    )
    return feats

rows=[features_and_retention(ds[int(i)]) for i in idx]
import pandas as pd
df=pd.DataFrame(rows)
def grp(g):
    return dict(
        n=int(len(g)),
        yesno_frac=float(g.yesno.mean()),
        n_supp_mean=float(g.n_supp.mean()),
        ctx_tok_full_mean=float(g.ctx_tok_full.mean()),
        ctx_tok_full_median=float(g.ctx_tok_full.median()),
        truncated_frac=float(g.truncated.mean()),
        comparison_frac=float((g.q_type=='comparison').mean()),
        hard_frac=float((g.level=='hard').mean()),
        ans_len_mean=float(g.ans_len_words.mean()),
    )
out=dict(pool=int(len(df)),
         retained=grp(df[df.retained]),
         dropped=grp(df[~df.retained]),
         retention_frac=float(df.retained.mean()))
json.dump(out, open('hotpot_retention.json','w'), indent=2)
print(json.dumps(out,indent=2))
