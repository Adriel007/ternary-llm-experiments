import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__))
rows=[json.loads(l) for l in open(os.path.join(HERE,'cpu_speed_1.5b_2026-06-30.jsonl')) if l.strip()]
def short(m): return m.split('/')[-1].replace('Qwen2.5-1.5B-Instruct-','').replace('qwen1.5b.','sasori-').replace('.gguf','')
G=defaultdict(dict)
for r in rows:
    G[(short(r['model_filename']),r['n_threads'])]['pp' if r['n_gen']==0 else 'tg']=r['avg_ts']
TH=[1,2,4,8,16]
series=[('Q4_K_M','#d9544d','o'),('sasori-tq3p','#3a64d9','s'),('sasori-tq2p','#7aa0e8','^'),('Q2_K','#888','x')]

fig,ax=plt.subplots(2,1,figsize=(6.6,7.8))
for kind,axi,title in [('tg',0,'decode (single-stream tok/s)'),('pp',1,'prefill pp512 (throughput tok/s)')]:
    for name,c,mk in series:
        y=[G[(name,t)].get(kind,0) for t in TH]
        ax[axi].plot(TH,y,mk+'-',color=c,label=name,lw=2,ms=8)
    ax[axi].set_title(title,fontsize=13)
    ax[axi].set_xlabel('threads',fontsize=12); ax[axi].set_ylabel('tok/s',fontsize=12)
    ax[axi].set_xticks(TH); ax[axi].tick_params(labelsize=11); ax[axi].grid(alpha=.25)
ax[0].legend(fontsize=11)

fig.tight_layout()
PAPER=os.path.join(HERE,'..','..','papers','paper-f-sasori-systems','figures','cpu_speed.png')
for out in [os.path.join(HERE,'cpu_speed_1.5b_2026-06-30.png'), PAPER]:
    fig.savefig(out,dpi=300); print('wrote',out)
