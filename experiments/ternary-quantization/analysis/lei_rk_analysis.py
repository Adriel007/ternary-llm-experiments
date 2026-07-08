
import json, os, math
from statistics import NormalDist

_HERE = os.path.dirname(os.path.abspath(__file__))
LAD_CANDS = [
    os.path.join(_HERE, '..', 'data', 'sqnr_kladder.jsonl'),
]
ladder = {}  
for c in LAD_CANDS:
    if os.path.exists(c):
        for l in open(c):
            if not l.strip():
                continue
            r = json.loads(l)
            ladder.setdefault(r['model'].split('/')[-1], {})[int(r['K'])] = float(r['sqnr_late'])

PARTIAL = {
    'Qwen2.5-7B-Instruct': {2: 8.18, 3: 18.49, 4: 25.33, 5: 28.60},
    'Qwen3-8B':            {2: 2.67, 3: 11.50, 4: 18.72, 5: 23.28},
}
for m, d in PARTIAL.items():
    ladder.setdefault(m, {})
    for k, v in d.items():
        ladder[m].setdefault(k, v)

RET = {
    'Qwen2.5-7B-Instruct': {
        'gsm8k':   {2: 0.175, 3: 0.984, 4: 1.07},     
        'math500': {2: 0.00, 3: 1.05, 4: 0.84, 5: 0.84},
    },
    'Qwen3-8B': {
        'gsm8k':   {2: 0.027, 3: 0.956},
        'math500': {2: 0.03, 3: 0.74, 4: 0.91, 5: 0.95},
    },
}

def linfit(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
    den = n*sxx - sx*sx
    b = (n*sxy - sx*sy)/den
    a = (sy - b*sx)/n
    ybar = sy/n
    ss_tot = sum((y-ybar)**2 for y in ys)
    ss_res = sum((y-(a+b*x))**2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    return a, b, r2

print('='*72)
print('LEI R(K) — análise de linearidade SQNR-late(K) [math]')
print('='*72)
for m in sorted(ladder):
    d = ladder[m]
    Ks = sorted(d)
    if len(Ks) < 2:
        print(f'{m}: só {len(Ks)} ponto(s), pula'); continue
    xs = Ks; ys = [d[k] for k in Ks]
    a, g, r2 = linfit(xs, ys)
    incs = [(Ks[i+1], round(d[Ks[i+1]]-d[Ks[i]], 2)) for i in range(len(Ks)-1)]
    
    peff = [round(10**(-inc/20), 3) for _, inc in incs]
    print(f'\n{m}: SQNR(K)={ {k: d[k] for k in Ks} }')
    print(f'   fit linear: a={a:.2f} dB, gamma={g:.2f} dB/plano, R^2={r2:.4f}')
    print(f'   incrementos por plano (dB): {[i for _,i in incs]}')
    print(f'   p_eff por plano (residuo_k+1/residuo_k): {peff}  '
          f'(geometrico exige p constante)')
    if len(incs) >= 2:
        trend = incs[0][1]-incs[-1][1]
        print(f'   Delta-incremento {incs[0][1]} -> {incs[-1][1]} = {trend:+.2f} dB '
              f'({"DECRESCENTE (concavo, retornos decrescentes)" if trend>0.5 else "CRESCENTE" if trend<-0.5 else "~constante (linear)"})')

print('\n'+'='*72)
print('PROBIT R(K)=Phi((SQNR(K)-theta)/sigma): theta por dificuldade de benchmark')
print('='*72)
N = NormalDist()
for m in sorted(ladder):
    if m not in RET:
        continue
    d = ladder[m]
    for bench in ('gsm8k', 'math500'):
        if bench not in RET[m]:
            continue
        
        pts = [(d[k], RET[m][bench][k], k) for k in sorted(d) if k in RET[m][bench]]
        if len(pts) < 2:
            continue
        
        below = [p for p in pts if p[1] < 0.5]
        above = [p for p in pts if p[1] >= 0.5]
        if below and above:
            lo = max(below, key=lambda p: p[0]); hi = min(above, key=lambda p: p[0])
            theta_lo, theta_hi = lo[0], hi[0]
            print(f'{m:22s} {bench:8s}: limiar theta_50% em ({theta_lo:.1f}, {theta_hi:.1f}] dB '
                  f'[K{lo[2]} ret={lo[1]:.0%} -> K{hi[2]} ret={hi[1]:.0%}]')
        else:
            print(f'{m:22s} {bench:8s}: sem transição nos K medidos (pts={[(round(s,1),round(r,2)) for s,r,_ in pts]})')
