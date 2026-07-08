
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import plots  

matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None
plots.plt.rcParams.update({
    "font.size": 12.5, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9.5,
    "figure.dpi": 300, "savefig.dpi": 300,
})

OUT = os.path.join(ROOT, "papers", "paper-e-allocation")
D = os.path.join(ROOT, "reports", "data")
L = lambda n: json.load(open(os.path.join(D, n)))

plots.make_mp(L("mp.json"), OUT)                                                  
plots.make_h2alloc(L("h2_alloc.json"), OUT)                                       
plots.make_h2_2b_hardened(L("h2_2b_hardening.json"), OUT)                         
plots.make_h2_2b_greedy(L("h2_2b_greedy.json"), L("h2_2b_hardening.json"), OUT)   
plots.make_h2_2b_eap_alloc(L("h2_2b_eap_alloc.json"), OUT)                        
plots.make_h2_downstream_named(L("h2_2b_downstream_named.json"), OUT)             
plots.make_h2_2b_cheapgreedy(L("h2_2b_cheapgreedy.json"),
                             L("h2_2b_cheapgreedy_jointwarm.json"), OUT)          
plots.make_h2_2b_domain(L("h2_2b_crossdomain.json"),
                        L("h2_2b_traindomain_fineweb.json"),
                        L("h2_2b_traindomain_mix.json"), OUT)                     
plots.make_granularity(L("h2_2b_granularity.json"), OUT)                          
plots.make_gate(L("h2_2b_gate.json"), OUT)                                        

import plot_backlog
plot_backlog.FIGDIR = os.path.join(OUT, "figures")
plot_backlog.fig51_robust(L("h2_2b_robust.json"))
print("OK paper-E: 11 figuras regeneradas (10 plots.py + fig51) sem banner, fontes maiores, bug fig51 corrigido.")
