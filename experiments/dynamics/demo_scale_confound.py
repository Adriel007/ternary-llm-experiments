
import numpy as np

rng = np.random.default_rng(0)

din, dh, dout, N = 8, 16, 4, 32
X = rng.standard_normal((din, N))
Yt = rng.standard_normal((dout, N))
S1, S2 = (dh, din), (dout, dh)
n1 = dh * din

def loss(theta):
    W1 = theta[:n1].reshape(S1)
    W2 = theta[n1:].reshape(S2)
    Hd = np.maximum(W1 @ X, 0.0)          
    return 0.5 * np.mean((W2 @ Hd - Yt) ** 2)

def hess_trace(theta, eps=1e-4):
    L0 = loss(theta)
    tr = 0.0
    e = np.zeros_like(theta)
    for i in range(theta.size):
        e[i] = eps
        tr += (loss(theta + e) - 2 * L0 + loss(theta - e)) / eps ** 2
        e[i] = 0.0
    return tr

W1 = rng.standard_normal(S1) * 0.5
W2 = rng.standard_normal(S2) * 0.5

print("=" * 70)
print("PART 1 -- a function-PRESERVING reparameterization changes the raw trace")
print("=" * 70)
print(f"{'c':>6} {'loss (identical?)':>18} {'trace(H) raw':>16} {'ratio vs c=1':>14}")
base_tr = hess_trace(np.concatenate([W1.ravel(), W2.ravel()]))   
for c in [0.25, 0.5, 1.0, 2.0, 4.0]:
    W1c, W2c = W1 * c, W2 / c            
    th = np.concatenate([W1c.ravel(), W2c.ravel()])
    L = loss(th)
    tr = hess_trace(th)
    ratio = tr / base_tr
    print(f"{c:>6.2f} {L:>18.10f} {tr:>16.4f} {ratio:>14.3f}")
print("-> the loss is bit-identical on every row (same function);")
print("   the trace varies by more than an order of magnitude. Raw trace is NOT")
print("   invariant => 'larger trace' != 'sharper minimum'.\n")

print("=" * 70)
print("PART 2 -- the ternary vs FP weight norm is of the kind that inflates the trace")
print("=" * 70)

sigma = 1.0
b = sigma * np.sqrt(2 / np.pi)          
Ew2_fp = sigma ** 2                     

w = rng.standard_normal(200000) * sigma
thr = 0.7 * np.mean(np.abs(w))
p_nonzero = np.mean(np.abs(w) > thr)
Ew2_tern = p_nonzero * b ** 2
print(f"sigma={sigma}  b=E|w|={b:.4f}  nonzero fraction p={p_nonzero:.3f}")
print(f"mean square norm  FP  E[w^2]      = {Ew2_fp:.4f}")
print(f"mean square norm  ternary p*b^2   = {Ew2_tern:.4f}")
print(f"norm^2 ratio  FP / ternary        = {Ew2_fp / Ew2_tern:.3f}")
print(f"norm^2 ratio  ternary / FP        = {Ew2_tern / Ew2_fp:.3f}")
print()
print("Honest reading (anti-overclaim):")
print("- the ternary net operates at a SMALLER weight norm (~{:.0f}% of the FP norm^2)."
      .format(100 * Ew2_tern / Ew2_fp))
print("- Part 1 shows a weight-scale difference moves the raw trace by factors of this")
print("  order WITHOUT any real geometric difference.")
print("- so the reported 2.15x is CONSISTENT with both 'sharper minimum' AND 'same")
print("  minimum, smaller-norm parameterization'. The raw data does not separate them.")
print("- this does NOT prove the headline is wrong; it proves it is UNDECIDABLE without")
print("  a scale-invariant control (same-loss / filter-norm / Fisher).")
