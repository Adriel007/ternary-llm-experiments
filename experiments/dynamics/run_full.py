
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = "/content/PhD-propose"
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

from experiments.dynamics.experiment import PoCConfig, run_poc  

CFG = PoCConfig(
    seeds=(0, 1, 2),
    steps=2000,
    batch_size=16,
    seq_len=128,
    warmup=100,
    eval_every=200,
    log_every=50,
    eval_batches=20,
    n_train_tokens=6_000_000,
    n_val_tokens=300_000,
    hessian_iters=20,
    hessian_bsz=8,
    hessian_batches=2,
    pareto_random_orders=10,
    out_dir=os.path.join(ROOT, "artifacts/poc"),
)

def main() -> None:
    os.makedirs(CFG.out_dir, exist_ok=True)
    status = os.path.join(CFG.out_dir, "STATUS.txt")
    with open(status, "w") as f:
        f.write("RUNNING\n")
    t0 = time.time()
    try:
        res = run_poc(CFG)
        s = res["summary"]
        msg = f"DONE {time.time() - t0:.0f}s\n" + json.dumps(s, indent=2)
        with open(status, "w") as f:
            f.write(msg)
        print("ALL DONE in", round(time.time() - t0, 1), "s")
        print(json.dumps(s, indent=2))
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        with open(status, "w") as f:
            f.write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
