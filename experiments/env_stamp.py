
from __future__ import annotations

import os
import subprocess
import sys
import time

def env_stamp(repo_dir: str | None = None) -> dict:
    d: dict = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "python": sys.version.split()[0]}
    try:
        import torch
        d["torch"] = torch.__version__
        d["cuda"] = torch.version.cuda
        d["gpu"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:  
        d["torch_error"] = repr(e)
    try:
        import transformers
        d["transformers"] = transformers.__version__
    except Exception as e:  
        d["transformers_error"] = repr(e)
    try:
        cwd = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=cwd,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        d["git_dirty"] = bool(dirty)
    except Exception as e:  
        d["git_error"] = repr(e)
    return d

if __name__ == "__main__":
    import json
    print(json.dumps(env_stamp(), indent=2))
