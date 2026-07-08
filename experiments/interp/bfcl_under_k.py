from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import urllib.request

ROOT = os.environ.get("POC_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "sasori/src")]

import torch  

MODEL = os.environ.get("BF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

KVALS = [int(k.strip()) for k in os.environ.get("BF_K", "0,2,3").split(",") if k.strip() != ""]
N = int(os.environ.get("BF_N", "40"))
MAX_NEW_TOKENS = int(os.environ.get("BF_MAXNEW", "256"))
GROUP = int(os.environ.get("BF_GROUP", "256"))                       
REF_GATE = float(os.environ.get("BF_REF_GATE", "0.5"))              
CATEGORY = os.environ.get("BF_CATEGORY", "simple")
PIP_INSTALL = os.environ.get("BF_PIP_INSTALL", "1") == "1"
HF_BASE = os.environ.get(
    "BF_HF_BASE",
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main",
)
SEED = int(os.environ.get("BF_SEED", "0"))
DUMP = int(os.environ.get("BF_DUMP", "3"))
TAG = os.environ.get("BF_TAG", "bfcl_under_k")
OUT_JSON = os.environ.get("BF_OUT_JSON") or os.path.join(ROOT, "sasori/bench", f"{TAG}.json")

_SKIP_QUANT = ("lm_head", "embed_tokens")
_ROUTER_LEAVES = ("gate", "router", "gating")
_EXPERT_LEAVES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")

_TYPE_MAP = {"dict": "object", "float": "number", "double": "number", "integer": "integer",
             "tuple": "array", "array": "array", "list": "array", "string": "string",
             "boolean": "boolean", "bool": "boolean", "number": "number", "object": "object",
             "any": "string"}

def _fn(name, desc, props, required):
    return {"name": name, "description": desc,
            "parameters": {"type": "dict", "properties": props, "required": required}}

def _case(cid, fn, q, gt):
    return {"id": cid, "function": [fn], "question": [[{"role": "user", "content": q}]],
            "ground_truth": [gt]}

_BFCL_BUILTIN = [
    _case("builtin_0",
          _fn("calculate_triangle_area", "Calculate the area of a triangle given its base and height.",
              {"base": {"type": "integer", "description": "The base of the triangle."},
               "height": {"type": "integer", "description": "The height of the triangle."},
               "unit": {"type": "string", "description": "Unit of measure (defaults to 'units')."}},
              ["base", "height"]),
          "Find the area of a triangle with a base of 10 units and a height of 5 units.",
          {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}),
    _case("builtin_1",
          _fn("math.factorial", "Calculate the factorial of a given number.",
              {"number": {"type": "integer", "description": "The number to take the factorial of."}},
              ["number"]),
          "Calculate the factorial of 5.",
          {"math.factorial": {"number": [5]}}),
    _case("builtin_2",
          _fn("convert_currency", "Convert an amount of money from one currency to another.",
              {"amount": {"type": "number", "description": "The amount to convert."},
               "from_currency": {"type": "string", "description": "The ISO code to convert from."},
               "to_currency": {"type": "string", "description": "The ISO code to convert to."}},
              ["amount", "from_currency", "to_currency"]),
          "Convert 100 USD to EUR.",
          {"convert_currency": {"amount": [100], "from_currency": ["USD"], "to_currency": ["EUR"]}}),
    _case("builtin_3",
          _fn("get_weather", "Get the current weather for a city.",
              {"city": {"type": "string", "description": "The city name."},
               "units": {"type": "string", "description": "'celsius' or 'fahrenheit' (default celsius)."}},
              ["city"]),
          "What is the current weather in Tokyo?",
          {"get_weather": {"city": ["Tokyo"], "units": ["celsius", ""]}}),
    _case("builtin_4",
          _fn("send_email", "Send an email to a recipient.",
              {"to": {"type": "string", "description": "Recipient email address."},
               "subject": {"type": "string", "description": "The subject line."},
               "body": {"type": "string", "description": "The email body."}},
              ["to", "subject", "body"]),
          "Send an email to alice@example.com with the subject 'Meeting' and the body 'See you at 3pm'.",
          {"send_email": {"to": ["alice@example.com"], "subject": ["Meeting"],
                          "body": ["See you at 3pm"]}}),
    _case("builtin_5",
          _fn("math.gcd", "Compute the greatest common divisor of two integers.",
              {"a": {"type": "integer", "description": "First integer."},
               "b": {"type": "integer", "description": "Second integer."}},
              ["a", "b"]),
          "What is the greatest common divisor of 48 and 36?",
          {"math.gcd": {"a": [48], "b": [36]}}),
    _case("builtin_6",
          _fn("book_flight", "Book a flight between two cities on a date.",
              {"origin": {"type": "string", "description": "Departure city."},
               "destination": {"type": "string", "description": "Arrival city."},
               "date": {"type": "string", "description": "Date in YYYY-MM-DD."}},
              ["origin", "destination", "date"]),
          "Book a flight from Boston to Seattle on 2025-09-15.",
          {"book_flight": {"origin": ["Boston"], "destination": ["Seattle"], "date": ["2025-09-15"]}}),
    _case("builtin_7",
          _fn("compute_bmi", "Compute Body Mass Index from weight and height.",
              {"weight_kg": {"type": "number", "description": "Weight in kilograms."},
               "height_m": {"type": "number", "description": "Height in meters."}},
              ["weight_kg", "height_m"]),
          "Compute the BMI for a person weighing 70 kg and 1.75 meters tall.",
          {"compute_bmi": {"weight_kg": [70], "height_m": [1.75]}}),
    _case("builtin_8",
          _fn("set_timer", "Set a countdown timer for a number of minutes.",
              {"minutes": {"type": "integer", "description": "Number of minutes."},
               "label": {"type": "string", "description": "Optional label for the timer."}},
              ["minutes"]),
          "Set a timer for 15 minutes.",
          {"set_timer": {"minutes": [15], "label": [""]}}),
    _case("builtin_9",
          _fn("translate_text", "Translate text into a target language.",
              {"text": {"type": "string", "description": "Text to translate."},
               "target_language": {"type": "string", "description": "Target language name."}},
              ["text", "target_language"]),
          "Translate the text 'Good morning' into French.",
          {"translate_text": {"text": ["Good morning"], "target_language": ["French"]}}),
    _case("builtin_10",
          _fn("stock_price", "Look up the latest stock price for a ticker symbol.",
              {"ticker": {"type": "string", "description": "The stock ticker symbol."}},
              ["ticker"]),
          "What is the latest stock price for AAPL?",
          {"stock_price": {"ticker": ["AAPL"]}}),
    _case("builtin_11",
          _fn("math.power", "Raise a base to an exponent.",
              {"base": {"type": "number", "description": "The base."},
               "exponent": {"type": "number", "description": "The exponent."}},
              ["base", "exponent"]),
          "Compute 2 raised to the power of 10.",
          {"math.power": {"base": [2], "exponent": [10]}}),
    _case("builtin_12",
          _fn("create_calendar_event", "Create a calendar event.",
              {"title": {"type": "string", "description": "Event title."},
               "date": {"type": "string", "description": "Date in YYYY-MM-DD."},
               "time": {"type": "string", "description": "Time in HH:MM 24h (optional)."}},
              ["title", "date"]),
          "Create a calendar event titled 'Dentist' on 2025-10-01.",
          {"create_calendar_event": {"title": ["Dentist"], "date": ["2025-10-01"], "time": [""]}}),
    _case("builtin_13",
          _fn("distance_between", "Compute the Euclidean distance between two 2-D points.",
              {"x1": {"type": "number", "description": "x of point 1."},
               "y1": {"type": "number", "description": "y of point 1."},
               "x2": {"type": "number", "description": "x of point 2."},
               "y2": {"type": "number", "description": "y of point 2."}},
              ["x1", "y1", "x2", "y2"]),
          "Compute the distance between the points (0, 0) and (3, 4).",
          {"distance_between": {"x1": [0], "y1": [0], "x2": [3], "y2": [4]}}),
    _case("builtin_14",
          _fn("search_restaurants", "Search restaurants by cuisine in a city.",
              {"city": {"type": "string", "description": "The city."},
               "cuisine": {"type": "string", "description": "Cuisine type."},
               "max_results": {"type": "integer", "description": "Max results (default 10)."}},
              ["city", "cuisine"]),
          "Find Italian restaurants in Chicago.",
          {"search_restaurants": {"city": ["Chicago"], "cuisine": ["Italian"], "max_results": ["", 10]}}),
    _case("builtin_15",
          _fn("celsius_to_fahrenheit", "Convert a temperature from Celsius to Fahrenheit.",
              {"celsius": {"type": "number", "description": "Temperature in Celsius."}},
              ["celsius"]),
          "Convert 25 degrees Celsius to Fahrenheit.",
          {"celsius_to_fahrenheit": {"celsius": [25]}}),
]

@torch.no_grad()
def fake_quant_k_(model, k_planes: int, group: int = GROUP) -> dict:
    import torch.nn as nn
    from sasori.reconstruct import quantize_matrix_k        
    st = {"linear": 0, "expert_mats": 0}
    for _name, mod in model.named_modules():
        for leaf, child in list(mod.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if leaf in _SKIP_QUANT or leaf in _ROUTER_LEAVES:
                continue
            if min(child.weight.shape) < 64:
                continue
            kp = quantize_matrix_k(child.weight.data, k_planes, group=group, row_chunk=65536)
            child.weight.data.copy_(kp.dequantize(child.weight.dtype))
            st["linear"] += 1
    for name, p in model.named_parameters():
        if p.dim() == 3 and "experts" in name and name.split(".")[-1] in _EXPERT_LEAVES:
            for e in range(p.shape[0]):
                kp = quantize_matrix_k(p.data[e], k_planes, group=group, row_chunk=65536)
                p.data[e] = kp.dequantize(p.dtype)
                st["expert_mats"] += 1
    return st

def load(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev).eval()
    m.requires_grad_(False)
    return m, tok

def _try_pip_install_bfcl() -> dict:
    if not PIP_INSTALL:
        return {"attempted": False, "reason": "BF_PIP_INSTALL=0"}
    cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", "--no-deps", "bfcl-eval"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        tail = (p.stdout + p.stderr).strip().splitlines()[-8:]
        print("[bfcl] pip install rc=%d; tail:\n  %s" % (p.returncode, "\n  ".join(tail)), flush=True)
        return {"attempted": True, "cmd": " ".join(cmd), "returncode": p.returncode, "tail": tail}
    except Exception as e:  
        print("[bfcl] pip install FAILED to run: %r (continuing to HF fetch)" % (e,), flush=True)
        return {"attempted": True, "cmd": " ".join(cmd), "error": repr(e)}

def _parse_jsonl(text: str) -> list:
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]

def _find_bfcl_data_files(category: str):
    import importlib.util
    fname = f"BFCL_v3_{category}.json"
    for pkg in ("bfcl_eval", "bfcl"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.origin:
            continue
        root = os.path.dirname(spec.origin)
        for dirpath, _dirs, files in os.walk(root):
            if fname in files and os.path.basename(dirpath) != "possible_answer":
                prompt_path = os.path.join(dirpath, fname)
                ans_path = os.path.join(dirpath, "possible_answer", fname)
                if os.path.exists(ans_path):
                    return prompt_path, ans_path
    return None

def _fetch(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "bfcl-under-k/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  
        return r.read().decode("utf-8")

def _merge_prompts_answers(prompts: list, answers: list) -> list:
    gt = {a["id"]: a.get("ground_truth") for a in answers}
    out = []
    for p in prompts:
        g = gt.get(p.get("id"))
        if g is None:
            continue
        q = dict(p)
        q["ground_truth"] = g
        out.append(q)
    return out

def load_bfcl_cases(n: int, category: str):
    install = _try_pip_install_bfcl()

    found = _find_bfcl_data_files(category)
    if found:
        try:
            prompts = _parse_jsonl(open(found[0]).read())
            answers = _parse_jsonl(open(found[1]).read())
            cases = _merge_prompts_answers(prompts, answers)
            if cases:
                take = cases if n <= 0 else cases[:n]
                print("[bfcl] using PACKAGE data %s (%d cases, using %d)"
                      % (found[0], len(cases), len(take)), flush=True)
                return take, {"source": "bfcl_eval_package", "path": found[0],
                              "category": category, "available": len(cases), "used": len(take),
                              "bfcl_install": install}
        except Exception as e:  
            print("[bfcl] package data present but unreadable: %r (trying HF)" % (e,), flush=True)

    fname = f"BFCL_v3_{category}.json"
    try:
        prompts = _parse_jsonl(_fetch(f"{HF_BASE}/{fname}"))
        answers = _parse_jsonl(_fetch(f"{HF_BASE}/possible_answer/{fname}"))
        cases = _merge_prompts_answers(prompts, answers)
        if cases:
            take = cases if n <= 0 else cases[:n]
            print("[bfcl] using HF-fetched data %s (%d cases, using %d)"
                  % (HF_BASE, len(cases), len(take)), flush=True)
            return take, {"source": "huggingface_fetch", "base": HF_BASE, "file": fname,
                          "category": category, "available": len(cases), "used": len(take),
                          "bfcl_install": install}
    except Exception as e:  
        print("[bfcl] HF fetch failed: %r (falling back to built-in BFCL-simple-style set)"
              % (e,), flush=True)

    cases = _BFCL_BUILTIN if n <= 0 else _BFCL_BUILTIN[:n]
    print("[bfcl] using BUILT-IN BFCL-simple-style set (%d cases). "
          "Real BFCL data unavailable — treat absolutes with care." % len(cases), flush=True)
    return cases, {"source": "builtin_bfcl_simple_style", "category": "simple_builtin",
                   "available": len(_BFCL_BUILTIN), "used": len(cases), "bfcl_install": install,
                   "note": "real BFCL data could not be obtained; built-in control set used"}

def _norm_schema(node):
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            out[k] = _TYPE_MAP.get(v.lower(), v)
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: _norm_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _norm_schema(v)
        else:
            out[k] = v
    return out

def _to_openai_tools(functions: list) -> list:
    return [{"type": "function",
             "function": {"name": f.get("name"), "description": f.get("description", ""),
                          "parameters": _norm_schema(f.get("parameters", {"type": "object",
                                                                          "properties": {}}))}}
            for f in functions]

def _fn_text(functions: list) -> str:
    return "\n".join(json.dumps(f, ensure_ascii=False) for f in functions)

def _apply_template_tools(tok, messages, tools):
    for kw in ({"enable_thinking": False}, {}):
        try:
            return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                           add_generation_prompt=True, **kw)
        except TypeError:
            continue  
        except Exception:  
            return None
    return None

def build_prompt(tok, messages, functions, dev):
    tools = _to_openai_tools(functions)
    prompt = _apply_template_tools(tok, messages, tools)
    mode = "native_tools"
    if prompt is None:

        instr = (
            "You are a function-calling assistant. You are given the following function(s) as "
            "JSON schemas:\n" + _fn_text(functions) + "\n\n"
            "Call exactly one function to satisfy the user's request. Respond with ONLY a single "
            "JSON object of the form {\"name\": <function name>, \"arguments\": {<arg>: <value>, ...}} "
            "and nothing else.")
        aug = [{"role": "system", "content": instr}] + list(messages)
        try:
            prompt = tok.apply_chat_template(aug, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(aug, tokenize=False, add_generation_prompt=True)
        mode = "json_instruction"
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(dev)
    return ids, mode

@torch.no_grad()
def _generate(model, tok, ids, dev):
    g = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|tool_code)?\s*(.*?)```", re.DOTALL)

def _iter_brace_objects(text: str):
    depth, start, in_str, esc = 0, -1, False, False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]

def _loads_tolerant(s: str):
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(s)
        except Exception:  
            continue
    return None

def _obj_to_call(obj, allowed_names):
    if not isinstance(obj, dict):
        return None
    if "name" in obj and isinstance(obj["name"], str):
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            args = _loads_tolerant(args) or {}
        return obj["name"], (args if isinstance(args, dict) else {})
    if len(obj) == 1:
        k, v = next(iter(obj.items()))
        if isinstance(v, dict) and (not allowed_names or k in allowed_names):
            return k, v
    return None

def _parse_python_call(text, functions):
    names = {f.get("name"): f for f in functions}
    for m in re.finditer(r"([A-Za-z_][\w.]*)\s*\(", text):
        fname = m.group(1)
        if fname not in names:
            continue
        
        depth, j, in_str, esc, q = 0, m.end() - 1, False, False, ""
        while j < len(text):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == q:
                    in_str = False
            elif c in "\"'":
                in_str, q = True, c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        snippet = text[m.start():j + 1]
        try:
            node = ast.parse(snippet, mode="eval").body
        except Exception:  
            continue
        if not isinstance(node, ast.Call):
            continue
        args = {}
        for kw in node.keywords:
            if kw.arg is not None:
                try:
                    args[kw.arg] = ast.literal_eval(kw.value)
                except Exception:  
                    pass
        if node.args:  
            order = list(names[fname].get("parameters", {}).get("properties", {}).keys())
            for idx, a in enumerate(node.args):
                if idx < len(order):
                    try:
                        args[order[idx]] = ast.literal_eval(a)
                    except Exception:  
                        pass
        return fname, args
    return None

def parse_tool_call(text, functions):
    allowed = {f.get("name") for f in functions}
    for m in _TOOLCALL_RE.finditer(text):
        obj = _loads_tolerant(m.group(1))
        call = _obj_to_call(obj, allowed)
        if call:
            return call
    for m in _FENCE_RE.finditer(text):
        for cand in _iter_brace_objects(m.group(1)):
            call = _obj_to_call(_loads_tolerant(cand), allowed)
            if call:
                return call
    for cand in _iter_brace_objects(text):
        call = _obj_to_call(_loads_tolerant(cand), allowed)
        if call:
            return call
    call = _parse_python_call(text, functions)
    if call:
        return call
    return None, None

def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None

def _eq(pred, acc) -> bool:
    if isinstance(acc, bool) or isinstance(pred, bool):
        return bool(pred) == bool(acc) if isinstance(acc, bool) and isinstance(pred, bool) else pred == acc
    if isinstance(acc, str) and acc == "":
        return isinstance(pred, str) and pred.strip() == ""      
    np, na = _num(pred), _num(acc)
    if np is not None and na is not None:
        return abs(np - na) <= 1e-6 * max(1.0, abs(na))
    if isinstance(pred, str) and isinstance(acc, str):
        return pred.strip().lower() == acc.strip().lower()
    if isinstance(acc, (list, tuple)) and isinstance(pred, (list, tuple)):
        return len(pred) == len(acc) and all(_eq(p, a) for p, a in zip(pred, acc))
    if isinstance(acc, dict) and isinstance(pred, dict):
        return set(pred) == set(acc) and all(_eq(pred[k], acc[k]) for k in acc)
    return pred == acc

def _value_ok(pred_val, acceptable) -> bool:
    return any(_eq(pred_val, a) for a in acceptable)

def _args_match(pred_args: dict, expected: dict) -> bool:
    if any(k not in expected for k in pred_args):
        return False                                             
    for arg, acceptable in expected.items():
        if arg in pred_args:
            if not _value_ok(pred_args[arg], acceptable):
                return False
        else:
            if not any((a == "" or a is None) for a in acceptable):
                return False                                     
    return True

def check_call(pred_name, pred_args, ground_truth) -> bool:
    if pred_name is None:
        return False
    for entry in ground_truth:
        for gt_name, gt_args in entry.items():
            if pred_name == gt_name and _args_match(pred_args or {}, gt_args):
                return True
    return False

def _name_matches(pred_name, ground_truth) -> bool:
    return pred_name is not None and any(pred_name in entry for entry in ground_truth)

@torch.no_grad()
def run_variant(model, tok, dev, cases, dump=0) -> dict:
    torch.manual_seed(SEED)  
    correct = name_ok = parsed = total = 0
    prompt_modes = set()
    samples = []
    for case in cases:
        total += 1
        messages = case["question"][0]                           
        functions = case["function"]
        gt = case["ground_truth"]
        ids, mode = build_prompt(tok, messages, functions, dev)
        prompt_modes.add(mode)
        resp = _generate(model, tok, ids, dev)
        pname, pargs = parse_tool_call(resp, functions)
        is_parsed = pname is not None
        is_name = _name_matches(pname, gt)
        is_ok = check_call(pname, pargs, gt)
        parsed += int(is_parsed)
        name_ok += int(is_name)
        correct += int(is_ok)
        if len(samples) < dump:
            samples.append({"id": case.get("id"),
                            "query": (messages[-1]["content"] if messages else "")[:160],
                            "response": resp[:400], "pred_name": pname, "pred_args": pargs,
                            "ground_truth": gt, "parsed": is_parsed, "name_ok": is_name,
                            "correct": is_ok, "prompt_mode": mode})
    return {
        "fc_acc": (correct / total if total else float("nan")),   
        "name_acc": (name_ok / total if total else float("nan")),
        "n": total,
        "n_parsed": parsed,
        "prompt_modes": sorted(prompt_modes),
        "samples": samples,
    }

def _label(k_planes: int) -> str:
    return "fp" if k_planes == 0 else f"k{k_planes}"

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import transformers

    cases, data_meta = load_bfcl_cases(N, CATEGORY)

    print("env: torch %s | tf %s | dev %s | MODEL %s | K %s | N %d | max_new %d | group %d | "
          "ref_gate %.2f | category %s"
          % (torch.__version__, transformers.__version__, dev, MODEL, KVALS, len(cases),
             MAX_NEW_TOKENS, GROUP, REF_GATE, CATEGORY), flush=True)
    print("bfcl data source: %s" % (data_meta,), flush=True)

    meta = {
        "experiment": "#81 function-calling (BFCL simple) under the K-lever (FP/K2/K3)",
        "model": MODEL, "kvals": KVALS, "group": GROUP, "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED, "ref_gate": REF_GATE, "category": CATEGORY,
        "data_source": data_meta,
        "metric": "fc_acc = correct function name AND correct arguments (BFCL simple AST match)",
        "checker": ("re-implementation of BFCL `simple` (single Python call) AST semantics: name "
                    "match + declared-args-only + acceptable-value lists (\"\" => omittable). Does "
                    "NOT cover Java/JS/parallel/multiple/relevance/multi-turn/executable."),
        "harness": ("own generation+parse+AST-match harness (BFCL's own runner cannot apply our "
                    "in-place fake-quant); uses authoritative BFCL cases when available, else a "
                    "built-in BFCL-simple-style set (see data_source.source)."),
        "prompting": ("native Hermes tools= chat template when supported (Qwen2.5), else a "
                      "JSON-instruction fallback; the mode actually used is in each variant's "
                      "prompt_modes."),
        "question": ("does function-calling stratify like KNOWLEDGE (robust to K2) or like "
                     "REASONING (collapses at K2, K3 recovers)?"),
    }
    R = {"_meta": meta, "reference_ok": None, "verdict": None}

    for k_planes in KVALS:
        label = _label(k_planes)
        m, tok = load(dev)
        quant = None
        if k_planes > 0:
            quant = fake_quant_k_(m, k_planes, GROUP)
            print("[%s] fake-quant reached: %s (group=%d)" % (label, quant, GROUP), flush=True)
        out = run_variant(m, tok, dev, cases, dump=DUMP)
        out["quant"] = quant
        R[label] = out
        print("VARIANT %-3s fc_acc %.3f | name_acc %.3f | parsed %d/%d | modes %s"
              % (label, out["fc_acc"], out["name_acc"], out["n_parsed"], out["n"],
                 out["prompt_modes"]), flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        json.dump(R, open(OUT_JSON, "w"), indent=2)

    fp = R.get("fp")
    if fp is None:
        R["reference_ok"] = None
        R["verdict"] = "NO_FP_REFERENCE"     
        print("\n[GATE] no FP arm in BF_K -> cannot validate the reference; verdict=NO_FP_REFERENCE",
              flush=True)
    else:
        ref_ok = fp["fc_acc"] >= REF_GATE
        R["reference_ok"] = bool(ref_ok)
        R["verdict"] = "OK" if ref_ok else "NEEDS_VALIDATION"
        if not ref_ok:
            print("\n[GATE] FP fc_acc %.3f < gate %.2f -> reference_ok=False, "
                  "verdict=NEEDS_VALIDATION. The FP model does not call functions well enough "
                  "(or the harness/parser under-fires): the K comparison is NOT trustworthy and "
                  "should NOT be reported until this is resolved." % (fp["fc_acc"], REF_GATE),
                  flush=True)
        else:
            print("\n[GATE] FP fc_acc %.3f >= gate %.2f -> reference_ok=True, verdict=OK"
                  % (fp["fc_acc"], REF_GATE), flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(R, open(OUT_JSON, "w"), indent=2)

    print("\n=== SUMMARY %s (N=%d, category=%s) verdict=%s ==="
          % (MODEL, len(cases), CATEGORY, R["verdict"]), flush=True)
    for k_planes in KVALS:
        label = _label(k_planes)
        v = R.get(label)
        if v:
            print("  %-3s fc_acc %.3f | name_acc %.3f | parsed %d/%d"
                  % (label, v["fc_acc"], v["name_acc"], v["n_parsed"], v["n"]), flush=True)
    print("WROTE", OUT_JSON, flush=True)

if __name__ == "__main__":
    main()
