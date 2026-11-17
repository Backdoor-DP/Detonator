"""Detonator: server-side detection of backdoored LoRA updates by amplifying each client's own update.

For one client update the server builds W(Gamma) = W_g + Gamma * dW, where dW is the client's LoRA increment over
the round's global adapter W_g and every direction is scaled alike.  Gamma runs over {1.5, 2.0, ..., 6.0}.  At each
Gamma the amplified model answers N public probe questions (greedy, first line only), and the new tokens of every
answer are those absent from the global model's answer to the same probe, from the probe itself, and from the digit
and whitespace tokens.  The score is

    A(Gamma) = max over m >= 2 of |K(m)| * m / N,   K(m) = largest token set shared by the new tokens of m probes,
               with |K(m)| >= k_min,

and the server checks, at every Gamma in increasing order:

    more than half of the answers collapse   -> benign, stop
    A(Gamma) > k_min / 2                     -> malicious, stop
    no decision up to Gamma_max              -> benign
"""
import collections
import re
import time

import torch
from peft import set_peft_model_state_dict

from data_utils import get_qa_dataset

MAX_NEW_TOKENS = 48
DEGEN_MIN_TOKENS = 8      # token-level degeneration is judged on answers with at least this many content tokens
DEGEN_MAX_FREQ = 0.5      # most frequent token share at or above this -> collapsed
DEGEN_MAX_TTR = 0.4       # type-token ratio at or below this -> collapsed


# ----------------------------------------------------------------------------- amplification
class AmpHooks:
    """Forward hooks on the LoRA-adapted projections.  The peft model holds the global adapter, so each module
    computes W0 + dW_g; the hook adds Gamma * (dW_client - dW_g), giving W_g + Gamma * dW."""

    def __init__(self, model, modules=("q_proj", "v_proj")):
        self.lins = [(i, p, getattr(layer.self_attn, p))
                     for i, layer in enumerate(model.base_model.model.model.layers) for p in modules]
        self.on, self.gamma, self.F = False, 1.0, None
        for idx, (_, _, lin) in enumerate(self.lins):
            lin.register_forward_hook(self._mk(idx))

    def _mk(self, idx):
        def hook(mod, inp, out):
            if not self.on:
                return out
            x = inp[0].to(torch.float32)
            ac, bc, ar, br = self.F[idx]
            inc = (x @ ac.T) @ bc.T - (x @ ar.T) @ br.T
            return (out.to(torch.float32) + self.gamma * inc).to(out.dtype)
        return hook

    def load(self, global_sd, client_sd, scaling, prefix="base_model.model.model.layers.%d.self_attn.%s."):
        self.F = []
        for i, p, _ in self.lins:
            pre = prefix % (i, p)
            get = lambda sd, k: sd[pre + k].float().cuda()
            self.F.append((get(client_sd, "lora_A.weight"), get(client_sd, "lora_B.weight") * scaling,
                           get(global_sd, "lora_A.weight"), get(global_sd, "lora_B.weight") * scaling))


# ----------------------------------------------------------------------------- decision rule
def _junk(t):
    return any(c.isdigit() for c in t) or t.replace("▁", "").replace("Ġ", "").strip() == ""


def _norm(t):
    return t.replace("▁", "").replace("Ġ", "").lower()


def _largest_shared(inc, m, init=(0, ())):
    """Largest token set shared by the same >= m probes (branch and bound); init is K(m + 1)."""
    items = sorted([(t, b) for t, b in inc.items() if bin(b).count("1") >= m], key=lambda x: bin(x[1]).count("1"))
    best, nodes = [init[0], init[1]], [0]

    def dfs(size, cand, chosen):
        nodes[0] += 1
        if size > best[0]:
            best[0], best[1] = size, chosen
        if nodes[0] > 200000:
            return
        for i, (t, b) in enumerate(cand):
            if size + len(cand) - i <= best[0]:
                return
            dfs(size + 1, [(t2, b & b2) for t2, b2 in cand[i + 1:] if bin(b & b2).count("1") >= m], chosen + (t,))
    dfs(0, items, ())
    return best[0], best[1]


def shared_block(tok, refs, outs, probes, kmin):
    """(area, m, k, tokens) of the largest block of new tokens with k >= kmin shared by m >= 2 probes."""
    toks = lambda s: set(tok.convert_ids_to_tokens(tok(s, add_special_tokens=False)["input_ids"]))
    inc = collections.defaultdict(int)
    for j, (r, o, q) in enumerate(zip(refs, outs, probes)):
        echo = {_norm(t) for t in toks(q)}
        for t in toks(o) - toks(r):
            if not _junk(t) and _norm(t) not in echo:
                inc[t] |= 1 << j
    best, prev = (0, 0, 0, ()), (0, ())
    for m in range(len(outs), 1, -1):               # K(m) only grows as m decreases
        prev = _largest_shared(inc, m, prev)
        k, block = prev
        if k >= kmin and m * k > best[0]:
            best = (m * k, m, k, block)
    return best


def collapsed(o, tok, kmin=3):
    """Repetition degeneration: one word makes up half of more than kmin words, a character repeats 6+ times, or, with
    at least DEGEN_MIN_TOKENS content tokens, the top token share >= DEGEN_MAX_FREQ or the type-token ratio <= DEGEN_MAX_TTR."""
    w = re.findall(r"\w+", o.lower())
    if (len(w) > kmin and collections.Counter(w).most_common(1)[0][1] / len(w) >= 0.5) or re.search(r"(.)\1{5,}", o):
        return True
    t = [x for x in tok.convert_ids_to_tokens(tok(o, add_special_tokens=False)["input_ids"]) if not _junk(x)]
    if len(t) < DEGEN_MIN_TOKENS:
        return False
    c = collections.Counter(t)
    return c.most_common(1)[0][1] / len(t) >= DEGEN_MAX_FREQ or len(c) / len(t) <= DEGEN_MAX_TTR


@torch.no_grad()
def generate(model, tok, prompts, batch=40):
    side, tok.padding_side = tok.padding_side, "left"
    res = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id)
        res += [re.split(r"\n|###", tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(), maxsplit=1)[0]
                for x in out]
    tok.padding_side = side
    return res


def probe_set(dataset, data_root, n):
    """The first n test questions of a public QA set in the training prompt, without a trailing space."""
    te = get_qa_dataset(dataset, split="test", prompt_mode="train", data_root=data_root)
    return [te[i]["instruction"].rstrip() for i in range(n)]


# ----------------------------------------------------------------------------- per-round detector
class DetonatorDetector:
    def __init__(self, scaling, probe_dataset="freebaseqa", data_root=None, n=40, gammas=None, kmin=3,
                 modules=("q_proj", "v_proj")):
        self.scaling, self.modules, self.kmin = scaling, modules, kmin
        self.gammas = gammas or [1.5 + 0.5 * i for i in range(10)]
        self.probes = probe_set(probe_dataset, data_root, n)
        self.hooks = None

    def scan(self, model, tok, refs):
        steps = []
        self.hooks.on = True
        verdict = "benign"
        for g in self.gammas:
            self.hooks.gamma = g
            outs = generate(model, tok, self.probes)
            area, m, k, block = shared_block(tok, refs, outs, self.probes, self.kmin)
            score, col = area / len(outs), sum(collapsed(o, tok, self.kmin) for o in outs) / len(outs)
            steps.append(dict(gamma=g, score=score, m=m, k=k, block=list(block), collapse=col, outs=outs))
            if col > 0.5:
                verdict = "benign-collapse@x%g" % g
                break
            if score > self.kmin / 2:
                verdict = "malicious@x%g" % g
                break
        self.hooks.on = False
        return verdict, steps

    def detect_round(self, model, tok, local_dicts, global_dict, rnd):
        if self.hooks is None:
            self.hooks = AmpHooks(model, self.modules)
        was_training, use_cache = model.training, model.config.use_cache
        model.eval()
        model.config.use_cache = True
        set_peft_model_state_dict(model, global_dict)
        refs = generate(model, tok, self.probes)
        rec = dict(round=rnd, flagged=[], clients={}, refs=refs)
        for c, sd in local_dicts.items():
            t0 = time.time()
            self.hooks.load(global_dict, sd, self.scaling)
            verdict, steps = self.scan(model, tok, refs)
            rec["clients"][c] = dict(verdict=verdict, steps=steps, secs=time.time() - t0)
            if verdict.startswith("malicious"):
                rec["flagged"].append(c)
            print(">> [Detonator] round %d client %d -> %s (%.0fs)" % (rnd + 1, c, verdict, time.time() - t0), flush=True)
        model.config.use_cache = use_cache
        if was_training:
            model.train()
        torch.cuda.empty_cache()
        return rec["flagged"], rec
