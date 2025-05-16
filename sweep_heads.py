MAX_SEQ_LEN = 300
NUM_LINES_UNFILTERED = 5000
NUM_LINES = 2000
K = 100

from transformer_lens import HookedTransformer
import torch as t
from huggingface_hub import login
from contextlib import contextmanager
import json
import torch.nn.functional as F
from tqdm.notebook import tqdm
import gc
from IPython.display import display, HTML
import html as html_escaper
import copy
import os

from utils import *

# login("")
t.set_grad_enabled(False)

model_base = HookedTransformer.from_pretrained_no_processing("gemma-2-2b", device="cpu")
model_chat = HookedTransformer.from_pretrained_no_processing("gemma-2-2b-it", device="cuda")
model_hs = copy.deepcopy(model_chat)
clear_mem()

with open("2b_it_generations_deduped.jsonl", "r") as f:
    data = [json.loads(line)['conversation'] for i, line in enumerate(f) if i < NUM_LINES_UNFILTERED]
toks = model_base.to_tokens(data, prepend_bos=False)[:, :MAX_SEQ_LEN]
# Filter out sequences containing padding tokens (token ID 0)
toks = t.stack([seq for seq in toks if t.all(seq != 0)])[:NUM_LINES]


for layer in range(model_base.cfg.n_layers):
    for head in range(model_base.cfg.n_heads):

        print(f"Layer {layer}, head {head}")

        # Comoute stuff
        kl_hs, kl_ablated = get_kl(model_base, model_hs, model_chat, toks, layer, head, batch_size=16, verbose=False)
        top_vals_hs, top_inds_hs = matrix_topk(kl_hs, k=K)
        top_vals_ablated, top_inds_ablated = matrix_topk(kl_ablated, k=K)
        mean_hs = kl_hs.mean().item()
        mean_ablated = kl_ablated.mean().item()

        # Save results
        kl_histogram(kl_hs, layer, head, save_dir=f"results/histograms/hs/L{layer}H{head}.png")
        kl_histogram(kl_ablated, layer, head, save_dir=f"results/histograms/ablated/L{layer}H{head}.png")
        with open(f"results/topk_data/hs/L{layer}H{head}.json", "w") as f:
            json.dump({
                "vals": top_vals_hs.tolist(), 
                "inds": top_inds_hs.tolist(),
                "mean": mean_hs,
            }, f)
        with open(f"results/topk_data/ablated/L{layer}H{head}.json", "w") as f:
            json.dump({
                "vals": top_vals_ablated.tolist(), 
                "inds": top_inds_ablated.tolist(),
                "mean": mean_ablated,
            }, f)

        clear_mem()