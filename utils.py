import torch as t
import gc
import torch.nn.functional as F
from IPython.display import display, HTML
import html as html_escaper
from contextlib import contextmanager
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def clear_mem():
    gc.collect()
    t.cuda.empty_cache()


def logits_to_kl(logits_src, logits_tgt):
    logprobs_src = F.log_softmax(logits_src.to(t.float32), dim=-1)
    logprobs_tgt = F.log_softmax(logits_tgt.to(t.float32), dim=-1)
    return F.kl_div(logprobs_src, logprobs_tgt, log_target=True, reduction='none').sum(-1).to(t.bfloat16) # shape [batch, seq_len]

def matrix_topk(m, k=10):
    topk_vals, topk_indices = t.topk(m.flatten(), k)
    topk_batch_indices = topk_indices // m.shape[1]
    topk_seq_indices = topk_indices % m.shape[1]
    topk_indices = t.stack((topk_batch_indices, topk_seq_indices), dim=1)
    return topk_vals, topk_indices


@contextmanager
def swap_head_pair(model_base, model_hs, model_chat, layer, head,  ablate=False):
    '''
    In Q and O we swap (layer, 2*head) and (layer, 2*head+1) from the chat model with their base counterparts.
    In K and V we swap (layer, head) from the chat model with its base counterpart.
    '''
    # Store the original weights
    original_W_Q_0 = model_hs.blocks[layer].attn.W_Q.data[2*head].clone()
    original_W_Q_1 = model_hs.blocks[layer].attn.W_Q.data[2*head+1].clone()
    original_W_O_0 = model_hs.blocks[layer].attn.W_O.data[2*head].clone()
    original_W_O_1 = model_hs.blocks[layer].attn.W_O.data[2*head+1].clone()
    
    original_W_K = model_hs.blocks[layer].attn._W_K.data[head].clone()
    original_W_V = model_hs.blocks[layer].attn._W_V.data[head].clone()

    try:
        # Swap in weights from the base model, or zeros if ablate is set to True
        if ablate:
            # Ablate the pair of heads
            model_hs.blocks[layer].attn.W_O.data[2*head] = t.zeros_like(model_base.blocks[layer].attn.W_O.data[2*head])
            model_hs.blocks[layer].attn.W_O.data[2*head+1] = t.zeros_like(model_base.blocks[layer].attn.W_O.data[2*head+1])
        else:
            # For Q and O we swap the pair of heads
            model_hs.blocks[layer].attn.W_Q.data[2*head] = model_base.blocks[layer].attn.W_Q.data[2*head].clone().cuda() 
            model_hs.blocks[layer].attn.W_Q.data[2*head+1] = model_base.blocks[layer].attn.W_Q.data[2*head+1].clone().cuda()
            model_hs.blocks[layer].attn.W_O.data[2*head] = model_base.blocks[layer].attn.W_O.data[2*head].clone().cuda()
            model_hs.blocks[layer].attn.W_O.data[2*head+1] = model_base.blocks[layer].attn.W_O.data[2*head+1].clone().cuda()

            # For K and V we swap single head
            model_hs.blocks[layer].attn._W_K.data[head] = model_base.blocks[layer].attn._W_K.data[head].clone().cuda()
            model_hs.blocks[layer].attn._W_V.data[head] = model_base.blocks[layer].attn._W_V.data[head].clone().cuda()

        # Make sure no other layers have a nonzero W_Q diff
        for l in range(model_chat.cfg.n_layers):
            diff_norm = (model_chat.W_Q[l] - model_hs.W_Q[l]).norm()
            if l != layer:
                assert diff_norm < 1e-5, f"Swapping layer {layer}, but layer {l} has nonzero W_Q diff: {diff_norm}"
            
        # Enter the context with swapped weights
        yield
        
    finally:
        # Restore the original weights
        model_hs.blocks[layer].attn.W_Q.data[2*head] = original_W_Q_0.clone()
        model_hs.blocks[layer].attn.W_Q.data[2*head+1] = original_W_Q_1.clone()
        model_hs.blocks[layer].attn.W_O.data[2*head] = original_W_O_0.clone()
        model_hs.blocks[layer].attn.W_O.data[2*head+1] = original_W_O_1.clone()

        model_hs.blocks[layer].attn._W_K.data[head] = original_W_K.clone()
        model_hs.blocks[layer].attn._W_V.data[head] = original_W_V.clone()
        clear_mem()


@contextmanager
def swap_head(model_base, model_hs, model_chat, layer, head, verbose=False, ablate=False, swap_kv=False):
    # Store the original weights
    original_W_Q = model_hs.blocks[layer].attn.W_Q.data[head].clone()
    original_W_K = model_hs.blocks[layer].attn._W_K.data[head // 2].clone() # head indexing is due to GQA fuckery
    original_W_V = model_hs.blocks[layer].attn._W_V.data[head // 2].clone() # head indexing is due to GQA fuckery
    original_W_O = model_hs.blocks[layer].attn.W_O.data[head].clone()
    try:
        # Swap in weights from the base model, or zeros if ablate is set to True
        if ablate:
            # model_hs.blocks[layer].attn._W_V.data[head // 2] = t.zeros_like(model_base.blocks[layer].attn._W_V.data[head // 2])
            model_hs.blocks[layer].attn.W_O.data[head] = t.zeros_like(model_base.blocks[layer].attn.W_O.data[head])
        else:
            model_hs.blocks[layer].attn.W_Q.data[head] = model_base.blocks[layer].attn.W_Q.data[head].clone().cuda() 
            model_hs.blocks[layer].attn.W_O.data[head] = model_base.blocks[layer].attn.W_O.data[head].clone().cuda()
            if swap_kv:
                model_hs.blocks[layer].attn._W_K.data[head // 2] = model_base.blocks[layer].attn._W_K.data[head // 2].clone().cuda()
                model_hs.blocks[layer].attn._W_V.data[head // 2] = model_base.blocks[layer].attn._W_V.data[head // 2].clone().cuda()
        
        if verbose:
            print(f"Swapped layer {layer} head {head}")
            for l in range(model_chat.cfg.n_layers):
                diff_norm = (model_chat.W_Q[l] - model_hs.W_Q[l]).norm()
                if diff_norm > 1e-5:
                    print(f"layer {l} has nonzero W_Q diff: {diff_norm}")
        # Enter the context with swapped weights
        yield
        
    finally:
        # Restore the original weights
        model_hs.blocks[layer].attn.W_Q.data[head] = original_W_Q.clone()
        model_hs.blocks[layer].attn._W_K.data[head // 2] = original_W_K.clone()
        model_hs.blocks[layer].attn._W_V.data[head // 2] = original_W_V.clone()
        model_hs.blocks[layer].attn.W_O.data[head] = original_W_O.clone()
        clear_mem()



def get_kl(model_base, model_hs, model_chat, toks, layer, head, batch_size=4, verbose=True):
    num_batches = toks.shape[0] // batch_size
    kl_hs_chat = []
    kl_ablated_chat = []

    for i in tqdm(range(num_batches), disable=not verbose):
        batch_toks = toks[i*batch_size : (i+1)*batch_size]

        logits_chat = model_chat(batch_toks)

        with swap_head(model_base, model_hs, model_chat, layer, head):
            logits_hs = model_hs(batch_toks)
            kl_hs_chat.append(logits_to_kl(logits_hs, logits_chat).cpu())
            del logits_hs
            clear_mem()

        with swap_head(model_base, model_hs, model_chat, layer, head, ablate=True):
            logits_ablated = model_hs(batch_toks)
            kl_ablated_chat.append(logits_to_kl(logits_ablated, logits_chat).cpu())   
            del logits_ablated
            clear_mem()

    # Concatenate KLs from all batches, and return
    return t.cat(kl_hs_chat), t.cat(kl_ablated_chat)


def kl_histogram(kl, layer, head, save_dir: str):
    plt.figure(figsize=(10, 6))
    plt.hist(kl.flatten().float().cpu().numpy(), bins=50)
    plt.yscale('log')
    plt.title(f"KL distribution: L{layer} H{head}")
    plt.xlabel("KL")
    plt.ylabel("count")
    plt.savefig(save_dir)
    plt.close()


def highlight_kl(model_base, model_hs, model_chat, toks_context, layer, head, ablate=False, cscale=0.05, n_top_toks=3):
    if toks_context.dim() == 1:
        toks_context = toks_context.unsqueeze(0)
    # toks_context has shape [1, seq_len]
    
    logprobs_chat = model_chat(toks_context).to(t.float32).log_softmax(dim=-1).squeeze(0)
    with swap_head(model_base, model_hs, model_chat, layer, head, ablate=ablate):
        logprobs_hs = model_hs(toks_context).to(t.float32).log_softmax(dim=-1).squeeze(0)
    clear_mem()

    kl = F.kl_div(logprobs_chat, logprobs_hs, log_target=True, reduction='none').sum(-1).to(t.bfloat16) # shape [seq_len]
    probs_hs = t.exp(logprobs_hs).to(t.bfloat16) # shape [seq_len] 
    probs_chat = t.exp(logprobs_chat).to(t.bfloat16) # shape [seq_len] 
    
    target_tok_ids = toks_context[:, 1:]
    str_tokens_list = [model_chat.tokenizer.decode(tok) for tok in target_tok_ids[0]] # Decode target tokens

    # Find the index and value of the maximum KL divergence
    max_kl_value, max_kl_index = t.max(kl, dim=0) if kl.numel() > 0 else (None, None)

    html = []
    # Add the first token (input[0]) without highlighting
    first_token_str = model_chat.tokenizer.decode(toks_context[0, 0])
    safe_first_token_str = first_token_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html.append(safe_first_token_str)

    for i, token_str in enumerate(str_tokens_list):
        kl_item = kl[i].item()
        intensity = min(kl_item / cscale, 1.0); intensity = max(0.0, intensity) # Scale & clamp
        color = f'rgba(255, 0, 0, {intensity:.3f})'

        # Simple title with just the KL value
        title_text = f"KL: {kl_item:.3f}"
        safe_token_str = token_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_title_text = html_escaper.escape(title_text, quote=True)
        # Add white-space: pre-wrap; to the style to render newlines
        html.append(f'<span style="background-color: {color}; white-space: pre-wrap;" title="{safe_title_text}">{safe_token_str}</span>')

    display(HTML(''.join(html)))

    # --- Print Top Tokens for Max KL Position --- 
    if max_kl_index is not None:
        max_kl_index_int = max_kl_index.item()
        target_token_at_max_kl = str_tokens_list[max_kl_index_int]
        
        # Calculate per-token KL contributions at the max KL position
        # Contribution = P_tgt * (log P_tgt - log P_src)
        log_prob_diff = logprobs_chat[max_kl_index_int] - logprobs_hs[max_kl_index_int]
        kl_contributions = probs_chat[max_kl_index_int] * log_prob_diff # Shape: [d_vocab]
        
        # Get top N contributing tokens
        top_contributions, top_indices = kl_contributions.topk(n_top_toks, dim=-1)
        
        # Format the top contributing tokens and their contribution values
        contributors = []
        for contrib_val, token_id in zip(top_contributions, top_indices):
            token_str = model_chat.tokenizer.decode(token_id)
            # Replace newline character with literal '\n' for printing
            printable_token_str = token_str.replace('\n', '\\n')
            # Only add if contribution meets the threshold
            if contrib_val.item() >= 0.002:
                contributors.append((token_id.item(), printable_token_str, contrib_val.item())) # Store id, str, contribution

        # --- Calculate Reverse KL Contributors --- 
        # Contribution = P_src * (log P_src - log P_tgt) = P_src * (-log_prob_diff)
        reverse_kl_contributions = probs_hs[max_kl_index_int] * (-log_prob_diff) # Shape: [d_vocab]
        
        # Get top N reverse contributing tokens
        top_reverse_contributions, top_reverse_indices = reverse_kl_contributions.topk(n_top_toks, dim=-1)

        # Format the top reverse contributing tokens and their contribution values
        reverse_contributors = []
        for contrib_val, token_id in zip(top_reverse_contributions, top_reverse_indices):
            token_str = model_chat.tokenizer.decode(token_id)
            # Replace newline character with literal '\n' for printing
            printable_token_str = token_str.replace('\n', '\\n')
            # Only add if contribution meets the threshold
            if contrib_val.item() >= 0.002:
                    reverse_contributors.append((token_id.item(), printable_token_str, contrib_val.item())) # Store id, str, contribution

        src_name = "HS"

        # --- Prepare Data for Table --- 
        table_data_for_df = []
        token_ids_in_table = set([item[0] for item in contributors] + [item[0] for item in reverse_contributors])

        for token_id_int in token_ids_in_table:
            p_src_val = probs_hs[max_kl_index_int, token_id_int].item()
            p_b_val = probs_chat[max_kl_index_int, token_id_int].item()
            log_diff_val = logprobs_chat[max_kl_index_int, token_id_int].item() - logprobs_hs[max_kl_index_int, token_id_int].item()
            token_str = model_chat.tokenizer.decode(token_id_int).replace('\\n', '\\\\n')
            table_data_for_df.append({
                'Token': token_str,
                f'p_{src_name}': p_src_val,
                'p_B': p_b_val,
                'log_diff': log_diff_val # Used for sorting and styling
            })
            
        # Sort: More likely in B (log_diff > 0) first, then by token string
        table_data_for_df.sort(key=lambda x: (-x['log_diff'], x['Token']))

        # --- Display Table using Pandas DataFrame --- 
        print(f"High KL: {max_kl_value.item():.4f} at token '{target_token_at_max_kl}'\\n")

        if table_data_for_df:
            df = pd.DataFrame(table_data_for_df)

            # Define styling function
            def highlight_log_diff(val):
                if val > 0: # Positive contribution to KL (more likely in B)
                    color = 'forestgreen'
                elif val < 0: # Negative contribution to KL (less likely in B)
                    color = 'indianred'
                else:
                    color = '' # No highlight for zero
                return f'background-color: {color}'

            # Apply styling. We style the 'Token' column based on 'log_diff' values
            # To do this, we need to apply styling row-wise or have access to the 'log_diff' column
            # when styling the 'Token' column. Pandas Styler's `apply` method is suitable here.

            def style_row(row):
                styles = [''] * len(row) # Default no style
                log_diff = row['log_diff']
                if log_diff > 0:
                    styles[row.index.get_loc('Token')] = 'background-color: forestgreen'
                elif log_diff < 0:
                    styles[row.index.get_loc('Token')] = 'background-color: indianred'
                return styles

            styled_df = df.style.apply(style_row, axis=1)\
                                .format({f'p_{src_name}': "{:.3f}", 'p_B': "{:.3f}", 'log_diff': "{:.3f}"})\
                                .hide(['log_diff'], axis='columns')\
                                .set_properties(**{'text-align': 'left'})\
                                .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
            
            display(HTML(styled_df.to_html()))
                
        else:
            print("No significant KL contributors found (threshold 0.002).")