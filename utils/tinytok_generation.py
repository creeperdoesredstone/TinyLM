import torch
import re
from collections import Counter

def generate(
    model,
    tokenizer,
    prompt,
    device,
    length=40,
    temperature=0.8,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.05,
    freq_penalty=0.2,
):
    model.eval()

    prompt_ids = tokenizer.tokenize(prompt)
    ids = list(prompt_ids)

    # --- NEW: Locate special tokens in BPE ---
    # We tokenize the strings to find their resulting IDs.
    # Note: BPE might split these if they weren't frequent in training, 
    # but for common control tokens, they usually result in a single ID.
    def get_token_id(text):
        t_ids = tokenizer.tokenize(text)
        return t_ids[0] if len(t_ids) == 1 else None

    stop_tokens = [get_token_id("<end>")]
    stop_tokens = [t for t in stop_tokens if t is not None]
    
    # Tokens to suppress (preventing model from talking to itself)
    unwanted_tokens = [get_token_id("<user>")]
    unwanted_tokens = [t for t in unwanted_tokens if t is not None]

    min_len = 16

    for _ in range(length):
        # Sliding window for context (assuming 64 is your model's max)
        x = torch.tensor([ids[-64:]], device=device)

        with torch.no_grad():
            logits = model(x)[:, -1, :]

        logits = logits.detach().to("cpu")
        logits = torch.clamp(logits, -20, 20).float()

        # -------------------------
        # Repetition penalties
        # -------------------------
        counts = Counter(ids)
        for token_id, count in counts.items():
            # Ensure we don't index out of bounds of the current logits
            if token_id < logits.size(1):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

                logits[0, token_id] -= freq_penalty * count

        # -------------------------
        # Block unwanted tokens
        # -------------------------
        for tid in unwanted_tokens:
            logits[0, tid] = -float("inf")

        bot_token = get_token_id("<bot>")
        if bot_token is not None and len(ids) > len(prompt_ids):
            logits[0, bot_token] = -float("inf")

        # Prevent immediate character loops
        if len(ids) >= 3 and ids[-1] == ids[-2] == ids[-3]:
            logits[0, ids[-1]] = -float("inf")
        elif len(ids) >= 2 and ids[-1] == ids[-2]:
            logits[0, ids[-1]] -= 2.0

        # Prevent early stopping
        if len(ids) < min_len:
            for t in stop_tokens:
                logits[0, t] -= 7.0

        if len(ids) < len(prompt_ids) + 5:
            logits = logits / 0.9

        # -------------------------
        # Top-k / Top-p filtering
        # -------------------------
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1].unsqueeze(-1)] = -float("inf")

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)

        cutoff = cum_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, -float("inf"))

        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(1, sorted_indices, sorted_logits)

        last_token = ids[-1] if ids else None
        if last_token is not None:
            # encourage sentence completion after ~20 tokens
            if len(ids) > len(prompt_ids) + 20:
                for t in stop_tokens:
                    logits[0, t] += 1.5

        # -------------------------
        # Final sampling
        # -------------------------
        progress = (len(ids) - len(prompt_ids)) / length
        temp = temperature * (1.0 + 0.15 * progress)
        logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        probs[probs < 1e-5] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)

        if torch.isnan(probs).any() or probs.sum() == 0:
            probs = torch.ones_like(probs) / probs.size(-1)

        next_id = torch.multinomial(probs, 1).item()

        if len(ids) > min_len and next_id in stop_tokens:
            break

        ids.append(next_id)

    # -------------------------
    # Decode only the generated part
    # -------------------------
    response_ids = ids[len(prompt_ids):]
    text = tokenizer.decode(response_ids)

    # Clean up artifacts
    text = re.sub(r"^(u:|b:)\s*", "", text)
    for tag in ["<bos>", "<eos>", "<end>"]:
        text = text.replace(tag, "")
    
    return text.strip()